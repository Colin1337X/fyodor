#include "compute.h"
#include "matvec_spv.h"

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* This provider implements one operation, F32 matrix-vector multiplication.
 * The transformer still defines execution order and all tensor semantics in C.
 * Cached weights are immutable, device-local copies. Inputs/outputs are small
 * coherent mappings reused across calls; there is no per-token weight upload.
 * Conservative per-model limits prevent this optional path exhausting VRAM. */
#define NYA_VK_CACHE_LIMIT 256U
#define NYA_VK_CACHE_BYTES (512ULL * 1024ULL * 1024ULL)

typedef struct nya_vk_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    void *mapped;
} nya_vk_buffer;

typedef struct nya_vk_weight {
    const void *source;
    size_t rows;
    size_t columns;
    nya_vk_buffer storage;
} nya_vk_weight;

typedef struct nya_vulkan_context nya_vulkan_context;
struct nya_vulkan_context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command;
    VkFence fence;
    VkDescriptorSetLayout descriptor_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptors;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    nya_vk_buffer input;
    nya_vk_buffer output;
    nya_vk_weight weights[NYA_VK_CACHE_LIMIT];
    size_t weight_count;
    VkDeviceSize cached_bytes;
    int failed;
};

/* A poisoned context no longer runs GPU commands even though its allocations
 * remain alive until model unload. Diagnostics must report that CPU fallback. */
int nya_vulkan_active(const nya_vulkan_context *context)
{
    return context != NULL && !context->failed;
}

/* Destroy in buffer-before-memory order, even after partial construction. */
static void nya_vk_buffer_free(nya_vulkan_context *context, nya_vk_buffer *buffer)
{
    if (buffer->mapped != NULL) vkUnmapMemory(context->device, buffer->memory);
    if (buffer->buffer != VK_NULL_HANDLE) vkDestroyBuffer(context->device, buffer->buffer, NULL);
    if (buffer->memory != VK_NULL_HANDLE) vkFreeMemory(context->device, buffer->memory, NULL);
    memset(buffer, 0, sizeof(*buffer));
}

/* Host buffers require coherence, so neither flush nor invalidate operations
 * are needed. Device-only buffers require device-local memory. A device lacking
 * either memory class falls back to the CPU instead of assuming heap behavior. */
static int nya_vk_buffer_create(nya_vulkan_context *context, VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags flags, nya_vk_buffer *buffer)
{
    VkBufferCreateInfo create = {0};
    VkMemoryRequirements requirements;
    VkMemoryAllocateInfo allocate = {0};
    uint32_t memory_type;
    memset(buffer, 0, sizeof(*buffer));
    create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create.size = size;
    create.usage = usage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(context->device, &create, NULL, &buffer->buffer) != VK_SUCCESS) return -1;
    vkGetBufferMemoryRequirements(context->device, buffer->buffer, &requirements);
    for (memory_type = 0; memory_type < context->memory_properties.memoryTypeCount; ++memory_type) {
        if ((requirements.memoryTypeBits & (1U << memory_type)) != 0 &&
            (context->memory_properties.memoryTypes[memory_type].propertyFlags & flags) == flags) break;
    }
    if (memory_type == context->memory_properties.memoryTypeCount) goto fail;
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(context->device, &allocate, NULL, &buffer->memory) != VK_SUCCESS ||
        vkBindBufferMemory(context->device, buffer->buffer, buffer->memory, 0) != VK_SUCCESS) goto fail;
    buffer->size = size;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
        vkMapMemory(context->device, buffer->memory, 0, size, 0, &buffer->mapped) != VK_SUCCESS) goto fail;
    return 0;
fail:
    nya_vk_buffer_free(context, buffer);
    return -1;
}

/* Every call is synchronous and externally serialized. The previous fence has
 * completed before we reuse command storage or touch mapped input/output. */
static int nya_vk_begin(nya_vulkan_context *context)
{
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkResetCommandPool(context->device, context->command_pool, 0) != VK_SUCCESS ||
        vkBeginCommandBuffer(context->command, &begin) != VK_SUCCESS) return -1;
    return 0;
}

static int nya_vk_submit(nya_vulkan_context *context)
{
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &context->command;
    if (vkEndCommandBuffer(context->command) != VK_SUCCESS ||
        vkResetFences(context->device, 1, &context->fence) != VK_SUCCESS ||
        vkQueueSubmit(context->queue, 1, &submit, context->fence) != VK_SUCCESS ||
        vkWaitForFences(context->device, 1, &context->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        /* A device error makes all subsequent calls choose CPU. Retire pending
         * commands before callers destroy staging resources or mapped buffers. */
        context->failed = 1;
        (void)vkDeviceWaitIdle(context->device);
        return -1;
    }
    return 0;
}

/* Stage a tensor once, then retain the device-local buffer until model unload.
 * The transfer-to-compute barrier makes copied bytes visible to shader reads. */
static nya_vk_buffer *nya_vk_weights(nya_vulkan_context *context, const void *source,
    size_t rows, size_t columns, VkDeviceSize bytes)
{
    size_t index;
    nya_vk_buffer staging = {0};
    nya_vk_weight *entry;
    VkBufferCopy copy = {0};
    VkBufferMemoryBarrier barrier = {0};
    for (index = 0; index < context->weight_count; ++index) {
        entry = &context->weights[index];
        if (entry->source == source && entry->rows == rows && entry->columns == columns) return &entry->storage;
    }
    if (context->weight_count == NYA_VK_CACHE_LIMIT ||
        bytes > NYA_VK_CACHE_BYTES - context->cached_bytes) return NULL;
    entry = &context->weights[context->weight_count];
    if (nya_vk_buffer_create(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging) != 0 ||
        nya_vk_buffer_create(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &entry->storage) != 0) goto fail;
    memcpy(staging.mapped, source, (size_t)bytes);
    if (nya_vk_begin(context) != 0) goto fail;
    copy.size = bytes;
    vkCmdCopyBuffer(context->command, staging.buffer, entry->storage.buffer, 1, &copy);
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = entry->storage.buffer;
    barrier.size = bytes;
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &barrier, 0, NULL);
    if (nya_vk_submit(context) != 0) goto fail;
    nya_vk_buffer_free(context, &staging);
    entry->source = source;
    entry->rows = rows;
    entry->columns = columns;
    context->weight_count += 1;
    context->cached_bytes += bytes;
    return &entry->storage;
fail:
    nya_vk_buffer_free(context, &staging);
    nya_vk_buffer_free(context, &entry->storage);
    return NULL;
}

void nya_vulkan_free(nya_vulkan_context *context)
{
    size_t index;
    if (context == NULL) return;
    if (context->device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(context->device);
        for (index = 0; index < context->weight_count; ++index) nya_vk_buffer_free(context, &context->weights[index].storage);
        nya_vk_buffer_free(context, &context->input);
        nya_vk_buffer_free(context, &context->output);
        if (context->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context->device, context->pipeline, NULL);
        if (context->pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context->device, context->pipeline_layout, NULL);
        if (context->descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context->device, context->descriptor_pool, NULL);
        if (context->descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context->device, context->descriptor_layout, NULL);
        if (context->fence != VK_NULL_HANDLE) vkDestroyFence(context->device, context->fence, NULL);
        if (context->command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(context->device, context->command_pool, NULL);
        vkDestroyDevice(context->device, NULL);
    }
    if (context->instance != VK_NULL_HANDLE) vkDestroyInstance(context->instance, NULL);
    free(context);
}

/* Enumeration counts are bounded and checked; hotplug races return CPU fallback.
 * Prefer a discrete GPU, then integrated/other devices. Vulkan 1.0 is sufficient
 * for this kernel, with no optional shader arithmetic features required. */
nya_vulkan_context *nya_vulkan_create(void)
{
    nya_vulkan_context *context = calloc(1, sizeof(*context));
    VkApplicationInfo application = {0};
    VkInstanceCreateInfo instance = {0};
    VkPhysicalDevice devices[64];
    VkDeviceQueueCreateInfo queue_info = {0};
    VkDeviceCreateInfo device_info = {0};
    VkCommandPoolCreateInfo pool = {0};
    VkCommandBufferAllocateInfo commands = {0};
    VkFenceCreateInfo fence = {0};
    VkDescriptorSetLayoutBinding bindings[3] = {{0}};
    VkDescriptorSetLayoutCreateInfo layout = {0};
    VkDescriptorPoolSize pool_size = {0};
    VkDescriptorPoolCreateInfo descriptors = {0};
    VkDescriptorSetAllocateInfo descriptor_allocate = {0};
    VkPushConstantRange push = {0};
    VkPipelineLayoutCreateInfo pipeline_layout = {0};
    VkShaderModuleCreateInfo shader_info = {0};
    VkComputePipelineCreateInfo pipeline = {0};
    VkShaderModule shader = VK_NULL_HANDLE;
    VkExtensionProperties extensions[256];
    uint32_t extension_count = 256;
    const char *instance_extensions[1];
    const char *device_extensions[1];
    uint32_t count = 64;
    uint32_t device_index;
    uint32_t selected_queue = 0;
    int best_score = -1;
    float priority = 1.0f;
    if (context == NULL) return NULL;
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Fyodor";
    application.apiVersion = VK_API_VERSION_1_0;
    instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance.pApplicationInfo = &application;
    /* MoltenVK implementations need the portability enumeration opt-in. */
    if (vkEnumerateInstanceExtensionProperties(NULL, &extension_count, extensions) == VK_SUCCESS) {
        for (uint32_t i = 0; i < extension_count; ++i) {
            if (strcmp(extensions[i].extensionName, "VK_KHR_portability_enumeration") == 0) {
                instance_extensions[0] = "VK_KHR_portability_enumeration";
                instance.enabledExtensionCount = 1;
                instance.ppEnabledExtensionNames = instance_extensions;
                instance.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            }
        }
    }
    if (vkCreateInstance(&instance, NULL, &context->instance) != VK_SUCCESS ||
        vkEnumeratePhysicalDevices(context->instance, &count, devices) != VK_SUCCESS) goto fail;
    for (device_index = 0; device_index < count; ++device_index) {
        VkPhysicalDeviceProperties properties;
        VkQueueFamilyProperties queues[128];
        uint32_t queue_count = 0;
        int score;
        vkGetPhysicalDeviceProperties(devices[device_index], &properties);
        if (properties.limits.maxComputeWorkGroupInvocations < 64 ||
            properties.limits.maxComputeWorkGroupSize[0] < 64 ||
            properties.limits.maxComputeSharedMemorySize < 64 * sizeof(float)) continue;
        score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score <= best_score) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, NULL);
        if (queue_count > 128) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[device_index], &queue_count, queues);
        for (uint32_t i = 0; i < queue_count; ++i) {
            if (queues[i].queueCount != 0 && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                context->physical = devices[device_index];
                context->properties = properties;
                selected_queue = i;
                best_score = score;
                break;
            }
        }
    }
    if (best_score < 0) goto fail;
    vkGetPhysicalDeviceMemoryProperties(context->physical, &context->memory_properties);
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = selected_queue;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    extension_count = 256;
    if (vkEnumerateDeviceExtensionProperties(context->physical, NULL, &extension_count, extensions) == VK_SUCCESS) {
        for (uint32_t i = 0; i < extension_count; ++i) {
            if (strcmp(extensions[i].extensionName, "VK_KHR_portability_subset") == 0) {
                device_extensions[0] = "VK_KHR_portability_subset";
                device_info.enabledExtensionCount = 1;
                device_info.ppEnabledExtensionNames = device_extensions;
            }
        }
    }
    if (vkCreateDevice(context->physical, &device_info, NULL, &context->device) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(context->device, selected_queue, 0, &context->queue);
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = selected_queue;
    if (vkCreateCommandPool(context->device, &pool, NULL, &context->command_pool) != VK_SUCCESS) goto fail;
    commands.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commands.commandPool = context->command_pool;
    commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commands.commandBufferCount = 1;
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkAllocateCommandBuffers(context->device, &commands, &context->command) != VK_SUCCESS ||
        vkCreateFence(context->device, &fence, NULL, &context->fence) != VK_SUCCESS) goto fail;
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 3;
    layout.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(context->device, &layout, NULL, &context->descriptor_layout) != VK_SUCCESS) goto fail;
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;
    descriptors.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptors.maxSets = 1;
    descriptors.poolSizeCount = 1;
    descriptors.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(context->device, &descriptors, NULL, &context->descriptor_pool) != VK_SUCCESS) goto fail;
    descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate.descriptorPool = context->descriptor_pool;
    descriptor_allocate.descriptorSetCount = 1;
    descriptor_allocate.pSetLayouts = &context->descriptor_layout;
    if (vkAllocateDescriptorSets(context->device, &descriptor_allocate, &context->descriptors) != VK_SUCCESS) goto fail;
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 2 * sizeof(uint32_t);
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &context->descriptor_layout;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(context->device, &pipeline_layout, NULL, &context->pipeline_layout) != VK_SUCCESS) goto fail;
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = sizeof(nya_matvec_spv);
    shader_info.pCode = nya_matvec_spv;
    if (vkCreateShaderModule(context->device, &shader_info, NULL, &shader) != VK_SUCCESS) goto fail;
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = shader;
    pipeline.stage.pName = "main";
    pipeline.layout = context->pipeline_layout;
    if (vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &pipeline, NULL, &context->pipeline) != VK_SUCCESS) goto fail;
    vkDestroyShaderModule(context->device, shader, NULL);
    return context;
fail:
    if (shader != VK_NULL_HANDLE) vkDestroyShaderModule(context->device, shader, NULL);
    nya_vulkan_free(context);
    return NULL;
}

/* Row-major weights use uint32 indexing in SPIR-V. Check host multiplication,
 * shader indexing, descriptor range and dispatch limits before any allocation.
 * Unsupported sizes are a normal CPU fallback, never a truncated GPU dispatch. */
int nya_vulkan_matvec(nya_vulkan_context *context, const void *weights,
    size_t rows, size_t columns, const float *input, float *output)
{
    VkDeviceSize weight_bytes;
    VkDeviceSize input_bytes;
    VkDeviceSize output_bytes;
    nya_vk_buffer *weight_buffer;
    VkDescriptorBufferInfo buffers[3] = {{0}};
    VkWriteDescriptorSet writes[3] = {{0}};
    VkBufferMemoryBarrier barrier = {0};
    uint32_t shape[2];
    if (context == NULL || context->failed || weights == NULL || input == NULL || output == NULL ||
        rows == 0 || columns == 0 || rows > UINT32_MAX || columns > UINT32_MAX ||
        columns > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / sizeof(float) / columns ||
        rows > UINT32_MAX / columns || rows > context->properties.limits.maxComputeWorkGroupCount[0]) return -1;
    weight_bytes = (VkDeviceSize)rows * columns * sizeof(float);
    input_bytes = (VkDeviceSize)columns * sizeof(float);
    output_bytes = (VkDeviceSize)rows * sizeof(float);
    if (weight_bytes > context->properties.limits.maxStorageBufferRange) return -1;
    weight_buffer = nya_vk_weights(context, weights, rows, columns, weight_bytes);
    if (weight_buffer == NULL) return -1;
    if (context->input.size < input_bytes) {
        nya_vk_buffer_free(context, &context->input);
        if (nya_vk_buffer_create(context, input_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &context->input) != 0) return -1;
    }
    if (context->output.size < output_bytes) {
        nya_vk_buffer_free(context, &context->output);
        if (nya_vk_buffer_create(context, output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &context->output) != 0) return -1;
    }
    memcpy(context->input.mapped, input, (size_t)input_bytes);
    buffers[0].buffer = weight_buffer->buffer; buffers[0].range = weight_bytes;
    buffers[1].buffer = context->input.buffer; buffers[1].range = input_bytes;
    buffers[2].buffer = context->output.buffer; buffers[2].range = output_bytes;
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = context->descriptors;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(context->device, 3, writes, 0, NULL);
    if (nya_vk_begin(context) != 0) return -1;
    shape[0] = (uint32_t)rows; shape[1] = (uint32_t)columns;
    vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline);
    vkCmdBindDescriptorSets(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
        context->pipeline_layout, 0, 1, &context->descriptors, 0, NULL);
    vkCmdPushConstants(context->command, context->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shape), shape);
    vkCmdDispatch(context->command, (uint32_t)rows, 1, 1);
    /* Queue submission makes host writes available to the GPU. The reverse
     * direction needs shader->host visibility plus a completed fence before
     * memcpy reads the result. Coherence alone is not an execution barrier. */
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = context->output.buffer;
    barrier.size = output_bytes;
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &barrier, 0, NULL);
    if (nya_vk_submit(context) != 0) return -1;
    memcpy(output, context->output.mapped, (size_t)output_bytes);
    return 0;
}
