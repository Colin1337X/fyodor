/* Optional original C host adapter; included after the native Vulkan helpers. */
#include <math.h>
#include "portable_spv.h"
typedef struct nya_vk_attention_state {
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    nya_vk_buffer buffers[4];
} nya_vk_attention_state;
static void nya_vk_attention_free(nya_vulkan_context *c)
{
    nya_vk_attention_state *s=c->attention_state;
    if (!s) return;
    for (unsigned i=0;i<4;++i) nya_vk_buffer_free(c,&s->buffers[i]);
    if (s->pipeline) vkDestroyPipeline(c->device,s->pipeline,NULL);
    if (s->pipeline_layout) vkDestroyPipelineLayout(c->device,s->pipeline_layout,NULL);
    if (s->pool) vkDestroyDescriptorPool(c->device,s->pool,NULL);
    if (s->layout) vkDestroyDescriptorSetLayout(c->device,s->layout,NULL);
    free(s);c->attention_state=NULL;
}
static int nya_vk_attention_init(nya_vulkan_context *c)
{
    nya_vk_attention_state *s=calloc(1,sizeof(*s));
    if (!s) return -1;
    c->attention_state=s;
    VkDescriptorSetLayoutBinding bindings[4]={{0}};
    for (unsigned i=0;i<4;++i) {bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo li={0};li.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;li.bindingCount=4;li.pBindings=bindings;
    VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};
    VkDescriptorPoolCreateInfo pi={0};pi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;pi.maxSets=1;pi.poolSizeCount=1;pi.pPoolSizes=&ps;
    VkDescriptorSetAllocateInfo ai={0};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorSetCount=1;ai.pSetLayouts=&s->layout;
    VkPushConstantRange push={VK_SHADER_STAGE_COMPUTE_BIT,0,28};
    VkPipelineLayoutCreateInfo pli={0};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&s->layout;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&push;
    VkShaderModuleCreateInfo si={0};si.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;si.codeSize=sizeof(nya_attention_spv);si.pCode=nya_attention_spv;
    VkShaderModule shader=VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device,&li,NULL,&s->layout)!=VK_SUCCESS || vkCreateDescriptorPool(c->device,&pi,NULL,&s->pool)!=VK_SUCCESS) goto fail;
    ai.descriptorPool=s->pool;
    if (vkAllocateDescriptorSets(c->device,&ai,&s->set)!=VK_SUCCESS || vkCreatePipelineLayout(c->device,&pli,NULL,&s->pipeline_layout)!=VK_SUCCESS || vkCreateShaderModule(c->device,&si,NULL,&shader)!=VK_SUCCESS) goto fail;
    VkComputePipelineCreateInfo ci={0};ci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=shader;ci.stage.pName="main";ci.layout=s->pipeline_layout;
    if (vkCreateComputePipelines(c->device,VK_NULL_HANDLE,1,&ci,NULL,&s->pipeline)!=VK_SUCCESS) goto fail;
    vkDestroyShaderModule(c->device,shader,NULL);return 0;
fail:
    if (shader) vkDestroyShaderModule(c->device,shader,NULL);
    nya_vk_attention_free(c);return -1;
}
static int nya_vk_attention(void *opaque,const nya_compute_attention *a)
{
    nya_vulkan_context *c=opaque;
    const char *setting=getenv("NYA_VULKAN_PORTABLE_ATTENTION");
    if (!setting || strcmp(setting,"1") || !nya_vulkan_active(c) || !a || !a->query || !a->keys || !a->values || !a->output ||
        !a->batch || !a->heads || !a->kv_heads || !a->width || a->width>256 || a->heads%a->kv_heads ||
        a->capacity>UINT32_MAX || a->position>=a->capacity || a->batch>a->capacity-a->position || a->window>UINT32_MAX || !isfinite(a->scale)) return -1;
    if (a->heads>UINT32_MAX/a->width || a->batch>UINT32_MAX/(a->heads*a->width) || a->capacity>UINT32_MAX/(a->kv_heads*a->width)) return -1;
    if (a->batch>SIZE_MAX/sizeof(float)/(a->heads*a->width) ||
        a->capacity>SIZE_MAX/sizeof(float)/(a->kv_heads*a->width)) return -1;
    const VkPhysicalDeviceLimits *limits=&c->properties.limits;
    if (a->heads>limits->maxComputeWorkGroupCount[0] || a->batch>limits->maxComputeWorkGroupCount[1] || limits->maxComputeWorkGroupInvocations<256 || limits->maxComputeWorkGroupSize[0]<256 || limits->maxComputeSharedMemorySize<1292 || limits->maxPushConstantsSize<28) return -1;
    size_t qbytes=a->batch*a->heads*a->width*sizeof(float),kvbytes=(a->position+a->batch)*a->kv_heads*a->width*sizeof(float);
    size_t sizes[4]={qbytes,kvbytes,kvbytes,qbytes};
    for (unsigned i=0;i<4;++i) if (sizes[i]>limits->maxStorageBufferRange || sizes[i]>NYA_VK_CACHE_BYTES/4) return -1;
    if (!c->attention_state && nya_vk_attention_init(c)) return -1;
    nya_vk_attention_state *s=c->attention_state;
    for (unsigned i=0;i<4;++i) if (s->buffers[i].size<sizes[i]) {
        nya_vk_buffer_free(c,&s->buffers[i]);
        if (nya_vk_buffer_create(c,sizes[i],VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&s->buffers[i])) return -1;
    }
    memcpy(s->buffers[0].mapped,a->query,qbytes);memcpy(s->buffers[1].mapped,a->keys,kvbytes);memcpy(s->buffers[2].mapped,a->values,kvbytes);
    VkDescriptorBufferInfo buffers[4]={{0}};VkWriteDescriptorSet writes[4]={{0}};
    for (unsigned i=0;i<4;++i) {buffers[i].buffer=s->buffers[i].buffer;buffers[i].range=sizes[i];writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=s->set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&buffers[i];}
    vkUpdateDescriptorSets(c->device,4,writes,0,NULL);
    if (nya_vk_begin(c)) return -1;
    struct {uint32_t width,heads,kvheads,window,position,count;float scale;} shape={(uint32_t)a->width,(uint32_t)a->heads,(uint32_t)a->kv_heads,(uint32_t)a->window,(uint32_t)a->position,(uint32_t)a->batch,a->scale};
    vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,s->pipeline);
    vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,s->pipeline_layout,0,1,&s->set,0,NULL);
    vkCmdPushConstants(c->command,s->pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),&shape);
    vkCmdDispatch(c->command,(uint32_t)a->heads,(uint32_t)a->batch,1);
    VkBufferMemoryBarrier barrier={0};barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.buffer=s->buffers[3].buffer;barrier.size=qbytes;
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&barrier,0,NULL);
    if (nya_vk_submit(c)) return -1;
    const float *output=s->buffers[3].mapped;
    for (size_t i=0;i<qbytes/sizeof(float);++i) if (!isfinite(output[i])) return -1;
    memcpy(a->output,output,qbytes);return 0;
}
