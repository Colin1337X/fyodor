# Learning-rate graph: Linux C-only validation

Fresh C17 and C23 Release builds pass all four TUI CTests: state/render,
snapshot, CLI and real POSIX PTY. Both source copies physically lack root
optcpp, backend/optcpp, CUDA/cutlass and CUDA/cute while NYA_ENABLE_OPTCPP=ON.
CUDA, Vulkan, ROCm, MLX and ONNX backends are disabled for these CPU checks.
Neither configuration discovers a C++ compiler, and ldd shows no C++ runtime.

The source/artifact manifests and complete logs are retained. verify.py checks
all recorded identities against the copied source and binaries. This validates
the latest learning-rate graph and existing TUI behavior; it is not a full-suite,
GPU, model performance or training performance result.
