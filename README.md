# ComfyUI Qwen Image Mixed Precision for NVIDIA Tesla V100 (Volta / sm_70)

**Run Qwen Image faster on an NVIDIA Tesla V100 32GB in ComfyUI.** This custom node adds a V100-specific mixed-precision execution path for **Qwen Image / Qwen Rapid**, using FP16 Tensor Core inputs for large GEMMs while keeping FP32 output and numerically sensitive paths where needed.

On the tested **Tesla V100 32GB** Qwen Rapid workflow, sampling improved from **91.378s to 63.923s** over 5 steps: **1.43x faster sampling** and about **30% lower total latency**, with zero NaN/Inf values and no obvious visual artifacts.

> Designed specifically for NVIDIA Volta `sm_70`. It does not modify ComfyUI core or your checkpoint, and the patch is reversible.

## Benchmark: Tesla V100 32GB + Qwen Rapid

| Metric | FP32 baseline | V100 Mixed Precision |
| --- | ---: | ---: |
| Total time | 92.087 s | **64.658 s** |
| Sampling time | 91.378 s | **63.923 s** |
| Time per step | 18.276 s | **12.785 s** |
| Peak VRAM | 29.13 GB | **28.63 GB** |
| Peak system RAM | 12.02 GB | 12.93 GB |
| Average GPU utilization | 99.1% | 81.8% |
| Peak GPU utilization | 100% | 100% |
| NaN / Inf | — | **0 / 0** |
| Image valid | Yes | **Yes** |

**Result: 1.42x faster total time, 1.43x faster sampling, and 29.8% lower total latency.**

The first run was excluded as warmup. The custom node and workflow were unchanged between measured runs. These numbers are from this specific workflow and environment and are not intended as a cross-GPU benchmark.

## Who is this for?

This project is for people still using an **NVIDIA Tesla V100 / Volta GPU for ComfyUI** who want to run modern Qwen Image workflows without giving up all of the V100's FP16 Tensor Core performance.

It may be useful if:

- Qwen Image or Qwen Rapid works on your V100 but FP32 generation is slow.
- Forcing pure FP16 causes numerical instability, black pixels/images, NaN, or Inf values.
- You have a **V100 32GB** and want a practical mixed FP16/FP32 path instead of replacing the GPU.
- You are experimenting with modern generative AI workloads on older **CUDA compute capability 7.0 (sm_70)** hardware.

## Why V100 needs a special path

Tesla V100 has strong FP16 Tensor Core throughput, but it predates native BF16 and FP8 execution used by newer NVIDIA architectures.

Modern Qwen Image models can also hit numerical problems when relevant operations are forced entirely into FP16. Running everything in FP32 avoids some of those problems, but leaves substantial V100 Tensor Core performance unused.

This node therefore uses a targeted mixed-precision policy:

- Large Q/K/V, MLP, and output-projection GEMMs can use **FP16 Tensor Core inputs**.
- GEMM output is retained in **FP32** where supported.
- Residual, normalization, modulation, gating, and other numerically sensitive paths remain **FP32**.
- The original model/checkpoint is not rewritten.

The goal is not generic automatic mixed precision. It is a deliberately narrow optimization for **Qwen Image on NVIDIA Volta V100**.

## Requirements

- NVIDIA Tesla V100 / compute capability 7.0 (`sm_70`)
- Tested on Tesla V100 32GB
- ComfyUI with a Qwen Image model
- PyTorch build where `torch.mm(fp16, fp16, out_dtype=torch.float32)` works

The node rejects non-`sm_70` GPUs and non-Qwen Image diffusion models.

## Installation

Clone this repository into `ComfyUI/custom_nodes`, then restart ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/lotus0000/ComfyUI-Qwen-V100-MixedPrecision.git
```

No additional Python package is required.

## ComfyUI workflow usage

Add **Qwen V100 Mixed Precision** after the final model patch or LoRA node and before the sampler:

```text
Checkpoint / LoRA / optional Flash controller
                    ↓
       Qwen V100 Mixed Precision
                    ↓
                 KSampler
```

Do **not** also add a separate `ModelComputeDtype fp16` node. This node sets the required compute policy itself.

### Inputs

- `model`: Qwen Image `MODEL` connection.
- `enabled=true`: apply the V100 mixed-precision patch.
- `enabled=false`: pass the original model through unchanged.
- `validate_numerics=false`: normal generation mode.
- `validate_numerics=true`: diagnostic mode. Checks transformer blocks for NaN/Inf/max-abs values and stops on the first non-finite operation.

For normal use, keep `enabled=true` and `validate_numerics=false`.

## Troubleshooting

### Pure FP16 produces black images or unstable output

That is one of the reasons this project exists. Do not force the entire Qwen Image model into FP16. Remove/bypass separate compute-dtype patches and let this node apply its targeted FP16/FP32 policy.

### Node says the GPU is unsupported

This project intentionally targets **NVIDIA Tesla V100 / Volta compute capability 7.0 (sm_70)**. It is not intended as an optimization for Turing, Ampere, Ada, Hopper, Blackwell, or other GPU architectures.

### NaN or Inf appears

Enable `validate_numerics=true`. The node will log numerical information for transformer blocks and stop when it finds the first non-finite operation, which can help identify incompatible model/workflow changes.

### Performance differs from the benchmark

The reported result is from one tested Qwen Rapid five-step workflow. Resolution, model variant, LoRA stack, ComfyUI/PyTorch versions, CPU/RAM, and other workflow components can change performance.

## Disable or uninstall

- Disable without changing the workflow: set `enabled=false`.
- Remove from one workflow: bypass/delete the node and reconnect the model directly to the sampler.
- Uninstall completely: remove this repository from `custom_nodes` and restart ComfyUI.

## Validation notes

The optimized path completed the tested workflow with **zero NaN/Inf values** and retained the expected image structure without the black-pixel artifacts seen in pure FP16 compute.

This project is intentionally specialized for **NVIDIA Tesla V100, Volta, sm_70, ComfyUI, and Qwen Image/Qwen Rapid**. It is not a general mixed-precision patch for other GPUs or model families.

## Search keywords

NVIDIA Tesla V100, V100 32GB, Volta, sm_70, CUDA compute capability 7.0, ComfyUI V100, Qwen Image V100, Qwen Rapid V100, Qwen Image ComfyUI, mixed precision, FP16 Tensor Core, FP32, PyTorch, generative AI.
