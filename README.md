# ComfyUI Qwen V100 Mixed Precision

A ComfyUI runtime model patch for **Qwen Image on NVIDIA Tesla V100 / Volta (`sm_70`)**.

It keeps large GEMMs on FP16 Tensor Core inputs while producing FP32 outputs where Qwen Image otherwise overflows FP16. Residual, normalization, modulation, and gating paths remain FP32.

On a tested five-step Qwen Rapid workflow, this reduced sampling time by about **30%** versus the FP32 baseline, with **1.43x sampling speedup**, zero NaN/Inf values, and no obvious visual artifacts.

The patch is reversible and does not modify ComfyUI core or the checkpoint.

## Benchmark

Test system: **NVIDIA Tesla V100 32 GB**, Qwen Rapid, 5 steps.

| Metric | FP32 baseline | V100 Mixed Precision |
| --- | ---: | ---: |
| Total time | 92.087 s | 64.658 s |
| Sampling time | 91.378 s | 63.923 s |
| Time per step | 18.276 s | 12.785 s |
| Peak VRAM | 29.13 GB | 28.63 GB |
| Peak system RAM | 12.02 GB | 12.93 GB |
| Average GPU utilization | 99.1% | 81.8% |
| Peak GPU utilization | 100% | 100% |
| NaN / Inf | — | 0 / 0 |
| Image valid | Yes | Yes |

**Result:** 1.42x faster total time, 1.43x faster sampling, and 29.8% lower total latency.

The first run was excluded as warmup. The custom node and workflow were unchanged between the measured runs.

These numbers are from this specific workflow and environment; they are not intended as a cross-GPU benchmark.

## Why this exists

Tesla V100 has strong FP16 Tensor Core performance, but modern Qwen Image workflows can hit numerical problems when forced into pure FP16. At the same time, running the relevant operations entirely in FP32 leaves substantial Tensor Core performance unused.

This node applies a V100-specific mixed-precision path: use FP16 where the hardware benefits, while retaining FP32 where Qwen Image needs the additional numerical range.

## Requirements

- NVIDIA Tesla V100 / compute capability 7.0
- ComfyUI with a Qwen Image model
- A PyTorch build where `torch.mm(fp16, fp16, out_dtype=torch.float32)` works

The node rejects non-`sm_70` GPUs and non-Qwen Image diffusion models.

## Installation

Clone this repository into `ComfyUI/custom_nodes`, then restart ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/lotus0000/ComfyUI-Qwen-V100-MixedPrecision.git
```

No additional Python package is required.

## Workflow usage

Add **Qwen V100 Mixed Precision** and connect it after the final model patch or LoRA node and before the sampler:

```text
Checkpoint / LoRA / optional Flash controller
                    ↓
       Qwen V100 Mixed Precision
                    ↓
                 KSampler
```

Do not also add a separate `ModelComputeDtype fp16` node. This node sets the required compute policy itself.

Inputs:

- `model`: the Qwen Image `MODEL` connection.
- `enabled=true`: apply the mixed-precision patch.
- `enabled=false`: pass the original model through unchanged.
- `validate_numerics=false`: normal generation mode.
- `validate_numerics=true`: diagnostic mode; checks every transformer block, logs NaN/Inf/max-abs values, and stops on the first non-finite operation.

For normal use, keep `enabled=true` and `validate_numerics=false`.

## Disable or uninstall

- Disable without changing the workflow: set `enabled=false`.
- Remove from one workflow: bypass or delete the node and reconnect the model directly to the sampler.
- Uninstall completely: remove this repository from `custom_nodes` and restart ComfyUI.

## Validation notes

The optimized path completed the tested workflow with zero NaN/Inf values and retained the expected image structure without the black-pixel artifacts seen in pure FP16 compute.

This project is intentionally specialized for **Tesla V100 / Volta**. It is not a general mixed-precision patch for other GPUs or model families.
