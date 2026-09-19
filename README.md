# ComfyUI Qwen V100 Mixed Precision

A ComfyUI runtime model patch for Qwen Image on NVIDIA Tesla V100 (`sm_70`).
It keeps large GEMMs on FP16 Tensor Core inputs while producing FP32 outputs
where Qwen Image otherwise overflows FP16. Residual, normalization, modulation,
and gating paths remain FP32.

The patch is reversible and does not modify ComfyUI core or the checkpoint.

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

Add **Qwen V100 Mixed Precision** and connect it after the final model patch or
LoRA node and before the sampler:

```text
Checkpoint / LoRA / optional Flash controller
                    ↓
       Qwen V100 Mixed Precision
                    ↓
                 KSampler
```

Do not also add a separate `ModelComputeDtype fp16` node. This node sets the
required compute policy itself.

Inputs:

- `model`: the Qwen Image `MODEL` connection.
- `enabled=true`: apply the mixed-precision patch.
- `enabled=false`: pass the original model through unchanged.
- `validate_numerics=false`: normal generation mode.
- `validate_numerics=true`: diagnostic mode; checks every transformer block,
  logs NaN/Inf/max-abs values, and stops on the first non-finite operation.

For normal use, keep `enabled=true` and `validate_numerics=false`.

## Disable or uninstall

- Disable without changing the workflow: set `enabled=false`.
- Remove from one workflow: bypass or delete the node and reconnect the model
  directly to the sampler.
- Uninstall completely: remove this repository from `custom_nodes` and restart
  ComfyUI.

## Validated configuration

Validated on a Tesla V100 32 GB with Qwen Image. In the tested five-step Qwen
Rapid workflow, the mixed path completed with zero NaN/Inf values and retained
the expected image structure without the black-pixel artifacts seen in pure
FP16 compute.

This project is specialized for V100. It is not a general mixed-precision patch
for other GPUs or model families.
