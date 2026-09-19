import logging
import types

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention_masked


def _fp16_linear_fp32(linear, x):
    x_half = x.to(torch.float16)
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        linear,
        dtype=torch.float16,
        device=x.device,
        bias_dtype=torch.float32,
        offloadable=True,
    )
    try:
        shape = x.shape[:-1] + (weight.shape[0],)
        out = torch.mm(x_half.reshape(-1, x_half.shape[-1]), weight.t(), out_dtype=torch.float32).reshape(shape)
        if bias is not None:
            out.add_(bias)
        return out
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, offload_stream)


def _check(options, label, *tensors):
    checker = options.get("qwen_v100_checker")
    if checker is not None:
        checker(label, *tensors)


def _mixed_mlp(mlp, x, options, label):
    x = mlp.net[0].proj(x.to(torch.float16))
    _check(options, f"{label}.up_proj", x)
    x = F.gelu(x, approximate=mlp.net[0].approximate)
    _check(options, f"{label}.gelu", x)
    x = mlp.net[1](x)
    x = _fp16_linear_fp32(mlp.net[2], x)
    _check(options, f"{label}.down_proj", x)
    return x


def _mixed_attention(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                     attention_mask=None, image_rotary_emb=None, transformer_options={}):
    batch_size = hidden_states.shape[0]
    seq_img = hidden_states.shape[1]
    seq_txt = encoder_hidden_states.shape[1]
    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    img_input = hidden_states.to(torch.float16)
    txt_input = encoder_hidden_states.to(torch.float16)
    img_query = _fp16_linear_fp32(self.to_q, img_input).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
    img_key = _fp16_linear_fp32(self.to_k, img_input).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
    img_value = _fp16_linear_fp32(self.to_v, img_input).view(batch_size, seq_img, self.heads, -1).transpose(1, 2)
    txt_query = _fp16_linear_fp32(self.add_q_proj, txt_input).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
    txt_key = _fp16_linear_fp32(self.add_k_proj, txt_input).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
    txt_value = _fp16_linear_fp32(self.add_v_proj, txt_input).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)
    _check(transformer_options, "attention.img_q_proj", img_query)
    _check(transformer_options, "attention.img_k_proj", img_key)
    _check(transformer_options, "attention.img_v_proj", img_value)
    _check(transformer_options, "attention.txt_q_proj", txt_query)
    _check(transformer_options, "attention.txt_k_proj", txt_key)
    _check(transformer_options, "attention.txt_v_proj", txt_value)

    img_query = self.norm_q(img_query)
    img_key = self.norm_k(img_key)
    txt_query = self.norm_added_q(txt_query)
    txt_key = self.norm_added_k(txt_key)
    _check(transformer_options, "attention.qk_norm", img_query, img_key, txt_query, txt_key)
    joint_query = torch.cat([txt_query, img_query], dim=2)
    joint_key = torch.cat([txt_key, img_key], dim=2)
    joint_value = torch.cat([txt_value, img_value], dim=2)

    if encoder_hidden_states_mask is not None:
        attn_mask = torch.zeros((batch_size, 1, seq_txt + seq_img), dtype=joint_query.dtype, device=hidden_states.device)
        attn_mask[:, 0, :seq_txt] = encoder_hidden_states_mask.to(joint_query.dtype)
    else:
        attn_mask = None

    extra_options["img_slice"] = [txt_query.shape[2], joint_query.shape[2]]
    for patch in transformer_patches.get("attn1_patch", []):
        out = patch(joint_query, joint_key, joint_value, pe=image_rotary_emb,
                    attn_mask=encoder_hidden_states_mask, extra_options=extra_options)
        joint_query = out.get("q", joint_query)
        joint_key = out.get("k", joint_key)
        joint_value = out.get("v", joint_value)
        image_rotary_emb = out.get("pe", image_rotary_emb)
        encoder_hidden_states_mask = out.get("attn_mask", encoder_hidden_states_mask)

    joint_query = apply_rope1(joint_query, image_rotary_emb)
    joint_key = apply_rope1(joint_key, image_rotary_emb)
    joint_hidden_states = optimized_attention_masked(
        joint_query, joint_key, joint_value, self.heads, attn_mask,
        transformer_options=transformer_options, skip_reshape=True,
    )
    _check(transformer_options, "attention.softmax_value", joint_hidden_states)
    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
    img_attn_output = joint_hidden_states[:, seq_txt:, :]
    img_attn_output = self.to_out[1](_fp16_linear_fp32(self.to_out[0], img_attn_output))
    txt_attn_output = _fp16_linear_fp32(self.to_add_out, txt_attn_output)
    _check(transformer_options, "attention.output_proj", img_attn_output, txt_attn_output)
    return img_attn_output, txt_attn_output


def _mixed_block(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb,
                 image_rotary_emb=None, timestep_zero_index=None, transformer_options={}):
    begin = transformer_options.get("qwen_v100_begin")
    if begin is not None:
        begin(self)
    hidden_states = hidden_states.float()
    encoder_hidden_states = encoder_hidden_states.float()
    temb = temb.float()
    img_mod_params = self.img_mod(temb)
    if timestep_zero_index is not None:
        temb = temb.chunk(2, dim=0)[0]
    txt_mod_params = self.txt_mod(temb)
    _check(transformer_options, "modulation.params", img_mod_params, txt_mod_params)
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

    img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1, timestep_zero_index)
    txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
    _check(transformer_options, "norm_modulation.1", img_modulated, img_gate1, txt_modulated, txt_gate1)
    img_attn_output, txt_attn_output = self.attn(
        hidden_states=img_modulated,
        encoder_hidden_states=txt_modulated,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        transformer_options=transformer_options,
    )
    hidden_states = self._apply_gate(img_attn_output, hidden_states, img_gate1, timestep_zero_index)
    encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate1, txt_attn_output)
    _check(transformer_options, "attention.residual", hidden_states, encoder_hidden_states)

    img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, timestep_zero_index)
    _check(transformer_options, "norm_modulation.2.image", img_modulated2, img_gate2)
    hidden_states = self._apply_gate(_mixed_mlp(self.img_mlp, img_modulated2, transformer_options, "image_mlp"), hidden_states, img_gate2, timestep_zero_index)
    _check(transformer_options, "image_mlp.residual", hidden_states)
    txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
    _check(transformer_options, "norm_modulation.2.text", txt_modulated2, txt_gate2)
    encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, _mixed_mlp(self.txt_mlp, txt_modulated2, transformer_options, "text_mlp"))
    _check(transformer_options, "text_mlp.residual", encoder_hidden_states)

    validator = transformer_options.get("qwen_v100_validator")
    if validator is not None:
        validator(self, encoder_hidden_states, hidden_states)
    return encoder_hidden_states, hidden_states


class QwenV100MixedPrecision:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "validate_numerics": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"

    def patch(self, model, enabled=True, validate_numerics=False):
        if not enabled:
            return (model,)
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
            raise RuntimeError("Qwen V100 Mixed Precision requires an NVIDIA sm_70 GPU")

        patched = model.clone()
        diffusion_model = patched.model.diffusion_model
        if diffusion_model.__class__.__name__ != "QwenImageTransformer2DModel":
            raise RuntimeError("Qwen V100 Mixed Precision only supports Qwen Image models")

        patched.set_model_compute_dtype(torch.float16)
        for index, block in enumerate(diffusion_model.transformer_blocks):
            patched.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}.forward",
                types.MethodType(_mixed_block, block),
            )
            patched.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}.attn.forward",
                types.MethodType(_mixed_attention, block.attn),
            )

        if validate_numerics:
            block_ids = {id(block): index for index, block in enumerate(diffusion_model.transformer_blocks)}
            current_block = {"index": -1}
            def begin(block):
                current_block["index"] = block_ids[id(block)]
            patched.model_options.setdefault("transformer_options", {})["qwen_v100_begin"] = begin
            def checker(label, *tensors):
                flat = []
                for tensor in tensors:
                    flat.extend(tensor if isinstance(tensor, tuple) else (tensor,))
                nan_count = sum(int(torch.isnan(t).sum().item()) for t in flat)
                inf_count = sum(int(torch.isinf(t).sum().item()) for t in flat)
                if nan_count or inf_count:
                    finite = [t[torch.isfinite(t)] for t in flat]
                    max_abs = max((float(t.abs().max().item()) for t in finite if t.numel()), default=float("nan"))
                    raise FloatingPointError(
                        f"Qwen V100 mixed precision first non-finite at block {current_block['index']} {label}: "
                        f"nan={nan_count}, inf={inf_count}, finite_max_abs={max_abs:.6g}"
                    )
            patched.model_options.setdefault("transformer_options", {})["qwen_v100_checker"] = checker
            def validator(block, text, image):
                index = block_ids[id(block)]
                nan_count = int(torch.isnan(text).sum().item() + torch.isnan(image).sum().item())
                inf_count = int(torch.isinf(text).sum().item() + torch.isinf(image).sum().item())
                max_abs = max(float(text.abs().max().item()), float(image.abs().max().item()))
                logging.info("[QwenV100Mixed] block=%d nan=%d inf=%d max_abs=%.6g", index, nan_count, inf_count, max_abs)
                if nan_count or inf_count:
                    raise FloatingPointError(f"Qwen V100 mixed precision produced non-finite values at block {index}: nan={nan_count}, inf={inf_count}")
            patched.model_options.setdefault("transformer_options", {})["qwen_v100_validator"] = validator

        logging.info("[QwenV100Mixed] enabled: FP16 GEMM inputs with FP32 output/residual on sm_70")
        return (patched,)


NODE_CLASS_MAPPINGS = {"QwenV100MixedPrecision": QwenV100MixedPrecision}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenV100MixedPrecision": "Qwen V100 Mixed Precision"}
