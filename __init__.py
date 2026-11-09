import logging
import types

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.qwen_image21.model import _gated_residual, _modulated_norm
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


def _mixed_swiglu(mlp, x, options):
    if mlp.fused:
        gate, up = _fp16_linear_fp32(mlp.gate_up, x).chunk(2, dim=-1)
    else:
        gate, up = _fp16_linear_fp32(mlp.gate_layer, x), _fp16_linear_fp32(mlp.proj, x)
    _check(options, "image_mlp.up_proj", gate, up)
    x = F.silu(gate).mul_(up)
    _check(options, "image_mlp.swiglu", x)
    x = _fp16_linear_fp32(mlp.out, x)
    _check(options, "image_mlp.down_proj", x)
    return x


def _mixed_attention21(self, x, pe, attn_fn, prefix_len, transformer_options={}):
    B, N, _ = x.shape
    x = x.to(torch.float16)
    q = _fp16_linear_fp32(self.to_q, x).view(B, N, self.heads, -1)
    k = _fp16_linear_fp32(self.to_k, x).view(B, N, self.heads, -1)
    v = _fp16_linear_fp32(self.to_v, x).view(B, N, self.heads, -1)
    _check(transformer_options, "attention.qkv_proj", q, k, v)
    patches = transformer_options.get("patches", {}).get("attn1_patch", [])
    if patches:
        q, k = self.norm_q(q), self.norm_k(k)
        q, k, v, pe = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), pe.transpose(1, 2)
        extra_options = {**transformer_options, "img_slice": [prefix_len, N]}
        for p in patches:
            out = p(q, k, v, pe=pe, attn_mask=None, extra_options=extra_options)
            q, k, v, pe = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe)
        q, k, v, pe = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), pe.transpose(1, 2).contiguous()
        q = apply_rope1(q, pe)
        k = apply_rope1(k, pe)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(self.norm_q, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
        comfy.ops.uncast_bias_weight(self.norm_q, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(self.norm_k, k_scale, None, k_stream)
    _check(transformer_options, "attention.qk_norm_rope", q, k)
    out = attn_fn(q, k, v, self.heads, preferred_attention=self.comfy_attention)
    _check(transformer_options, "attention.softmax_value", out)
    out = _fp16_linear_fp32(self.to_out[0], out)
    _check(transformer_options, "attention.output_proj", out)
    return out


def _mixed_block21(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
    begin = transformer_options.get("qwen_v100_begin")
    if begin is not None:
        begin(self)
    scale1, gate1, scale2, gate2, zero = mod
    x = _gated_residual(x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len, zero), pe, attn_fn, prefix_len, transformer_options), gate1, prefix_len)
    _check(transformer_options, "attention.residual", x)
    x = _gated_residual(x, _mixed_swiglu(self.img_mlp, _modulated_norm(self.img_norm2, x, scale2, prefix_len, zero), transformer_options), gate2, prefix_len)
    _check(transformer_options, "image_mlp.residual", x)

    validator = transformer_options.get("qwen_v100_validator")
    if validator is not None:
        validator(self, x)
    return x


# 2.1 keeps an fp32 compute dtype so the residual and modulation are fp32 before the block loop
# and blocks only update the residual in place, matching the per-block malloc graph
MIXED_FORWARDS = {
    "QwenImageTransformer2DModel": (_mixed_block, _mixed_attention, torch.float16),
    "QwenImage21Transformer2DModel": (_mixed_block21, _mixed_attention21, torch.float32),
}


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
        if diffusion_model.__class__.__name__ not in MIXED_FORWARDS:
            raise RuntimeError("Qwen V100 Mixed Precision only supports Qwen Image and Qwen Image 2.1 models")
        block_forward, attention_forward, compute_dtype = MIXED_FORWARDS[diffusion_model.__class__.__name__]

        patched.set_model_compute_dtype(compute_dtype)
        for index, block in enumerate(diffusion_model.transformer_blocks):
            patched.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}.forward",
                types.MethodType(block_forward, block),
            )
            patched.add_object_patch(
                f"diffusion_model.transformer_blocks.{index}.attn.forward",
                types.MethodType(attention_forward, block.attn),
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
            def validator(block, *tensors):
                index = block_ids[id(block)]
                nan_count = sum(int(torch.isnan(t).sum().item()) for t in tensors)
                inf_count = sum(int(torch.isinf(t).sum().item()) for t in tensors)
                max_abs = max(float(t.abs().max().item()) for t in tensors)
                logging.info("[QwenV100Mixed] block=%d nan=%d inf=%d max_abs=%.6g", index, nan_count, inf_count, max_abs)
                if nan_count or inf_count:
                    raise FloatingPointError(f"Qwen V100 mixed precision produced non-finite values at block {index}: nan={nan_count}, inf={inf_count}")
            patched.model_options.setdefault("transformer_options", {})["qwen_v100_validator"] = validator

        logging.info("[QwenV100Mixed] enabled: FP16 GEMM inputs with FP32 output/residual on sm_70")
        return (patched,)


NODE_CLASS_MAPPINGS = {"QwenV100MixedPrecision": QwenV100MixedPrecision}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenV100MixedPrecision": "Qwen V100 Mixed Precision"}
