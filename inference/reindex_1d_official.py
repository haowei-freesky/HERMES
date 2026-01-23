import torch
from typing import Tuple
from transformers import DynamicCache


def get_cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2]


def contiguous_kv(past_key_values):
    new_legacy = []
    for (k_layer, v_layer) in past_key_values:
        new_legacy.append((k_layer.contiguous(), v_layer.contiguous()))
    return new_legacy


def _get_rotary_module(llm) -> torch.nn.Module:
    if hasattr(llm, "rotary_emb"):
        return llm.rotary_emb
    if hasattr(llm, "model") and hasattr(llm.model, "rotary_emb"):
        return llm.model.rotary_emb

    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        if len(llm.model.layers) > 0 and hasattr(llm.model.layers[0], "self_attn"):
            if hasattr(llm.model.layers[0].self_attn, "rotary_emb"):
                return llm.model.layers[0].self_attn.rotary_emb
    raise AttributeError("Cannot find rotary_emb module on language_model")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_1d(x, cos, sin):
    if cos.dim() == 3 and x.dim() == 4:
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def rotary_delta(cos_old, sin_old, cos_new, sin_new):
    cos_delta = cos_new * cos_old + sin_new * sin_old
    sin_delta = sin_new * cos_old - cos_new * sin_old
    return cos_delta, sin_delta


@torch.inference_mode()
def compute_cos_sin_for_positions(
    llm, 
    seq_len: int, 
    position_ids_1d: torch.Tensor, 
    dtype: torch.dtype, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    rotary_emb = _get_rotary_module(llm)
    
    config = llm.config if hasattr(llm, "config") else llm.model.config
    hidden_size = getattr(config, "hidden_size", 4096)
    
    position_ids = position_ids_1d.view(1, -1).to(device)
    
    dummy_h = torch.zeros((1, seq_len, hidden_size), device=device, dtype=dtype)
    
    
    seq_len_rope = position_ids.max().item() + 1
    cos, sin = rotary_emb(dummy_h, seq_len=seq_len_rope)
    cos = cos[position_ids]
    sin = sin[position_ids]
    
    cos = cos.to(dtype)
    sin = sin.to(dtype)
    
    return cos, sin

@torch.inference_mode()
def compute_cos_sin_with_custom_base(
    llm,
    seq_len: int,
    position_ids_1d: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    rope_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    config = llm.config if hasattr(llm, "config") else llm.model.config
    hidden_size = getattr(config, "hidden_size", 4096)
    num_attention_heads = getattr(config, "num_attention_heads", 32)
    head_dim = hidden_size // num_attention_heads
    
    
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    
    freqs = torch.outer(position_ids_1d.float(), inv_freq)
    
    
    emb = torch.cat([freqs, freqs], dim=-1)
    
    cos = emb.cos().unsqueeze(0).to(dtype)  # [1, seq_len, head_dim]
    sin = emb.sin().unsqueeze(0).to(dtype)  # [1, seq_len, head_dim]
    
    return cos, sin


def get_layer_rope_base(
    layer_idx: int,
    num_layers: int,
    short_term_ratio: float = 0.3,
    long_term_ratio: float = 0.3,
    short_term_base: float = 10000.0,
    mid_term_base_range: Tuple[float, float] = (10000.0, 50000.0),
    long_term_base: float = 50000.0,
) -> float:
    
    short_term_threshold = int(num_layers * short_term_ratio)
    long_term_threshold = int(num_layers * (1 - long_term_ratio))
    
    if layer_idx < short_term_threshold:
        
        return short_term_base
    elif layer_idx >= long_term_threshold:
        
        return long_term_base
    else:
        
        mid_start = short_term_threshold
        mid_end = long_term_threshold
        progress = (layer_idx - mid_start) / max(mid_end - mid_start, 1)
        base_start, base_end = mid_term_base_range
        return base_start + (base_end - base_start) * progress


@torch.inference_mode()
def apply_rotary_delta_to_keys_only(
    key_states: torch.Tensor, 
    cos_delta: torch.Tensor, 
    sin_delta: torch.Tensor
) -> torch.Tensor:
    
    k_rot = apply_rotary_pos_emb_1d(key_states, cos_delta, sin_delta)
    return k_rot

