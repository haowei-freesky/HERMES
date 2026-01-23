import re
import time
import torch
import torch.nn.functional as F
import types
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoProcessor
from transformers import DynamicCache
from logzero import logger
from abstract_Hermes import Abstract_Hermes

from reindex_1d_official import (
    _get_rotary_module,
    get_cache_seq_len,
    contiguous_kv,
    compute_cos_sin_for_positions,
    rotary_delta,
    apply_rotary_delta_to_keys_only,
)


class LlavaOneVision_Hermes(Abstract_Hermes):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size, kv_size):
        super().__init__(config)
        
        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 32
        self.num_layers = num_layers
        
        # Memory hierarchy
        self.memory_adaptive_mode = 'memory_hierarchy'
        self.memory_mode = 'hierarchy'
        self.short_term_ratio = 0.3
        self.long_term_ratio = 0.3
        self.short_term_threshold = int(self.num_layers * self.short_term_ratio)
        self.long_term_threshold = int(self.num_layers * (1 - self.long_term_ratio))
        
        self._position_ids_cache = [None for _ in range(num_layers)]
        self.token_activity_cache = [None for _ in range(num_layers)]
        
        self._layer_position_ids = {}
        self._hook_handles = []
        self._register_forward_hooks()
        
        self.total_processed_frames = 0
        
        self._patch_rotary_embeddings()

    def _patch_rotary_embeddings(self):
        """Patch rotary embedding to support dynamic resizing based on logical position IDs"""
        def patched_forward(rotary_self, x, seq_len=None):
            required_len = getattr(rotary_self, '_required_seq_len', 0)
            if seq_len is not None and required_len > seq_len:
                seq_len = required_len
            return rotary_self._original_forward(x, seq_len=seq_len)

        if hasattr(self.language_model.model, 'layers'):
            for layer in self.language_model.model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    rotary_emb = layer.self_attn.rotary_emb
                    if not hasattr(rotary_emb, '_original_forward'):
                        rotary_emb._original_forward = rotary_emb.forward
                        rotary_emb.forward = types.MethodType(patched_forward, rotary_emb)
                        rotary_emb._required_seq_len = 0

    def _set_rotary_required_len(self, required_len):
        """Set required sequence length for rotary embeddings"""
        if hasattr(self.language_model.model, 'layers'):
            for layer in self.language_model.model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb._required_seq_len = required_len

    def _register_forward_hooks(self):
        if not hasattr(self.language_model, 'model') or not hasattr(self.language_model.model, 'layers'):
            return
        
        def create_hook(layer_idx):
            def hook(module, args, kwargs):
                if layer_idx in self._layer_position_ids:
                    kwargs['position_ids'] = self._layer_position_ids[layer_idx]
                return args, kwargs
            return hook
        
        for layer_idx, layer in enumerate(self.language_model.model.layers):
            handle = layer.register_forward_pre_hook(create_hook(layer_idx), with_kwargs=True)
            self._hook_handles.append(handle)
    
    def _clear_forward_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _prune_activity_cache(self, keep_indices_per_layer):
        if all(c is None for c in self.token_activity_cache):
            return

        for layer_idx in range(self.num_layers):
            if self.token_activity_cache[layer_idx] is not None:
                keep_indices = keep_indices_per_layer[layer_idx]
                if not isinstance(keep_indices, torch.Tensor):
                    keep_indices = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
                
                curr_len = self.token_activity_cache[layer_idx].shape[0]
                keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < curr_len)]
                
                if keep_indices.numel() > 0:
                    self.token_activity_cache[layer_idx] = self.token_activity_cache[layer_idx][keep_indices]
                else:
                    self.token_activity_cache[layer_idx] = torch.zeros(0, device=self.device)

    def _append_position_ids_layer(self, layer_idx: int, start: int, length: int):
        device = self.device
        new_pos = torch.arange(start, start + length, device=device, dtype=torch.long)
        if self._position_ids_cache[layer_idx] is None:
            self._position_ids_cache[layer_idx] = new_pos
        else:
            self._position_ids_cache[layer_idx] = torch.cat(
                [self._position_ids_cache[layer_idx], new_pos], dim=0
            )
    
    def _append_position_ids(self, start_per_layer, length: int):
        if isinstance(start_per_layer, int):
            for layer_idx in range(self.num_layers):
                self._append_position_ids_layer(layer_idx, start_per_layer, length)
        else:
            for layer_idx in range(self.num_layers):
                self._append_position_ids_layer(layer_idx, start_per_layer[layer_idx], length)

    def _get_cache_seq_len_per_layer(self) -> list:
        if self.kv_cache is None:
            return [0] * self.num_layers
        
        lengths = []
        for layer_idx in range(len(self.kv_cache)):
            k_layer, v_layer = self.kv_cache[layer_idx]
            lengths.append(k_layer.shape[2])
        return lengths

    def _get_next_start_pos_per_layer(self) -> list:
        next_pos = []
        for layer_idx in range(self.num_layers):
            if (layer_idx < len(self._position_ids_cache) and 
                self._position_ids_cache[layer_idx] is not None and 
                self._position_ids_cache[layer_idx].numel() > 0):
                next_pos.append(self._position_ids_cache[layer_idx][-1].item() + 1)
            else:
                next_pos.append(0)
        return next_pos
    
    def _sanitize_keep_indices(self, keep_indices_1d: torch.Tensor, seq_len: int) -> torch.Tensor:
        keep_indices_1d = keep_indices_1d.to(self.device, dtype=torch.long)
        keep_indices_1d = keep_indices_1d[(keep_indices_1d >= 0) & (keep_indices_1d < seq_len)]
        if keep_indices_1d.numel() == 0:
            return torch.tensor([0], device=self.device, dtype=torch.long)
        keep_indices_1d = torch.unique(keep_indices_1d, sorted=True)
        return keep_indices_1d

    def _build_position_ids(self, past_len: int, q_len: int, batch: int) -> torch.Tensor:
        pos_1d = torch.arange(past_len, past_len + q_len, device=self.device, dtype=torch.long)
        position_ids = pos_1d.view(1, -1).expand(batch, -1)
        return position_ids

    @torch.inference_mode()
    def _shrink_positions_and_rerotate_keys(self, keep_indices_per_layer):
        device = self.device
        
        curr_lens = self._get_cache_seq_len_per_layer()
        
        for layer_idx in range(self.num_layers):
            if (self._position_ids_cache[layer_idx] is None or 
                self._position_ids_cache[layer_idx].numel() != curr_lens[layer_idx]):
                self._position_ids_cache[layer_idx] = torch.arange(
                    curr_lens[layer_idx], device=device, dtype=torch.long
                )
        max_pos_limit = getattr(self.language_model.config, "max_position_embeddings", 32768)
        compact_threshold = max_pos_limit - 1024
        
        current_max_pos = 0
        for cache in self._position_ids_cache:
             if cache is not None and cache.numel() > 0:
                 current_max_pos = max(current_max_pos, cache[-1].item())
        
        should_compact = current_max_pos > compact_threshold
        
        if should_compact:
            logger.info(f"[Shrink] Max position {current_max_pos} > {compact_threshold}. Compacting position IDs (Shift Left).")
        
        old_position_ids_cache = [cache.clone() for cache in self._position_ids_cache]

        sample_k = self.kv_cache[0][0]
        dtype = sample_k.dtype
        
        new_kv_cache = []
        
        for layer_idx, (k_layer, v_layer) in enumerate(self.kv_cache):
            
            keep_indices_layer = keep_indices_per_layer[layer_idx]
            if not isinstance(keep_indices_layer, torch.Tensor):
                keep_indices_layer = torch.as_tensor(keep_indices_layer, dtype=torch.long, device=device)
            
            seq_len_layer = k_layer.shape[2]
            safe_idx = self._sanitize_keep_indices(keep_indices_layer, seq_len_layer)
            
            if safe_idx.numel() == 0:
                logger.warning(f"Layer {layer_idx}: After sanitization, keep_indices is empty; keeping first token")
                safe_idx = torch.tensor([0], device=device, dtype=torch.long)

            is_long_term = (layer_idx >= self.long_term_threshold)
            
            # === Extract Kept Tokens ===
            k_kept = torch.index_select(k_layer, dim=2, index=safe_idx)
            v_kept = torch.index_select(v_layer, dim=2, index=safe_idx)
            old_pos_kept = old_position_ids_cache[layer_idx].index_select(0, safe_idx)
            
            # === Determine New Positions for Kept Tokens ===
            if should_compact:
                new_pos_kept = torch.arange(len(safe_idx), device=device, dtype=torch.long)
            else:
                new_pos_kept = old_pos_kept

            # === Handle RoPE for Kept Tokens ===
            if should_compact:
                cos_old, sin_old = compute_cos_sin_for_positions(
                    self.language_model, len(old_pos_kept), old_pos_kept, dtype, device
                )
                cos_new, sin_new = compute_cos_sin_for_positions(
                    self.language_model, len(new_pos_kept), new_pos_kept, dtype, device
                )
                cos_delta, sin_delta = rotary_delta(
                    cos_old, sin_old, cos_new, sin_new
                )
                k_kept_final = apply_rotary_delta_to_keys_only(k_kept, cos_delta, sin_delta)
            else:
                k_kept_final = k_kept
            
            # === Handle Folding (Summary Token) for Long-term Layers ===
            if is_long_term:
                mask = torch.ones(seq_len_layer, dtype=torch.bool, device=device)
                mask[safe_idx] = False
                prune_indices = torch.nonzero(mask).squeeze(1)
                
                if prune_indices.numel() > 0:
                    k_pruned = torch.index_select(k_layer, dim=2, index=prune_indices)
                    v_pruned = torch.index_select(v_layer, dim=2, index=prune_indices)
                    
                    v_summary = v_pruned.mean(dim=2, keepdim=True)
                    
                    old_pos_pruned = old_position_ids_cache[layer_idx].index_select(0, prune_indices)
                    
                    if should_compact:
                        summary_pos_id = len(safe_idx)
                    else:
                        summary_pos_id = new_pos_kept[-1].item() + 1
                    
                    summary_pos_tensor = torch.tensor([summary_pos_id], device=device, dtype=torch.long)
                    target_pos_pruned = summary_pos_tensor.repeat(old_pos_pruned.shape[0])
                    
                    cos_old, sin_old = compute_cos_sin_for_positions(
                        self.language_model, len(old_pos_pruned), old_pos_pruned, dtype, device
                    )
                    cos_new, sin_new = compute_cos_sin_for_positions(
                        self.language_model, len(target_pos_pruned), target_pos_pruned, dtype, device
                    )
                    
                    cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)
                    
                    k_pruned_aligned = apply_rotary_delta_to_keys_only(k_pruned, cos_delta, sin_delta)
                    
                    k_summary_final = k_pruned_aligned.mean(dim=2, keepdim=True)
                    
                    k_final = torch.cat([k_kept_final, k_summary_final], dim=2)
                    v_final = torch.cat([v_kept, v_summary], dim=2)
                    
                    new_pos_layer = torch.cat([new_pos_kept, summary_pos_tensor])
                    
                else:
                    k_final = k_kept_final
                    v_final = v_kept
                    new_pos_layer = new_pos_kept
            else:
                k_final = k_kept_final
                v_final = v_kept
                new_pos_layer = new_pos_kept

            new_kv_cache.append((k_final.contiguous(), v_final.contiguous()))
            
            self._position_ids_cache[layer_idx] = new_pos_layer.clone()
        
        self.kv_cache = new_kv_cache
        contiguous_kv(self.kv_cache)
        
        new_lens = self._get_cache_seq_len_per_layer()
        logger.info(f"Layer-wise shrink completed. Lengths: min={min(new_lens)}, max={max(new_lens)}, "
                   f"first={new_lens[0]}, last={new_lens[-1]}")
    
    @torch.inference_mode()
    def predict_next_question(self, max_new_tokens=100, temperature=0.7):
        """
        Generates queries to drive attention calculation.
        Includes Semantic Anchors for better coverage.
        """
        if hasattr(self, 'conv_history') and len(self.conv_history) > 0:
            last_q, last_a, last_options = self.conv_history[-1]
            
            option_match = re.match(r'^\s*(?:\()?([A-Z])(?:\))?\.?\s*$', last_a)
            if option_match and last_options is not None:
                option_char = option_match.group(1)
                option_idx = ord(option_char) - ord('A')
                if 0 <= option_idx < len(last_options):
                    last_a = last_options[option_idx]

            last_a = re.sub(r'[A-Z]\) ', '', last_a)

            last_round_history = f"Question: {last_q} Answer: {last_a}"
            
            global_query = f"Context summary: {last_round_history}. Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme."
            local_query = f"Find recent details related to: {last_round_history}. Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships."
            
        else:
            global_query = (
                "Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme."
            )
            local_query = (
                "Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships."
            )
            
        print(f"Local question: {local_query}")
        print(f"Global question: {global_query}")
        return local_query, global_query
    
    @torch.inference_mode()
    def encode_init_prompt(self):
        super().encode_init_prompt()
        self.total_processed_frames = 0
        curr_lens = self._get_cache_seq_len_per_layer()
        for layer_idx in range(self.num_layers):
            self._position_ids_cache[layer_idx] = torch.arange(
                curr_lens[layer_idx], device=self.device, dtype=torch.long
            )

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk):
        if video_chunk is None or (hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0):
            return

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        pixel_values_videos = self.processor.video_processor(
            video_chunk, return_tensors="pt"
        ).pixel_values_videos.to(self.device, self.dtype)
        video_features = self._get_video_features(pixel_values_videos)

        start_pos_per_layer = self._get_next_start_pos_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]
        
        max_start_pos = max(start_pos_per_layer) if start_pos_per_layer else 0
        self._set_rotary_required_len(max_start_pos + q_len)
        
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids = self._build_position_ids(start_pos_per_layer[layer_idx], q_len, batch)
            self._layer_position_ids[layer_idx] = position_ids
        
        default_position_ids = self._build_position_ids(start_pos_per_layer[0], q_len, batch)
        
        out = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=default_position_ids,
            cache_position=None,
            attention_mask=None,
        )
        self.kv_cache = out.past_key_values
        contiguous_kv(self.kv_cache)

        self._append_position_ids(start_pos_per_layer, q_len)
        self.last_encoded_frames = video_chunk.shape[0]
        self.total_processed_frames += video_chunk.shape[0]
        
        self._layer_position_ids.clear()

    @torch.inference_mode()
    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers):
        if self.kv_cache is None:
            logger.warning("No KV-Cache to prune")
            return
        if not keep_indices_all_layers or len(keep_indices_all_layers[0]) == 0:
            logger.warning("Empty keep_indices; skip pruning")
            return

        self._shrink_positions_and_rerotate_keys(keep_indices_all_layers)
        self._prune_activity_cache(keep_indices_all_layers)
        
        logger.info(f"Strict-shrunk KV cache. New length: {get_cache_seq_len(self.kv_cache)}")

    def allocate_budget_by_depth(self, total_budget, num_layers):
        """Allocate budget uniformly across layers."""
        budget_per_layer = [total_budget // num_layers] * num_layers
        diff = total_budget - sum(budget_per_layer)
        budget_per_layer[-1] += diff
        return budget_per_layer

    def repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _compute_attention_scores_manually(self, input_ids, past_key_values):
        device = self.device
        start_pos_per_layer = self._get_next_start_pos_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]
        
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        config = self.language_model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        
        attention_weights_list = []
        
        hidden_states = inputs_embeds
        
        for layer_idx in range(num_layers):
            layer = self.language_model.model.layers[layer_idx]
            
            past_k, past_v = past_key_values[layer_idx]
            
            position_ids = self._build_position_ids(start_pos_per_layer[layer_idx], q_len, batch)
            
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm(hidden_states)
            
            attn = layer.self_attn
            
            query_states = attn.q_proj(hidden_states_norm)
            query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)
            
            key_states = attn.k_proj(hidden_states_norm)
            key_states = key_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            
            value_states = attn.v_proj(hidden_states_norm)
            value_states = value_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            
            kv_seq_len = position_ids.max().item() + 1
            rotary_emb = _get_rotary_module(self.language_model)
            cos, sin = rotary_emb(value_states, kv_seq_len)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)
            
            # GQA
            if num_key_value_heads != num_heads:
                key_states = self.repeat_kv(key_states, num_heads // num_key_value_heads)
                value_states = self.repeat_kv(value_states, num_heads // num_key_value_heads)
            
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_weights_list.append(attn_weights)
        
        return attention_weights_list

    def prune_kv_cache_by_attention(self, attn_weights_local, attn_weights_global, attn_weights_mixed, num_keep=3000):
        device = self.device
        visual_start_idx = self.visual_start_idx
        num_layers = len(attn_weights_local)
        
        curr_cache_lens = self._get_cache_seq_len_per_layer()
        
        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        total_budget = num_keep * num_layers
        budget_per_layer = self.allocate_budget_by_depth(total_budget, num_layers)
        
        keep_indices_all_layers = []
        all_layers_visual_attn = []
        all_layers_stats = []

        layer_raw_scores = []
        layer_configs = []

        for layer_idx in range(len(attn_weights_local)):
            curr_len = curr_cache_lens[layer_idx]
            if self.token_activity_cache[layer_idx] is None:
                self.token_activity_cache[layer_idx] = torch.zeros(curr_len, device=device, dtype=torch.float32)
            elif self.token_activity_cache[layer_idx].shape[0] < curr_len:
                diff = curr_len - self.token_activity_cache[layer_idx].shape[0]
                zeros = torch.zeros(diff, device=device, dtype=torch.float32)
                self.token_activity_cache[layer_idx] = torch.cat([self.token_activity_cache[layer_idx], zeros])
            
            if self.token_activity_cache[layer_idx].shape[0] > curr_len:
                 self.token_activity_cache[layer_idx] = self.token_activity_cache[layer_idx][:curr_len]

            # Memory hierarchy mode
            if layer_idx < self.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                attn_source = "local"
                layer_recency_alpha = 1
                k = 20
                
            elif layer_idx >= self.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                attn_source = "global"
                layer_recency_alpha = 0
                k = 0.0
                
            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                attn_source = "mixed"
                progress = (layer_idx - self.short_term_threshold) / (self.long_term_threshold - self.short_term_threshold)
                layer_recency_alpha = 1.0 - 0.9 * progress
                k = 20 - 19.5 * progress
            
            visual_attn_weights = layer_attn_weights[0].mean(dim=0)[:,visual_start_idx:-1*question_len].mean(dim=0)
            all_layers_visual_attn.append(visual_attn_weights)
            
            num_visual_tokens = visual_attn_weights.shape[0]
            
            end_idx = visual_start_idx + num_visual_tokens
            if end_idx <= self.token_activity_cache[layer_idx].shape[0]:
                self.token_activity_cache[layer_idx][visual_start_idx:end_idx] += visual_attn_weights
            
            layer_budget = budget_per_layer[layer_idx]
            
            if layer_type == 'long-term':
                layer_budget = max(0, layer_budget - 1)

            all_layers_stats.append({
                'layer': layer_idx, 'layer_type': layer_type,
                'recency_alpha': layer_recency_alpha, 'k': k, 'attn_source': attn_source
            })

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            
            # Exponential decay
            recency_weights = torch.exp(-k * time_distances)
            
            attn_norm = (visual_attn_weights - visual_attn_weights.min()) / \
                        (visual_attn_weights.max() - visual_attn_weights.min() + 1e-6)
            recency_norm = (recency_weights - recency_weights.min()) / \
                        (recency_weights.max() - recency_weights.min() + 1e-6)
            
            raw_score = attn_norm * (1 - layer_recency_alpha) + recency_norm * layer_recency_alpha
            
            layer_raw_scores.append(raw_score)
            layer_configs.append({
                'budget': min(layer_budget, num_visual_tokens),
                'layer_type': layer_type,
                'visual_start_idx': visual_start_idx
            })
        
        refined_scores = [s.clone() for s in layer_raw_scores]
        
        for i in range(len(refined_scores) - 2, -1, -1):
            current_type = layer_configs[i]['layer_type']
            
            if current_type == 'long-term':
                gamma = 0.4  
            elif current_type == 'mid-term':
                gamma = 0.3  
            else:
                gamma = 0.1
            
            score_current = refined_scores[i]
            score_next = refined_scores[i+1]
            
            if score_current.shape[0] != score_next.shape[0]:
                score_next_reshaped = score_next.view(1, 1, -1)
                score_next_interp = F.interpolate(
                    score_next_reshaped, 
                    size=score_current.shape[0], 
                    mode='linear', 
                    align_corners=False
                ).view(-1)
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next_interp
            else:
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next

        for layer_idx, score in enumerate(refined_scores):
            config = layer_configs[layer_idx]
            actual_num_keep = config['budget']
            start_idx = config['visual_start_idx']
            
            topk_indices_relative = torch.topk(score, actual_num_keep, sorted=False)[1]
            topk_indices_absolute = topk_indices_relative + start_idx
            topk_indices_absolute_sorted = torch.sort(topk_indices_absolute)[0]
            
            keep_indices = torch.cat([
                torch.arange(start_idx, device=device), 
                topk_indices_absolute_sorted
            ]).tolist()
            
            keep_indices_all_layers.append(keep_indices)
        
        if all_layers_stats:
            print(f"\n{'='*80}")
            print(f"[Memory Hierarchy KV Cache Pruning]")
            print(f"Memory Mode: {self.memory_mode} | Inter-layer Consistency Enabled")
            print(f"{'='*80}")
            
            short_term_stats = [s for s in all_layers_stats if s.get('layer_type') == 'short-term']
            mid_term_stats = [s for s in all_layers_stats if s.get('layer_type') == 'mid-term']
            long_term_stats = [s for s in all_layers_stats if s.get('layer_type') == 'long-term']
            
            total_kept = sum(len(keep_indices_all_layers[s['layer']]) - visual_start_idx for s in all_layers_stats)
            avg_recency_alpha = sum(s['recency_alpha'] for s in all_layers_stats) / len(all_layers_stats)
            print(f"Overall: total_kept={total_kept}, avg_recency_α={avg_recency_alpha:.3f}")
            print(f"  Short-term: {len(short_term_stats)} layers")
            print(f"  Mid-term:   {len(mid_term_stats)} layers")
            print(f"  Long-term:  {len(long_term_stats)} layers")
            print(f"{'='*80}\n")
            
        return keep_indices_all_layers, all_layers_visual_attn
    
    @torch.inference_mode()
    def pseudo_forward(self, local_question=None, global_question=None, input_text=None):
        device = self.device
        
        if local_question is None:
            local_question = "What is happening in the video?"
        if global_question is None:
            global_question = "What is the main topic of the video?"
        
        local_input_ids = self.processor.tokenizer(local_question).input_ids
        local_input_ids = torch.as_tensor([local_input_ids], device=device, dtype=torch.int)

        start_pos_per_layer = self._get_next_start_pos_per_layer()
        q_len_local = local_input_ids.shape[1]
        batch = local_input_ids.shape[0]
        
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids = self._build_position_ids(start_pos_per_layer[layer_idx], q_len_local, batch)
            self._layer_position_ids[layer_idx] = position_ids
        
        position_ids_local = self._build_position_ids(start_pos_per_layer[0], q_len_local, batch)

        max_start = max(start_pos_per_layer) if start_pos_per_layer else 0
        self._set_rotary_required_len(max_start + q_len_local)

        use_flash_attn = (hasattr(self.language_model.config, '_attn_implementation') and 
                        self.language_model.config._attn_implementation in ["flash_attention_2", "sdpa"])

        if use_flash_attn:
            attn_weights_local = self._compute_attention_scores_manually(local_input_ids, self.kv_cache)
        else:
            out_local = self.language_model(
                input_ids=local_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_local,
                cache_position=None,
                attention_mask=None,
            )
            attn_weights_local = out_local.attentions
        
        global_input_ids = self.processor.tokenizer(global_question).input_ids
        global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.int)
        
        q_len_global = global_input_ids.shape[1]
        
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids = self._build_position_ids(start_pos_per_layer[layer_idx], q_len_global, batch)
            self._layer_position_ids[layer_idx] = position_ids
        
        position_ids_global = self._build_position_ids(start_pos_per_layer[0], q_len_global, batch)
        
        self._set_rotary_required_len(max_start + q_len_global)
        
        if use_flash_attn:
            attn_weights_global = self._compute_attention_scores_manually(global_input_ids, self.kv_cache)
        else:
            out_global = self.language_model(
                input_ids=global_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_global,
                cache_position=None,
                attention_mask=None,
            )
            attn_weights_global = out_global.attentions
        
        mixed_question = local_question + "; " + global_question
        mixed_input_ids = self.processor.tokenizer(mixed_question).input_ids
        mixed_input_ids = torch.as_tensor([mixed_input_ids], device=device, dtype=torch.int)
        
        q_len_mixed = mixed_input_ids.shape[1]
        
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids = self._build_position_ids(start_pos_per_layer[layer_idx], q_len_mixed, batch)
            self._layer_position_ids[layer_idx] = position_ids
        
        position_ids_mixed = self._build_position_ids(start_pos_per_layer[0], q_len_mixed, batch)
        
        self._set_rotary_required_len(max_start + q_len_mixed)
        
        if use_flash_attn:
            attn_weights_mixed = self._compute_attention_scores_manually(mixed_input_ids, self.kv_cache)
        else:
            out_mixed = self.language_model(
                input_ids=mixed_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_mixed,
                cache_position=None,
                attention_mask=None,
            )
            attn_weights_mixed = out_mixed.attentions
        
        self._layer_position_ids.clear()
        
        print(f"GPU memory usage: {self.get_gpu_memory_usage_gb()} GB")
        current_k_states_len = self.kv_cache[0][0].shape[2]
        
        keep_indices_all_layers, visual_attn_weights_all_layers = self.prune_kv_cache_by_attention(
            attn_weights_local, attn_weights_global, attn_weights_mixed, 
            num_keep=self.kv_size
        )
        
        if current_k_states_len > self.kv_size:
            print(f"Applying KV-Cache compression due to k_states > {self.kv_size}")
            self.apply_kv_cache_pruning_strict(keep_indices_all_layers)
    
    @torch.inference_mode()
    def predict_and_compress(self, max_new_tokens=100, temperature=0.7):
        local_question, global_question = self.predict_next_question(max_new_tokens, temperature)
        self.pseudo_forward(local_question, global_question)
        
        return local_question, global_question

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, temperature=0, repetition_penalty=1.1, pseudo_forward=False):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []

        start_time = time.perf_counter()
        
        prompt = input_text['prompt']
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        past_lens_prefill = self._get_cache_seq_len_per_layer()
        start_pos_prefill = self._get_next_start_pos_per_layer()

        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]
        
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids = self._build_position_ids(start_pos_prefill[layer_idx], q_len_prefill, batch)
            self._layer_position_ids[layer_idx] = position_ids
        
        position_ids = self._build_position_ids(start_pos_prefill[0], q_len_prefill, batch)

        max_start_pos = max(start_pos_prefill) if start_pos_prefill else 0
        self._set_rotary_required_len(max_start_pos + q_len_prefill)

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=self.kv_cache,
            position_ids=position_ids,
            cache_position=None,
            attention_mask=None,
        )
        past_key_values = out.past_key_values
        logits = out.logits
        
        self._append_position_ids(start_pos_prefill, q_len_prefill)
        self._layer_position_ids.clear()

        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]

            if repetition_penalty != 1.0 and len(output_ids) > 0:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(
                    scaled_logits, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf')
                )
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])

            output_ids.append(token)
            if token in stop_token_ids:
                break

            curr_start_pos = self._get_next_start_pos_per_layer()
            
            max_curr_start = max(curr_start_pos) if curr_start_pos else 0
            self._set_rotary_required_len(max_curr_start + 1)
            
            self._layer_position_ids.clear()
            for layer_idx in range(self.num_layers):
                pos_step = torch.tensor([curr_start_pos[layer_idx]], device=device, dtype=torch.long)
                position_ids_layer = pos_step.view(1, 1)
                self._layer_position_ids[layer_idx] = position_ids_layer
            
            pos_step = torch.tensor([curr_start_pos[0]], device=device, dtype=torch.long)
            position_ids = pos_step.view(1, 1)

            out = self.language_model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=None,
                attention_mask=None,
            )
            
            if (not pseudo_forward) and (step == 0):
                end_time = time.perf_counter()
                print(f"TTFT: {end_time - start_time} seconds")
                
            logits = out.logits
            past_key_values = out.past_key_values

            self._append_position_ids(curr_start_pos, 1)
            self._layer_position_ids.clear()

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        if not pseudo_forward:
            current_question = input_text['question']
            current_options = None
            formatted_question = input_text.get('formatted_question', None)
            if formatted_question:
                option_matches = re.findall(r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)', formatted_question, re.DOTALL)
                if option_matches:
                    current_options = [opt.strip() for opt in option_matches]
            self.conv_history.append((current_question, output, current_options))
            logger.info(f"Saved conversation to history. Total conversations: {len(self.conv_history)}")

        self._truncate_kv_cache(past_lens_prefill)
        for layer_idx in range(self.num_layers):
            if (self._position_ids_cache[layer_idx] is not None and 
                self._position_ids_cache[layer_idx].numel() > past_lens_prefill[layer_idx]):
                self._position_ids_cache[layer_idx] = self._position_ids_cache[layer_idx][
                    :past_lens_prefill[layer_idx]
                ].contiguous()

        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}")
        return output
    
    def _truncate_kv_cache(self, target_lengths):
        if self.kv_cache is None:
            return
        
        truncated_cache = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
            if isinstance(target_lengths, int):
                target_len = target_lengths
            else:
                target_len = target_lengths[layer_idx]
            
            truncated_k = k_cache[:, :, :target_len, :]
            truncated_v = v_cache[:, :, :target_len, :]
            truncated_cache.append((truncated_k, truncated_v))
        
        if isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(truncated_cache)
        else:
            self.kv_cache = truncated_cache


def load_model(model_path='llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1, kv_size=None, recency_ratio=None, k_ratio=None):
    device = 'cuda'
    n_frame_tokens = 196
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    
    system_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)
    
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = LlavaOneVision_Hermes.__new__(LlavaOneVision_Hermes)
    model.__dict__ = base_model.__dict__.copy()

    Abstract_Hermes.__init__(
        model, 
        processor, 
        n_frame_tokens, 
        init_prompt_ids.tolist(), 
        n_local, 
        topk, 
        chunk_size,
        kv_size,
        recency_ratio,
        k_ratio
    )
    
    num_layers = base_model.language_model.config.num_hidden_layers
    model.num_layers = num_layers
    model._position_ids_cache = [None for _ in range(num_layers)]
    
    model.memory_adaptive_mode = 'memory_hierarchy'
    model.memory_mode = 'hierarchy'
    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(model.num_layers * model.short_term_ratio)
    model.long_term_threshold = int(model.num_layers * (1 - model.long_term_ratio))
    model.token_activity_cache = [None for _ in range(num_layers)]
    
    model.total_processed_frames = 0
    
    model._layer_position_ids = {}
    model._hook_handles = []
    
    logger.info(f'n_init: {init_prompt_ids.shape[1] if n_init is None else n_init}')
    logger.info(f'n_local: {n_local}')
    logger.info(f'topk: {topk}')
    logger.info(f'chunk_size: {chunk_size}')
    logger.info(f'kv_size: {kv_size}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
