import gc
import math
import torch
import numpy as np

from data_util import extract_turns
from conv_util import build_chat_input, generate_stream
from indices_util import get_safety_token_indices, get_last_user_turn_indices

def register_attention_amplification_hooks(
    model,
    last_user_indices: list[int],
    safety_indices: list[int],
    amplification_factor: float = 5.0,
) -> list:
   
    hooks = []
    user_set = set(last_user_indices)
    safety_set = set(safety_indices)

    log_boost = math.log(amplification_factor)

    for layer in model.model.layers:
        def _make_hook(user_positions, safety_positions, boost):
            def _hook(module, args, kwargs):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if hidden_states is None:
                    return
                seq_len = hidden_states.shape[1]

                attn_mask = kwargs.get("attention_mask", None)
                if attn_mask is None:
                    return

                modified_mask = attn_mask.clone()

                if seq_len <= 1:
                    kv_len = modified_mask.shape[-1]
                    for s_pos in safety_positions:
                        if s_pos < kv_len:
                            modified_mask[:, :, 0, s_pos] += boost
                else:
                    # Prefill step: boost only last-user-turn query positions.
                    for q_pos in user_positions:
                        if q_pos < seq_len:
                            for s_pos in safety_positions:
                                if s_pos < seq_len and s_pos <= q_pos:
                                    modified_mask[:, :, q_pos, s_pos] += boost

                kwargs["attention_mask"] = modified_mask
                return

            return _hook


        h = layer.self_attn.register_forward_pre_hook(
            _make_hook(user_set, safety_set, log_boost),
            with_kwargs=True,
        )
        hooks.append(h)

    return hooks

@torch.no_grad()
def compute_concept_scores_for_turn(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    current_turn_start: int,
    c_harm: torch.Tensor,
    target_layer: int = 11,
    device: str = "cuda",
) -> torch.Tensor:

    input_ids      = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    c_harm         = c_harm.to(model.device).float()
    c_unit         = c_harm / c_harm.norm()

    captured = {}

    # extract q and k
    def qk_hook(module, input, output):
        captured[module._hook_name] = output.detach()

    target_attn = model.model.layers[target_layer].self_attn
    target_attn.q_proj._hook_name = "q"
    target_attn.k_proj._hook_name = "k"

    hooks = [
        target_attn.q_proj.register_forward_hook(qk_hook),
        target_attn.k_proj.register_forward_hook(qk_hook),
    ]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=False,
    )

    for h in hooks:
        h.remove()    

    # reconstruct q and k
    q = captured["q"][0].float()
    k = captured["k"][0].float()
    
    config = model.config
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    d_k = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    seq_len = q.shape[0]

    q = q.view(seq_len, n_heads,    d_k).transpose(0, 1)
    k = k.view(seq_len, n_kv_heads, d_k).transpose(0, 1)
    kv_repeats = n_heads // n_kv_heads
    k = k.repeat_interleave(kv_repeats, dim=0)

    scale = d_k ** 0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    # attention modification via mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    current_device = attn_scores.device
    local_causal_mask = causal_mask.to(current_device)

    attn_scores = attn_scores + local_causal_mask.unsqueeze(0)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_avg     = attn_weights.mean(dim=0)

    
    # calculate projection
    h = outputs.hidden_states[target_layer + 1][0].float()
    h_norm = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    local_c_unit = c_unit.to(h.device)
    harm_proj = h_norm @ local_c_unit

    harm_proj_scale = 10
    local_harm_proj = harm_proj.to(attn_avg.device)
    weighted_harm = attn_avg * local_harm_proj * harm_proj_scale # scale for readability
    concept_scores = weighted_harm.max(dim=-1).values

    return concept_scores[current_turn_start:].cpu()

def run_defense_on_conversation(
    model,
    tokenizer,
    conv: dict,
    c_harm_all_layers: torch.Tensor,
    target_layer: int = 11,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    amplification_factor = 20,
    defense = "VANILLA", 
    tau = 0.3
) -> dict:

    gc.collect()
    torch.cuda.empty_cache()
    
    
    turns = extract_turns(model, tokenizer, max_new_tokens, device, defense, conv)
    user_turns = [t for t in turns if t["role"] == "user"]
    c_harm = c_harm_all_layers[target_layer].float()

    accumulated_ids = []
    total_concept_scores_on_flagged_tokens = 0

    for turn_idx, turn in enumerate(user_turns):
        turn_token_ids = tokenizer.encode(
            turn["content"],
            add_special_tokens=(turn_idx == 0),
        )

        turn_start = len(accumulated_ids)
        accumulated_ids.extend(turn_token_ids)

        print(f"     Computing concept scores at layer {target_layer}…")

        input_ids_tensor     = torch.tensor([accumulated_ids], dtype=torch.long)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        scores = compute_concept_scores_for_turn(
            model=model,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            current_turn_start=turn_start,
            c_harm=c_harm,
            target_layer=target_layer,
            device=device,
        )

        if turn_start == 0:
            scores = scores[1:]  # skip BOS token


        flagged_pos = (scores > 0).nonzero(as_tuple=True)[0].tolist()

        print(f"Flagged tokens: {len(flagged_pos)}")

        if flagged_pos:
            for sc in scores[flagged_pos].tolist():
                # keep track of score throughout conversation
                total_concept_scores_on_flagged_tokens += sc 


    final_input_raw = build_chat_input(tokenizer, turns, add_generation_prompt=True)

    tensor_keys = {"input_ids", "attention_mask"}
    final_input = {
        k: (v.to(model.device) if k in tensor_keys else v)
        for k, v in final_input_raw.items()
    }

    result = {
        "target_question" : conv.get("target_question", ""),
        "stream_a_text"   : None,
        "stream_b_text"   : None,
        "label"           : conv.get("label", ""),
        "total_concept_scores_on_flagged_tokens" : total_concept_scores_on_flagged_tokens,
    }



    if total_concept_scores_on_flagged_tokens < tau:    

        # normal output generation for baselines
        if defense != "SA":
            stream_a_text = generate_stream(
                model, tokenizer, final_input["input_ids"],
                final_input["attention_mask"], max_new_tokens, device
            )
            result.update(stream_a_text=stream_a_text,
                        final_response=stream_a_text)
            
        # mask attention towards safety instructions
        else:
            safety_indices = get_safety_token_indices(tokenizer, turns)
            last_user_indices = get_last_user_turn_indices(tokenizer, turns)
            amplification_factor = -np.inf

            attn_hooks = register_attention_amplification_hooks(
                model, last_user_indices, safety_indices, amplification_factor,
            )
            try:
                stream_a_text = generate_stream(
                    model, tokenizer, final_input["input_ids"],
                    final_input["attention_mask"], max_new_tokens, device
                )
            finally:
                for h in attn_hooks:
                    h.remove()
            result.update(stream_a_text=stream_a_text,
                    final_response=stream_a_text)

    # score above threshold
    else:
        
        if defense != "SA": # normal output generation for baselines
            stream_a_ids = generate_stream(
                model, tokenizer, final_input["input_ids"],
                final_input["attention_mask"], max_new_tokens, device            
            )
            stream_a_text = tokenizer.decode(stream_a_ids[0], skip_special_tokens=True)

            result.update(
                stream_a_text=stream_a_text,
                final_response=stream_a_text,
            )

        else: # attention amplification
            safety_indices = get_safety_token_indices(tokenizer, turns)
            last_user_indices = get_last_user_turn_indices(tokenizer, turns)

            attn_hooks = register_attention_amplification_hooks(
                model, last_user_indices, safety_indices, amplification_factor,   
            )
            
            try:
                stream_b_ids = generate_stream(
                    model, tokenizer, final_input["input_ids"],
                    final_input["attention_mask"],
                    max_new_tokens, device
                )
                
            finally:
                for h in attn_hooks:
                    h.remove()

            stream_b_text = tokenizer.decode(stream_b_ids[0], skip_special_tokens=True)

            result.update(
                stream_b_text=stream_b_text,
                final_response=stream_b_text,
            )
            
    return result
