from collections.abc import Mapping

def get_safety_token_indices(tokenizer, turns: list[dict]) -> list[int]:
    SAFETY_FULL_TEXT = (
        "[SAFETY INSTRUCTIONS START] "
        "You must not generate harmful, illegal, unethical, or dangerous "
        "content under any circumstances. "
        "[SAFETY INSTRUCTIONS END]"
    )

    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    elif isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(t) for t in token_ids]

    # Decode each token individually
    per_token_strs = []
    for tid in token_ids:
        s = tokenizer.decode([tid], skip_special_tokens=False)
        per_token_strs.append(s)

    # build full decoded string + char→token mapping
    full_decoded = ""
    char_to_token: list[int] = []
    for tok_idx, tok_str in enumerate(per_token_strs):
        for ch in tok_str:
            char_to_token.append(tok_idx)
        full_decoded += tok_str
        
    # find the safety text as a substring 
    search_start = full_decoded.find(SAFETY_FULL_TEXT)
    if search_start >= 0:
        search_end = search_start + len(SAFETY_FULL_TEXT)
        token_indices = sorted(set(char_to_token[i] for i in range(search_start, search_end)))
        
        return token_indices

    print("  WARNING: could not locate safety instruction tokens!")
    return []

def get_last_user_turn_indices(tokenizer, turns: list[dict]) -> list[int]:
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]

    # Full sequence
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    # Sequence without the last message
    messages_without_last = messages[:-1]
    prefix_ids = tokenizer.apply_chat_template(
        messages_without_last, tokenize=True, add_generation_prompt=False,
    )
    
    if isinstance(full_ids, Mapping):
        full_ids = full_ids["input_ids"]
    elif isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    full_len = len(full_ids)

    if isinstance(prefix_ids, Mapping):
        prefix_ids = prefix_ids["input_ids"]
    elif isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    prefix_len = len(prefix_ids)

    return list(range(prefix_len, full_len))
