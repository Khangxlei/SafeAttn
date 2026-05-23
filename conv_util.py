import torch

def build_chat_input(tokenizer, turns: list[dict], add_generation_prompt: bool = True) -> dict:
    running_messages = []
    prev_len = 0
    
    for _, turn in enumerate(turns):
        running_messages.append({"role": turn["role"], "content": turn["content"]})
        
        partial_enc = tokenizer.apply_chat_template(
            running_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        )
        
        # length of current query
        cur_len = partial_enc["input_ids"].shape[1]
        
        turn_idx = turn["turn_idx"]
        prev_len = cur_len

    # Process the final full prompt
    full_messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    
    full_enc = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=True
    )

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"]
    }

@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: str
):    
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    terminators = []
    if isinstance(tokenizer.eos_token_id, list):
        terminators.extend(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    # Check for end-of-turn tokens
    vocab = tokenizer.get_vocab()
    for special_token in ["<|im_end|>", "<|eot_id|>"]:
        if special_token in vocab:
            terminators.append(vocab[special_token])
            
    terminators = list(set(terminators))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=terminators,
        use_cache=True,
    )
        
    return tokenizer.decode(output_ids[:, input_ids.shape[1]:][0], skip_special_tokens=True)
