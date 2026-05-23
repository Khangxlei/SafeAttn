import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["instruction"] for item in data]

@torch.no_grad()
def extract_last_token_hidden_states(model, tokenizer, prompts: list[str], batch_size: int = 8,max_length: int = 1024, device: str = "cuda") -> torch.Tensor:
    all_hidden: list[torch.Tensor] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)

        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1
        stacked = torch.stack(outputs.hidden_states, dim=0)
        batch_indices = torch.arange(stacked.size(1), device=device)
        last_token = stacked[:, batch_indices, seq_lengths, :]
        last_token = last_token.permute(1, 0, 2).cpu()
        all_hidden.append(last_token)

    return torch.cat(all_hidden, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden-state activations"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="",
        help="Extension to append to output filenames",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="activations"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
    )
    model.eval()

    for split in ("train", "val"):
        harmful_path = Path(args.data_dir) / f"harmful_{split}.json"
        harmless_path = Path(args.data_dir) / f"harmless_{split}.json"

        harmful_prompts = load_prompts(str(harmful_path))
        harmless_prompts = load_prompts(str(harmless_path))

        print(f"Extracting harmful activations …")
        h_harmful = extract_last_token_hidden_states(
            model, tokenizer, harmful_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

        print(f"  Extracting harmless activations …")
        h_harmless = extract_last_token_hidden_states(
            model, tokenizer, harmless_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

        out_harmful = Path(args.output_dir) / f"{split}_harmful{args.ext}.pt"
        out_harmless = Path(args.output_dir) / f"{split}_harmless{args.ext}.pt"
        torch.save(h_harmful, out_harmful)
        torch.save(h_harmless, out_harmless)
        print(f"Saved {out_harmful}")
        print(f"Saved {out_harmless}")


if __name__ == "__main__":
    main()
