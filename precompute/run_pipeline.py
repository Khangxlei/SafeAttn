import argparse
import os
import sys

import torch

def main():
    parser = argparse.ArgumentParser(description="Offline pipeline for harmful directional vector construction")
    parser.add_argument("--model_name",type=str,default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--activations_dir", type=str, default="../activations")
    parser.add_argument("--vectors_dir", type=str, default="../vectors")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    python = sys.executable 
    
    ext = ""
    if args.model_name == "Qwen/Qwen2.5-7B-Instruct":
        ext = "_qwen"
    elif args.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        ext = "_llama3"
    elif args.model_name == "meta-llama/Meta-Llama-3-70B-Instruct":
        ext = "_llama3_70b"
    else:
        ext = ""

    # Extract activations 
    cmd = (
        f'"{python}" extract_activations.py'
        f" --model_name {args.model_name}"
        f" --data_dir {args.data_dir}"
        f" --output_dir {args.activations_dir}"
        f" --batch_size {args.batch_size}"
        f" --max_length {args.max_length}"
        f" --dtype {args.dtype}"
        f" --device {args.device}"
        f" --ext {ext}"
    )
    os.system(cmd)

    # Compute harmful directional vector
    cmd = (
        f'"{python}" compute_vectors.py'
        f" --activations_dir {args.activations_dir}"
        f" --output_dir {args.vectors_dir}"
        f" --ext {ext}"
    )
    os.system(cmd)

    # Evaluate optimal layer
    cmd = (
        f'"{python}" evaluate_layers.py'
        f" --activations_dir {args.activations_dir}"
        f" --vectors_dir {args.vectors_dir}"
        f" --output_dir {args.results_dir}"
        f" --ext {ext}"
    )
    os.system(cmd)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()