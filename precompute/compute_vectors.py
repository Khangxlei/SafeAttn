import argparse
import os
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Compute harmful directional vector for all layers"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="../activations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../vectors",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    harmful = torch.load(
        Path(args.activations_dir) / f"train_harmful{args.ext}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    harmless = torch.load(
        Path(args.activations_dir) / f"train_harmless{args.ext}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()

    # compute harmful vector 
    mean_harmful = harmful.mean(dim=0)
    mean_harmless = harmless.mean(dim=0)
    harm_dirs =  mean_harmful - mean_harmless

    out_path = Path(args.output_dir) / f"harm_direction{args.ext}.pt"
    torch.save(harm_dirs, out_path)
    print(f"Saved harm direction vectors to {out_path}")
    
if __name__ == "__main__":
    main()
