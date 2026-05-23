import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def evaluate(
    val_harmful: torch.Tensor,
    val_harmless: torch.Tensor,
    harm_dirs: torch.Tensor,
) -> dict:
    
    n_layers = harm_dirs.shape[0]
    results = {}

    for layer in range(n_layers):
        direction = harm_dirs[layer]

        if direction.norm() < 1e-8:
            results[layer] = 0.5
            continue

        # project activations onto harmful directional vector
        unit = direction / direction.norm()
        activations_harmful = val_harmful[:, layer, :]
        activations_harmless = val_harmless[:, layer, :]
        scores_harmful = activations_harmful @ unit
        scores_harmless = activations_harmless @ unit

        scores = torch.cat([scores_harmful, scores_harmless]).numpy()
        labels = np.concatenate([
            np.ones(len(scores_harmful)),
            np.zeros(len(scores_harmless)),
        ])

        auc = roc_auc_score(labels, scores)
        results[layer] = float(auc)

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate harmful directional vector at each layer."
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=""
    )
    parser.add_argument("--activations_dir", type=str, default="../activations")
    parser.add_argument("--vectors_dir", type=str, default="../vectors")
    parser.add_argument("--output_dir", type=str, default="../results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading validation activations …")
    val_harmful = torch.load(
        Path(args.activations_dir) / f"val_harmful{args.ext}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    
    val_harmless = torch.load(
        Path(args.activations_dir) / f"val_harmless{args.ext}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()

    print("Loading harm direction vectors …")
    harm_dirs = torch.load(
        Path(args.vectors_dir) / f"harm_direction{args.ext}.pt",
        map_location="cpu",
        weights_only=True,
    ).float()


    results = evaluate(val_harmful, val_harmless, harm_dirs)

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer}  (AUC = {results[best_layer]:.4f})")


    json_path = Path(args.output_dir) / f"roc_auc_scores{args.ext}.json"
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved scores to {json_path}")

if __name__ == "__main__":
    main()
