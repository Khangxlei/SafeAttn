import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import gc
import argparse
import math
import random

from utils.main_util import compute_concept_scores_for_turn, register_attention_amplification_hooks, run_defense_on_conversation
from utils.conv_util import build_chat_input
from utils.indices_util import get_safety_token_indices, get_last_user_turn_indices
from utils.benchmark_util import extract_numerical_answer, extract_mc_answer, evaluate_gsm8k, evaluate_mmlu, evaluate_gpqa, evaluate_hellaswag


def main():

    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks on defensive models"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--defense", type=str,
        default="VANILLA",
    )
    parser.add_argument(
        "--target_layer", type=int, default=11,
        help="Transformer layer for concept scoring",
    )
    parser.add_argument(
        "--harm_vector_path", type=str,
        default="vectors/harm_direction_qwen.pt",
    )
    parser.add_argument(
        "--amplification_factor", type=float, default=20,
        help="Amplify attention weights"
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="Upper threshold"
    )

    args = parser.parse_args()    

    print("Loading harm direction vectors …")
    c_harm_all = torch.load(
        args.harm_vector_path, map_location="cpu", weights_only=True,
    ).float()
    
    print(f"\nLoading Base Model: {args.model_name}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype="auto"
    )

    gsm8k_result = evaluate_gsm8k(base_model, base_tokenizer, c_harm_all=c_harm_all, \
                                  target_layer=args.target_layer, num_samples=100, \
                                  defense=args.defense, amplification_factor=args.amplification_factor, \
                                  tau=args.tau)
    
    if gsm8k_result:
        accuracy, num_samples = gsm8k_result
        print("-"*40)
        print("GSM8K RESULTS")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {int(accuracy * num_samples)}/{num_samples}")
        print("-"*40)

    mmlu_result = evaluate_mmlu(base_model, base_tokenizer, c_harm_all=c_harm_all, target_layer=args.target_layer, num_samples=100, defense=args.defense, amplification_factor=args.amplification_factor, tau=args.tau)
    if mmlu_result:
        print("\n" + "-"*40)
        print("MMLU RESULTS")
        print(f"Accuracy: {mmlu_result:.2%}")
        print("-"*40)
        
    gpqa_result = evaluate_gpqa(base_model, base_tokenizer, c_harm_all=c_harm_all, target_layer=args.target_layer, num_samples=100, defense=args.defense, amplification_factor=args.amplification_factor, tau=args.tau)
    if gpqa_result is not None:
        print("\n" + "-"*40)
        print("GPQA RESULTS")
        print("-"*40)
        print(f"Accuracy: {gpqa_result:.2%}")
        print("-"*40)

    hellaswag_result = evaluate_hellaswag(base_model, base_tokenizer, c_harm_all=c_harm_all, target_layer=args.target_layer, num_samples=100, defense=args.defense, amplification_factor=args.amplification_factor, tau=args.tau)
    if hellaswag_result is not None:
        print("\n" + "-"*40)
        print("HELLASWAG RESULTS")
        print(f"Accuracy: {hellaswag_result:.2%}")
        print("-"*40)
    
    if base_model:
        del base_model
    if base_tokenizer:
        del base_tokenizer
    
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

if __name__ == "__main__":
    main()
    