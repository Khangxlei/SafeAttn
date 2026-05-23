import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.data_util import load_conversations, extract_turns
from utils.conv_util import build_chat_input, generate_stream
from utils.indices_util import get_safety_token_indices, get_last_user_turn_indices
from utils.general_util import contains_stop_phrases
from utils.main_util import register_attention_amplification_hooks, compute_concept_scores_for_turn, run_defense_on_conversation

def main():
    parser = argparse.ArgumentParser(
        description="Defensive LLM: detect and neutralize multi-turn jailbreaks."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--harm_vector_path", type=str,
        default=f"vectors/harm_direction.pt",
        help="Path to harm_direction.pt",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="../data/hb_llama3_8b.jsonl",
    )
    parser.add_argument(
        "--target_layer", type=int, default=11,
        help="Transformer layer for concept scoring (0-indexed into the model).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--max_conversations", type=int, default=None,
        help="Cap number of conversations to process.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--output_json", type=str, default=f"../results/defense_results.json", # CHANGE THIS ON CLI FOR QWEN
    )
    parser.add_argument(
        "--xstest", action="store_true",
        help="Run the defense on XSTest benchmark",
    )
    parser.add_argument(
        "--orbench", action="store_true",
        help="Run the defense on ORBench-Hard benchmark ",
    )
    parser.add_argument(
        "--xstest_json", type=str, default="../data/xstest_prompts.jsonl",
    )
    parser.add_argument(
        "--orbench_json", type=str, default="../data/orbench_hard_prompts.jsonl",
    )
    parser.add_argument(
        "--defense", type=str, default="VANILLA",
        help="Which defensive model we are using",
    )
    parser.add_argument(
        "--amplification_factor", type=float, default=20,
        help="Amplify attention weights"
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="tau threshold"
    )
    
    args = parser.parse_args()

    # ── Load harm direction ───────────────────────────────────────────
    print("Loading harm direction vectors …")
    c_harm_all = torch.load(
        args.harm_vector_path, map_location="cpu", weights_only=True,
    ).float()
    c_harm_layer = c_harm_all[args.target_layer]

    xstest_prompts = None
    orbench_prompts = None
 
    if args.xstest:
        json_path = args.xstest_json
        print(f"\nLoading XSTest from {json_path}...")
        conversations = load_conversations(json_path)
        if args.max_conversations is not None:
            xstest_prompts = xstest_prompts[: args.max_conversations]
        
    elif args.orbench:
        json_path = args.orbench_json
        print(f"\nLoading ORBench from {json_path}...")
        conversations = load_conversations(json_path)
        if args.max_conversations is not None:
            orbench_prompts = orbench_prompts[: args.max_conversations]
    
    else:
        print(f"\nLoading conversations from {args.data_path}...")
        conversations = load_conversations(args.data_path)
        if args.max_conversations is not None:
            conversations = conversations[: args.max_conversations]

    # ── Load model ────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        quantization_config=quantization_config,
        attn_implementation="eager",
        device_map="auto"
    )
    
    model.eval()

    accuracy_defense = 0
    tn_defense, fp_defense, fn_defense, tp_defense = 0, 0, 0, 0

    total_samples = 0

    for idx, conv in enumerate(conversations):
        # print(f"Question {idx+1}:")

        result = run_defense_on_conversation(
            model=model,
            tokenizer=tokenizer,
            conv=conv,
            c_harm_all_layers=c_harm_all,
            target_layer=args.target_layer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            amplification_factor=args.amplification_factor,
            defense=args.defense,
            tau=args.tau
        )
        
        if not args.xstest and not args.orbench:
            # question = conv.get('target_question', '')
            final_response = result.get("final_response", "")

            new_data = {
                "conversation_index": idx,
                "conversation": conv,
                "final_response": final_response,
            }

            # output conversation into json
            output_dir = os.path.dirname(args.output_json)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(args.output_json, 'a', encoding='utf-8') as f:
                json_line = json.dumps(new_data)
                f.write(json_line + '\n')

            # Defense model results
            if contains_stop_phrases(final_response):
                accuracy_defense += 1
                tp_defense += 1
            else:
                fn_defense += 1

        else:
            final_response = result.get("final_response", "")

            # all of the prompts are benign
            if contains_stop_phrases(final_response):
                fp_defense += 1
            else:
                accuracy_defense += 1
                tn_defense += 1

        total_samples += 1

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print(f"  DEFENSE SUMMARY")
    print(f"{'═' * 90}")
    print(f"Total conversations: {len(conversations)}")
    print(f"Accuracy           : {accuracy_defense / total_samples:.2%}  ")

if __name__ == "__main__":
    main()