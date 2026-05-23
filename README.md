# SafeAttn
This is the official repository for SafeAttn. 

## Quick Start

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables
Fill out your API keys inside `.env`:
- `HF_TOKEN`: Hugging Face Token (required)
- `OPENAI_API_KEY`: OpenAI API Key for judge/attack automation (optional)

### Preliminary Study Results
#### Terminal Command
```bash
python precompute/attention_safety_experiment.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir ../results \
```

#### CLI Arguments:
- `--model`: HF model ID names. Default: `meta-llama/Meta-Llama-3-8B-Instruct`
- `--data`: Directory path to `.jsonl` data file. Default: `../data/hb_llama3_8b.jsonl`
- `--num_generate`: Max generated tokens. Default: `1024`
- `--output_dir`: Directory path for output. Default: `../results`
- `--max_conversations`: Maximum number of prompts to load from data. Default: `none`

### Harmful Directional Vector Construction
This will construct the pre-inference, one-time procedure to derive the harmful directional vector for the corresponding model. Harmful vectors for Llama3-8B, Llama3-70B, and Qwen2.5-7B are stored in the `vectors` directory.
#### Terminal Command
```bash
python precompute/run_pipeline.py \
  --batch_size 8 \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct
```

#### CLI Arguments:
- `--model`: HF model ID names. Default: `meta-llama/Meta-Llama-3-8B-Instruct`
- `--data_dir`: Directory path for data. Default: `../data`
- `--activations_dir`: Directory path to store and access latent activations. Default: `../activations`
- `--vectors_dir`: Directory path to store and access vectors. Default: `../vectors`
- `-batch_size`: Batch size. Default: `8`.


