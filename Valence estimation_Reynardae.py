#!/usr/bin/env python3
# This script performs batched sentiment analysis by the Reynaerdae-7B-Chat model,
# checkpointing intermediate results and producing a final CSV.

import os            # Check for file existence
import time          # Measure durations
import csv           # Read/write CSV checkpoint files
import re            # Regex for extracting numeric scores
import argparse      # Parse command-line arguments
import random

import torch         # PyTorch for model inference
from transformers import (
    AutoTokenizer,        # Load tokenizer
    AutoModelForCausalLM, # Load causal language model
    BitsAndBytesConfig    # Configure 4-bit quantization
)

import pandas as pd  # DataFrame operations
import numpy as np   # Numeric operations
from tqdm import tqdm # Progress bar


#------------------------------------------------------------------------------------------------------------
#PROMPT TEMPLATES
#------------------------------------------------------------------------------------------------------------
# Either one of these prompts can be used in the `build_prompts` function by passing it as the `prompt_fn` argument.

# ENGLISH prompt (zero-shot setting) — the model relies on its pre-existing understanding of valence
def zero_shot_prompt_english(text):
    return (
        "You are a Dutch language expert analyzing the valence of Belgian Dutch texts. "
        "Participants responded to: \"What is going on now or since the last prompt, and how do you feel about it?\" "
        "Carefully read the response of the participant:\n"
        f'"{text}"\n'
        "Your task is to rate its sentiment from 1 (very negative) to 7 (very positive). "
        "Return ONLY a single numerical rating enclosed in brackets, e.g. [X], with no additional text. "
        "Output Format: [number]."
    )

# DUTCH prompt (zero-shot setting) — the model relies on its pre-existing understanding of valence
def zero_shot_prompt_dutch(text):
    return (
        f"Je spreekt Belgisch-Nederlands als moedertaal. \\\n"
        f"Gelieve de valentie van de {text} te beoordelen op schuifschalen van -50 (zeer slecht) tot +50 (zeer goed). \\\n"
        f"Geef ALLEEN scores tussen vierkante haken.\n\n"
        f"{text} : [Score]"
    )

# FEW-SHOT prompt — includes example inputs and outputs to guide the model
def few_shot_prompt(text):
    return (
        "You are a Dutch language expert analyzing the valence of Belgian Dutch texts. Participants "
        "responded to the question: "
        "'What is going on now or since the last prompt, and how do you feel about it?' "
        "They also rated their own emotional valence on a continuous scale from -50 (very negative) to "
        "+50 (very positive). Your task is to read each response and rate its sentiment from 1 (very "
        "negative) to 7 (very positive). Return ONLY a single numerical rating enclosed in square "
        "brackets, e.g. [X]. Provide no explanation or additional text. "
        "Output Format: [number].\n\n"
        "Below are a few examples of an input and the participant's valence rating to guide your rating:\n"
        "Input: (Text1)\nOutput: [10]\n"
        "Input: (Text2)\nOutput: [-10]\n"
        "Input: (Text3)\nOutput: [30]\n"
        "Input: (Text4)\nOutput: [45]\n"
        "Input: (Text5)\nOutput: [40]\n\n"
        f"Input: {text}\nOutput:"
    )


def parse_args():
    """
    Parse command-line arguments with defaults pointing to your dataset files.
    """
    parser = argparse.ArgumentParser(
        description="Fast batched, quantized valence scoring with Reynaerde"
    )
    # Input CSV path containing texts to score
    parser.add_argument(
        "--input_csv",
        default=r"path to the csv file",
        help="Path to the input CSV with texts to score"
    )
    # Column name containing the input texts
    parser.add_argument(
        "--text_column",
        default="column name in the csv file",
        help="Column name containing input texts"
    )
    # Checkpoint CSV for intermediate index and score
    parser.add_argument(
        "--checkpoint_csv",
        default="Reynaerde_valence_checkpoint.csv",
        help="Path to save intermediate checkpoint CSV"
    )
    # Final output CSV with added valence scores
    parser.add_argument(
        "--output_csv",
        default="Reynaerde-7B-chat_valence_estimates.csv",
        help="Path to write final scored CSV"
    )
    # HuggingFace model identifier
    parser.add_argument(
        "--model_name",
        default="ReBatch/Reynaerde-7B-Chat",
        help="HuggingFace model identifier"
    )
    # Number of texts per GPU batch
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of texts per GPU batch"
    )
    # Index to resume from if checkpoint exists
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Index to resume from if checkpoint exists"
    )
    # Maximum tokens to generate per prompt
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max tokens to generate for each prompt"
    )
    return parser.parse_args()


def build_prompts(texts, prompt_fn):
    """
    Wrap each raw text in the chosen prompt template.
    Replace zero_shot_prompt_english with zero_shot_prompt_dutch or few_shot_prompt as needed.
    """
    return [prompt_fn(t) for t in texts]


def calculate_batch_scores(batch_texts, tokenizer, model, gen_kwargs, device, prompt_fn):
    """
    Tokenize a batch of prompts, run inference, and extract numeric valence scores.
    """
    prompts = build_prompts(batch_texts, prompt_fn)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,          # Pad to longest in batch
        truncation=True,       # Truncate over max_length
        max_length=512         # Max prompt length
    ).to(device)               # Move to GPU/CPU

    with torch.inference_mode():  # No gradients needed for inference
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )

    scores = []
    for inp_ids, out_ids in zip(inputs.input_ids, outputs):
        # Extract only the newly generated tokens (beyond the prompt)
        new_tokens = out_ids[inp_ids.shape[0]:].tolist()
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Regex to find integer 1–7 within optional brackets
        m = re.search(r'\[?\s*([1-7])\s*\]?', decoded)
        scores.append(int(m.group(1)) if m else None)
    return scores


def main():
    seed = 42  # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()   # Parse CLI args or defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Choose device

    # Load tokenizer and 4-bit quantized model for speed and memory efficiency
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token       # Use EOS as padding token
    tokenizer.padding_side = "left"                 # Pad on the left for batch generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()  # Disable dropout for deterministic inference
    print(f"Loaded model in {time.time() - start_time:.2f}s on {device}")

    # Read input CSV and extract texts list
    df = pd.read_csv(args.input_csv)
    texts = df[args.text_column].fillna("").astype(str).tolist()

    # Initialize checkpoint file if it doesn't exist yet
    if not os.path.exists(args.checkpoint_csv):
        with open(args.checkpoint_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'valence_score'])

    # Determine resume index from checkpoint
    processed = sum(1 for _ in open(args.checkpoint_csv)) - 1
    start_idx = max(processed, args.resume)
    print(f"Resuming at index {start_idx}/{len(texts)}")

    # Generation parameters
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    # Batch inference loop with checkpointing
    # Replace zero_shot_prompt_english with zero_shot_prompt_dutch or few_shot_prompt as needed
    with open(args.checkpoint_csv, 'a', newline='') as fout:
        writer = csv.writer(fout)
        for i in tqdm(range(start_idx, len(texts), args.batch_size)):
            batch = texts[i:i + args.batch_size]
            try:
                scores = calculate_batch_scores(batch, tokenizer, model, gen_kwargs, device, zero_shot_prompt_english)
            except Exception as e:
                print(f"Error in batch {i}-{i+len(batch)}: {e}")
                scores = [None] * len(batch)
            for j, sc in enumerate(scores):
                writer.writerow([i + j, sc])
            fout.flush()  # Write to disk after each batch

    # Merge checkpoint with original DataFrame and save final CSV
    chk = pd.read_csv(args.checkpoint_csv)
    chk['valence_score'] = pd.to_numeric(chk['valence_score'], errors='coerce')
    df_final = df.reset_index().merge(
        chk[['idx', 'valence_score']],
        left_on='index',
        right_on='idx',
        how='left'  # Keep all rows; 'inner' would drop rows with no score
    )
    df_final['Reynaerde-7B-chat-valences'] = df_final['valence_score']
    df_final.to_csv(args.output_csv, index=False)
    print(f"Saved final results to {args.output_csv} ({len(df_final)} rows)")


if __name__ == "__main__":
    main()