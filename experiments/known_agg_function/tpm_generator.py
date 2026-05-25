import os
import sys
import json
import csv
import time
import logging
import torch
import numpy as np

# 1. External repository path injection
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

# Hugging Face compatibility patch
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

from easy_transformer.EasyTransformer import EasyTransformer

# Configure logging
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "known_agg_function.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
INPUT_JSON = os.path.join(PROJECT_ROOT, "data", "raw", "known_function", "ioi_imbalanced_70_30.json")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "csv", "tpm_logits_comparison.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

BATCH_SIZE = 50


def get_target_token_ids(tokenizer, target_names):
    """Finds the exact token IDs for the target names (usually preceded by a space)."""
    target_ids = {}
    for name in target_names:
        # GPT-2 tokenizes words differently if they have a leading space
        space_name = f" {name}"
        
        # Encode returns a list, we take the first (and hopefully only) token
        raw_id = tokenizer.encode(name)[0]
        space_id = tokenizer.encode(space_name)[0]
        
        # We prefer the spaced version for in-sentence continuation
        target_ids[name] = space_id
        
        logger.info(f"Token resolution for '{name}': No-space ID = {raw_id} | Space ID = {space_id} (Selected)")
        
    return target_ids


def main():
    logger.info("Starting known aggregation function experiment...")
    
    if not os.path.exists(INPUT_JSON):
        logger.error(f"Input JSON not found at {INPUT_JSON}. Please run the generator first.")
        return

    logger.info("Loading GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.eval()

    # Define targets and get their token IDs
    target_names = ["John", "Mary"]
    target_ids = get_target_token_ids(model.tokenizer, target_names)
    
    id_john = target_ids["John"]
    id_mary = target_ids["Mary"]

    logger.info(f"Loading dataset from {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    total_prompts = len(dataset)
    logger.info(f"Loaded {total_prompts} prompts. Starting inference loop...")

    # Initialize CSV file with headers
    with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Prompt_Index", "Prompt_Type", "True_Target", 
            "Micro_Prob_John", "Micro_Prob_Mary", 
            "Macro_Prob_John", "Macro_Prob_Mary"
        ])

    # Batch processing
    for i in range(0, total_prompts, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        
        # We need the true target (IO) to know which probability should be higher
        true_targets = [item["IO"] for item in batch]
        prompt_types = [item["type"] for item in batch]

        # Tokenize and move to GPU
        tokens = model.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = tokens["input_ids"].cuda()
        attention_mask = tokens["attention_mask"].cuda()

        with torch.no_grad():
            # Run model. Returns logits of shape [batch, seq_len, vocab_size]
            logits = model(input_ids)
            
        # We only care about the prediction for the NEXT token, 
        # which corresponds to the LAST token in the input sequence
        # We use the attention mask to find the actual last token before padding
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        batch_indices = torch.arange(len(batch)).cuda()
        last_token_logits = logits[batch_indices, sequence_lengths, :]

        # --- MICRO LEVEL CALCULATION ---
        # Probability over the entire 50,257 vocabulary
        micro_probs = torch.softmax(last_token_logits, dim=-1)
        
        micro_prob_john = micro_probs[:, id_john].cpu().numpy()
        micro_prob_mary = micro_probs[:, id_mary].cpu().numpy()

        # --- MACRO LEVEL CALCULATION ---
        # Isolate only the logits for John and Mary
        macro_logits = torch.stack([
            last_token_logits[:, id_john], 
            last_token_logits[:, id_mary]
        ], dim=-1)
        
        # Apply softmax ONLY to these two logits to force a binary decision
        macro_probs = torch.softmax(macro_logits, dim=-1)
        
        macro_prob_john = macro_probs[:, 0].cpu().numpy()
        macro_prob_mary = macro_probs[:, 1].cpu().numpy()

        # Write batch results to CSV
        with open(OUTPUT_CSV, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for j in range(len(batch)):
                writer.writerow([
                    i + j,
                    prompt_types[j],
                    true_targets[j],
                    f"{micro_prob_john[j]:.6f}",
                    f"{micro_prob_mary[j]:.6f}",
                    f"{macro_prob_john[j]:.6f}",
                    f"{macro_prob_mary[j]:.6f}"
                ])

        if (i + BATCH_SIZE) % 1000 == 0 or (i + BATCH_SIZE) >= total_prompts:
            logger.info(f"Processed {min(i + BATCH_SIZE, total_prompts)}/{total_prompts} prompts")
            
    logger.info(f"Experiment finished! Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()