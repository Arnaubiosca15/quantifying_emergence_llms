import os
import sys
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
from easy_transformer.ioi_dataset import IOIDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Output directory targeting the standard processed data folder
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw","DATASET_INPUT_NIS_v2")
os.makedirs(DATA_DIR, exist_ok=True)

N_TOTAL = 50000
BATCH_SIZE = 200


def main():
    # Load model
    logger.info("Loading pretrained GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.set_use_headwise_qkv_input(True)
    model.set_use_attn_result(True) 

    # Generate dataset
    logger.info(f"Generating dataset of {N_TOTAL} prompts...")
    ioi_dataset = IOIDataset(prompt_type="mixed", N=N_TOTAL, tokenizer=model.tokenizer, prepend_bos=False)
    all_tokens = ioi_dataset.toks.long()

    # Define targeted hooks across all layers
    hook_names = ["hook_embed"]  # Required by cache to avoid KeyError
    for layer in range(12):
        hook_names.append(f"blocks.{layer}.attn.hook_v")
        hook_names.append(f"blocks.{layer}.attn.hook_z")

    # Initialize results dictionary
    accumulator = {name: [] for name in hook_names}

    # Processing loop
    logger.info(f"Starting batch processing (Batch size: {BATCH_SIZE})")
    for i in range(0, N_TOTAL, BATCH_SIZE):
        batch_tokens = all_tokens[i : i + BATCH_SIZE].cuda()
        
        with torch.no_grad():
            _, temp_cache = model.run_with_cache(
                batch_tokens, 
                names_filter=lambda name: name in hook_names
            )
        
        # Accumulate tensors excluding the dummy hook_embed
        for name in hook_names:  
            if name != "hook_embed": 
                accumulator[name].append(temp_cache[name].detach().cpu().numpy())

        logger.info(f"Processed: {min(i + batch_tokens.shape[0], N_TOTAL)}/{N_TOTAL}")
        del temp_cache
        torch.cuda.empty_cache()

    logger.info("Concatenating and saving layer/head tensors...")
    for layer in range(12):
        layer_v_full = np.concatenate(accumulator[f"blocks.{layer}.attn.hook_v"], axis=0)
        layer_z_full = np.concatenate(accumulator[f"blocks.{layer}.attn.hook_z"], axis=0)
        
        for head in range(12):
            # Isolate specific head dimensions
            head_v = layer_v_full[:, :, head, :]
            head_z = layer_z_full[:, :, head, :]
            
            # Save using standard project paths
            v_path = os.path.join(DATA_DIR, f"layer_{layer}_head_{head}_input_V_dim64.npy")
            z_path = os.path.join(DATA_DIR, f"layer_{layer}_head_{head}_output_Z_dim64.npy")
            
            np.save(v_path, head_v)
            np.save(z_path, head_z)

    logger.info(f"OK: {N_TOTAL} prompts processed and saved successfully.")


if __name__ == "__main__":
    main()