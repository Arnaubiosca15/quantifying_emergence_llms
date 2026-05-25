import os
import sys
import logging
import torch
import numpy as np

# 1. Configuración de rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")  #as easy transformer is old, we need to inject this var manually because in huggingface it no longer exists
# ---------------------------------------------------------

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Relative paths: save to data/raw/circuit
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "circuit")

N_TOTAL = 100000
BATCH_SIZE = 200

# IOI Circuit Heads to keep active
CIRCUIT_HEADS = [
    (9, 9), (10, 0), (9, 6),                     # Name Movers
    (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), 
    (10, 10), (11, 2), (11, 9),                  # Backup Name Movers
    (10, 7), (11, 10),                           # Negative Name Movers
    (7, 3), (7, 9), (8, 6), (8, 10),             # S-Inhibition Heads
    (5, 5), (5, 8), (5, 9), (6, 9),              # Induction Heads
    (3, 0), (0, 1), (0, 10),                     # Duplicate Token Heads
    (2, 2), (4, 11)                              # Previous Token Heads
]


def get_ablation_hook(heads_to_ablate):
    """Returns a hook function that zero-ablates specific heads."""
    def hook_fn(z, hook):
        # z shape: [batch, seq_len, num_heads, d_head]
        for head_idx in heads_to_ablate:
            z[:, :, head_idx, :] = 0.0
        return z
    return hook_fn


def generate_circuit_dataset():
    """Generates the dataset by running IOI prompts through an isolated circuit."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Loading GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.set_use_headwise_qkv_input(True)
    model.set_use_attn_result(True)

    logger.info(f"Generating IOI dataset ({N_TOTAL} prompts)...")
    ioi_dataset = IOIDataset(prompt_type="mixed", N=N_TOTAL, tokenizer=model.tokenizer, prepend_bos=False)
    all_tokens = ioi_dataset.toks.long()

    # Prepare ablation hooks (ablate everything NOT in the circuit)
    ablation_hooks = []
    for layer in range(12):
        heads_to_ablate = [h for h in range(12) if (layer, h) not in CIRCUIT_HEADS]
        if heads_to_ablate:
            hook_name = f"blocks.{layer}.attn.hook_z"
            ablation_hooks.append((hook_name, get_ablation_hook(heads_to_ablate)))

    hook_input_768 = "blocks.0.hook_resid_pre"
    hook_output_768 = "blocks.11.hook_resid_post"
    cache_names = [hook_input_768, hook_output_768, "hook_embed"]    # Added "hook_embed" because Easy-Transformer internally requires it to read batch_size
    accum_in = []
    accum_out = []

    logger.info(f"Starting batch processing (Batch size: {BATCH_SIZE})")

    for i in range(0, N_TOTAL, BATCH_SIZE):
        batch_tokens = all_tokens[i:i + BATCH_SIZE].cuda()
        
        with torch.no_grad():
            for hook_name, hook_fn in ablation_hooks:
                model.add_hook(hook_name, hook_fn)
                
            _, temp_cache = model.run_with_cache(
                batch_tokens, 
                names_filter=lambda name: name in cache_names
            )
            
            # CRITICAL: Reset hooks to avoid accumulation
            model.reset_hooks()
        
        # Extract only the last token from the 768-dim residual stream
        in_768 = temp_cache[hook_input_768][:, -1, :].cpu().numpy()
        out_768 = temp_cache[hook_output_768][:, -1, :].cpu().numpy()
        
        accum_in.append(in_768)
        accum_out.append(out_768)

        # Log progress every 5 batches
        if (i + BATCH_SIZE) % (BATCH_SIZE * 5) == 0 or (i + BATCH_SIZE) >= N_TOTAL:
            logger.info(f"Processed: {min(i + BATCH_SIZE, N_TOTAL)}/{N_TOTAL}")
        
        del temp_cache
        torch.cuda.empty_cache()

    logger.info("Concatenating arrays...")
    final_in_768 = np.concatenate(accum_in, axis=0)
    final_out_768 = np.concatenate(accum_out, axis=0)

    ruta_in = os.path.join(DATA_DIR, "circuit_INPUT_dim768.npy")
    ruta_out = os.path.join(DATA_DIR, "circuit_OUTPUT_dim768.npy")

    np.save(ruta_in, final_in_768)
    np.save(ruta_out, final_out_768)

    logger.info(f"Dataset successfully saved at {DATA_DIR}")
    logger.info(f"INPUT shape: {final_in_768.shape}")
    logger.info(f"OUTPUT shape: {final_out_768.shape}")


if __name__ == "__main__":
    generate_circuit_dataset()