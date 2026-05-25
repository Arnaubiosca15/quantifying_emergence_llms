import os
import sys
import logging
import torch
import numpy as np

# 1. External repository path injection
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

# Hugging Face compatibility patch for older Easy-Transformer versions
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Relative paths for data input and results output
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw","single_head")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    logger.info("Loading pretrained GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.set_use_headwise_qkv_input(True)
    model.set_use_attn_result(True) 

    # Generate IOI dataset
    n_samples = 10
    ioi_dataset = IOIDataset(prompt_type="mixed", N=n_samples, tokenizer=model.tokenizer, prepend_bos=False)
    tokens = ioi_dataset.toks.long().cuda()

    # Define targeted 64-dimensional hooks
    layer = 9
    head = 9

    names_hooks = [
        f"blocks.{layer}.attn.hook_q",
        f"blocks.{layer}.attn.hook_k",
        f"blocks.{layer}.attn.hook_v",
        f"blocks.{layer}.attn.hook_z"
    ]

    # Set up cache and execution filter
    cache = {}

    def filtro_hooks(name_hook):
        return name_hook in names_hooks

    model.cache_some(cache, filtro_hooks)
    _ = model(tokens)

    # Extract tensors with shape [batch, tokens, heads, 64]
    input_values = cache[f"blocks.{layer}.attn.hook_v"]
    output_pre_proj = cache[f"blocks.{layer}.attn.hook_z"]

    # Isolate specific head 9 dimensions
    head_9_input_v = input_values[:, :, head, :].cpu().numpy()
    head_9_output_z = output_pre_proj[:, :, head, :].cpu().numpy()

    # Save target arrays using absolute paths
    ruta_in = os.path.join(DATA_DIR, f"layer_{layer}_head_{head}_input_V_dim64.npy")
    ruta_out = os.path.join(DATA_DIR, f"layer_{layer}_head_{head}_output_Z_dim64.npy")

    np.save(ruta_in, head_9_input_v)
    np.save(ruta_out, head_9_output_z)

    logger.info(f"Input tensor (V) saved successfully. Shape: {head_9_input_v.shape}")
    logger.info(f"Output tensor (Z) saved successfully. Shape: {head_9_output_z.shape}")


if __name__ == "__main__":
    main()