import os
import sys
import json
import random
import logging

# 1. External repository path injection
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

# Hugging Face compatibility patch
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Import only the raw dictionaries/templates to bypass internal hardcoded limits
from easy_transformer.ioi_dataset import ABBA_TEMPLATES, BABA_TEMPLATES, NOUNS_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Output directory setup
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "known_function")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "ioi_imbalanced_70_30.json")

N_ABBA = 700
N_BABA = 300
ALLOWED_NAMES = ["John", "Mary"]


def generate_binary_prompts(templates, num_samples, prompt_type):
    """Generates IOI prompts strictly restricted to 2 names to prevent infinite loops."""
    prompts = []
    
    for _ in range(num_samples):
        template = random.choice(templates)
        
        # Sample exactly 2 unique names without replacement
        name_a, name_b = random.sample(ALLOWED_NAMES, 2)
        
        # Inject random nouns into the template
        prompt_text = template
        for noun_class, options in NOUNS_DICT.items():
            if noun_class in prompt_text:
                prompt_text = prompt_text.replace(noun_class, random.choice(options))
        
        # Inject names based on template placeholders
        prompt_text = prompt_text.replace("[A]", name_a)
        prompt_text = prompt_text.replace("[B]", name_b)
        
        # Standard IOI Logic: 
        # ABBA -> A is Subject (S), B is Indirect Object (IO)
        # BABA -> A is Indirect Object (IO), B is Subject (S)
        if prompt_type == "ABBA":
            io_name = name_b
            s_name = name_a
        else: 
            io_name = name_a
            s_name = name_b
            
        prompts.append({
            "text": prompt_text,
            "IO": io_name,
            "S": s_name,
            "type": prompt_type
        })
        
    return prompts


def main():
    logger.info("Initializing custom constrained IOI generator...")
    random.seed(42)  # Ensure reproducibility

    logger.info(f"Generating {N_ABBA} ABBA prompts (70%)...")
    dataset_abba = generate_binary_prompts(ABBA_TEMPLATES, N_ABBA, "ABBA")
    
    logger.info(f"Generating {N_BABA} BABA prompts (30%)...")
    dataset_baba = generate_binary_prompts(BABA_TEMPLATES, N_BABA, "BABA")

    # Merge and shuffle dataset
    dataset_combined = dataset_abba + dataset_baba
    
    logger.info("Shuffling combined dataset to ensure random distribution...")
    random.shuffle(dataset_combined)

    logger.info(f"Saving dataset to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_combined, f, indent=4, ensure_ascii=False)

    logger.info("Dataset generated successfully!")


if __name__ == "__main__":
    main()