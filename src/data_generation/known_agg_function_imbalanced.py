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

# Import raw dictionaries to bypass internal limits
from easy_transformer.ioi_dataset import ABBA_TEMPLATES, BABA_TEMPLATES, NOUNS_DICT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "known_function")
os.makedirs(DATA_DIR, exist_ok=True)

TOTAL_PROMPTS = 1000
ALLOWED_NAMES = ["John", "Mary"]
# Define the percentage of ABBA templates (imbalance ratios)
IMBALANCE_RATIOS = [50, 60, 70, 80, 90]


def generate_binary_prompts(templates, num_samples, prompt_type):
    """Generates IOI prompts strictly restricted to 2 names."""
    prompts = []
    
    for _ in range(num_samples):
        template = random.choice(templates)
        name_a, name_b = random.sample(ALLOWED_NAMES, 2)
        
        prompt_text = template
        for noun_class, options in NOUNS_DICT.items():
            if noun_class in prompt_text:
                prompt_text = prompt_text.replace(noun_class, random.choice(options))
        
        prompt_text = prompt_text.replace("[A]", name_a)
        prompt_text = prompt_text.replace("[B]", name_b)
        
        if prompt_type == "ABBA":
            io_name, s_name = name_b, name_a
        else: 
            io_name, s_name = name_a, name_b
            
        prompts.append({
            "text": prompt_text,
            "IO": io_name,
            "S": s_name,
            "type": prompt_type
        })
        
    return prompts


def main():
    logger.info("Initializing multi-ratio IOI generator...")
    random.seed(42)

    for ratio in IMBALANCE_RATIOS:
        n_abba = int(TOTAL_PROMPTS * (ratio / 100.0))
        n_baba = TOTAL_PROMPTS - n_abba
        
        logger.info(f"Generating Ratio {ratio}/{100-ratio} ({n_abba} ABBA, {n_baba} BABA)...")
        
        dataset_abba = generate_binary_prompts(ABBA_TEMPLATES, n_abba, "ABBA")
        dataset_baba = generate_binary_prompts(BABA_TEMPLATES, n_baba, "BABA")
        
        dataset_combined = dataset_abba + dataset_baba
        random.shuffle(dataset_combined)
        
        output_file = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_combined, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved -> {output_file}")

    logger.info("All datasets generated successfully!")


if __name__ == "__main__":
    main()