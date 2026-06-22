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

# IMBALANCE IS NOW ON THE CAUSE (class), NOT THE TEMPLATE.
# Each ratio is the percentage of prompts whose indirect object (IO) is Mary.
# Template (ABBA / BABA) is held fixed at 50/50 WITHIN EACH CAUSE GROUP, so it
# never co-varies with the imbalance axis and cannot leak into the TPM rows.
CAUSE_RATIOS = [50, 60, 70, 80, 90]   # % of prompts with IO = Mary


def fill_template(template_type, io_name):
    """
    Builds a single IOI prompt with a SPECIFIED indirect object (io_name) and a
    SPECIFIED template family (template_type), restricted to the two names.

    Slot convention (same as the original generator):
        ABBA -> IO = name_b , S = name_a
        BABA -> IO = name_a , S = name_b
    We invert it to *place* the requested io_name in the correct slot.
    """
    s_name = "Mary" if io_name == "John" else "John"

    templates = ABBA_TEMPLATES if template_type == "ABBA" else BABA_TEMPLATES
    prompt_text = random.choice(templates)

    # Fill the object nouns
    for noun_class, options in NOUNS_DICT.items():
        if noun_class in prompt_text:
            prompt_text = prompt_text.replace(noun_class, random.choice(options))

    # Place names so that the IO lands in the slot dictated by the template type
    if template_type == "ABBA":      # IO = B, S = A
        name_a, name_b = s_name, io_name
    else:                            # BABA: IO = A, S = B
        name_a, name_b = io_name, s_name

    prompt_text = prompt_text.replace("[A]", name_a).replace("[B]", name_b)

    return {
        "text": prompt_text,
        "IO": io_name,
        "S": s_name,
        "type": template_type
    }


def generate_cause_group(io_name, n_samples):
    """
    Generates n_samples prompts for a fixed cause (io_name), with the template
    family alternated EXACTLY 50/50 (ABBA/BABA). This fixed, balanced template
    composition is identical across every ratio, so the per-cause effect
    distribution does not drift as the class imbalance changes.
    """
    prompts = []
    for k in range(n_samples):
        template_type = "ABBA" if (k % 2 == 0) else "BABA"
        prompts.append(fill_template(template_type, io_name))
    return prompts


def main():
    logger.info("Initializing CLASS-imbalanced IOI generator (template held 50/50)...")
    random.seed(42)

    for ratio in CAUSE_RATIOS:
        n_mary = int(TOTAL_PROMPTS * (ratio / 100.0))
        n_john = TOTAL_PROMPTS - n_mary

        logger.info(f"Generating Ratio {ratio}/{100-ratio} "
                    f"({n_mary} Mary=IO, {n_john} John=IO; each 50/50 ABBA/BABA)...")

        dataset = generate_cause_group("Mary", n_mary) + \
                  generate_cause_group("John", n_john)
        random.shuffle(dataset)

        # File name kept compatible with the downstream pipeline loader.
        output_file = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved -> {output_file}")

    logger.info("All datasets generated successfully!")


if __name__ == "__main__":
    main()