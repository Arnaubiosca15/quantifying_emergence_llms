import os
import sys
import json
import random
import logging

# 1. External repository path injection
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

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
CAUSE_RATIOS = [50, 60, 70, 80, 90]   # % of prompts with IO = Mary
SWITCH_PERC = 0.0  # Controls the fraction of prompts with swapped first-clause order

# ============================================================================
#  HARDCODED VOCABULARY AND TEMPLATES
# ============================================================================
PLACES = [
    "store", "garden", "restaurant", "school",
    "hospital", "office", "house", "station"
]

OBJECTS = [
    "ring", "kiss", "bone", "basketball",
    "computer", "necklace", "drink", "snack"
]

TEMPLATES = [
    "When [IO] and [S] went to the [PLACE], [S] gave a [OBJECT] to [IO]",
    "When [IO] and [S] arrived at the [PLACE], [S] gave a [OBJECT] to [IO]",
    "After [IO] and [S] went to the [PLACE], [S] gave a [OBJECT] to [IO]",
    "While [IO] and [S] were at the [PLACE], [S] gave a [OBJECT] to [IO]",
    "When [IO] and [S] were at the [PLACE], [S] decided to give a [OBJECT] to [IO]",
    "After [IO] and [S] arrived at the [PLACE], [S] decided to give a [OBJECT] to [IO]",
    "When [IO] and [S] visited the [PLACE], [S] gave a [OBJECT] to [IO]",
    "Once [IO] and [S] reached the [PLACE], [S] gave a [OBJECT] to [IO]",
]

TEMPLATES_SWITCHED = [
    "When [S] and [IO] went to the [PLACE], [S] gave a [OBJECT] to [IO]",
    "When [S] and [IO] arrived at the [PLACE], [S] gave a [OBJECT] to [IO]",
    "After [S] and [IO] went to the [PLACE], [S] gave a [OBJECT] to [IO]",
    "While [S] and [IO] were at the [PLACE], [S] gave a [OBJECT] to [IO]",
    "When [S] and [IO] were at the [PLACE], [S] decided to give a [OBJECT] to [IO]",
    "After [S] and [IO] arrived at the [PLACE], [S] decided to give a [OBJECT] to [IO]",
    "When [S] and [IO] visited the [PLACE], [S] gave a [OBJECT] to [IO]",
    "Once [S] and [IO] reached the [PLACE], [S] gave a [OBJECT] to [IO]",
]

def generate_prompts(ratio, total_prompts, switch_perc):
    n_mary = int(total_prompts * (ratio / 100.0))
    n_john = total_prompts - n_mary

    prompts = []

    for io_name, s_name, count in [("Mary", "John", n_mary), ("John", "Mary", n_john)]:
        for _ in range(count):
            place = random.choice(PLACES)
            obj = random.choice(OBJECTS)
            tmpl_idx = random.randrange(len(TEMPLATES))

            switched = random.random() < switch_perc

            if switched:
                template = TEMPLATES_SWITCHED[tmpl_idx]
            else:
                template = TEMPLATES[tmpl_idx]

            text = (
                template
                .replace("[IO]", io_name)
                .replace("[S]", s_name)
                .replace("[PLACE]", place)
                .replace("[OBJECT]", obj)
            )

            prompts.append({
                "text": text,
                "IO": io_name,
                "S": s_name,
                "TEMPLATE_IDX": tmpl_idx,
                "[PLACE]": place,
                "[OBJECT]": obj,
                "switched": switched
            })

    random.shuffle(prompts)
    return prompts

def main():
    logger.info("Initializing Generator (Fixed Templates and Vocab)...")
    random.seed(42)

    for ratio in CAUSE_RATIOS:
        logger.info(f"Generating Ratio {ratio}/{100-ratio}...")

        dataset = generate_prompts(ratio, TOTAL_PROMPTS, SWITCH_PERC)

        output_file = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved -> {output_file}")

    logger.info("All datasets generated successfully!")

if __name__ == "__main__":
    main()