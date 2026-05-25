import os
import json
import random
import logging

# Configuración de rutas relativas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "nim")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "nim_dataset_uniform_hint.json")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Parámetros del experimento
STATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
PROMPTS_PER_STATE = 1000

# Templates con PISTA ESTRATÉGICA (Hint). 
# La pista sugiere los números objetivo sin explicar la regla matemática completa.
TEMPLATES_WITH_HINT = [
    """Nim is a mathematical game. Players take turns removing 1 or 2 objects. The player who takes the last object wins.
Pro-tip: Try to leave your opponent with exactly 3, 6, or 9 objects to trap them.
Objects left: 5. Move: 2
Objects left: 3. Move: 1
Objects left: {N}. Move:""",

    """Let's play Nim. You can take 1 or 2 items.
Hint: Winning strategy involves leaving multiples of 3 (like 3, 6, 9) for the other player.
Remaining: 4. Take: 1
Remaining: 8. Take: 2
Remaining: {N}. Take:""",

    """Game of Nim (take 1 or 2). Strategy: force the opponent to face 3, 6, or 9 objects.
State: 6 objects. Action: 2
State: 2 objects. Action: 1
State: {N} objects. Action:""",

    """Playing Nim. Take 1 or 2. Advice: target leaving 3, 6, or 9 behind.
Left: 9 -> Play: 1
Left: 4 -> Play: 2
Left: {N} -> Play:"""
]

def main():
    logger.info("Starting uniform Nim dataset generation (WITH HINT)...")
    dataset = []
    
    random.seed(42) # Reproducibilidad

    for state in STATES:
        for _ in range(PROMPTS_PER_STATE):
            template = random.choice(TEMPLATES_WITH_HINT)
            prompt = template.replace("{N}", str(state))
            
            dataset.append({
                "prompt": prompt,
                "current_state": state,
                "is_p_position": state % 3 == 0
            })

    logger.info("Shuffling dataset to ensure uniform random distribution...")
    random.shuffle(dataset)

    logger.info(f"Saving {len(dataset)} total prompts to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    logger.info("\n--- Example Prompt Generated (With Hint) ---")
    logger.info(dataset[0]["prompt"])
    logger.info("--------------------------------------------\n")
    logger.info("Dataset with hints generated successfully!")

if __name__ == "__main__":
    main()