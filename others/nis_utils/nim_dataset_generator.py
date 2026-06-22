import os
import json
import random
import logging

# Configuración de rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "nim")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "nim_dataset_uniform.json")

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

# Few-Shot Templates para "enseñar" a GPT-2 a responder con un solo dígito.
# Terminan abruptamente en "Move: " o "Take: " para inducir la predicción del número.
TEMPLATES = [
    """Nim is a mathematical game. Players take turns removing 1 or 2 objects. The player who takes the last object wins.
Objects left: 5. Move: 2
Objects left: 3. Move: 1
Objects left: {N}. Move:""",

    """Let's play Nim. You can take 1 or 2 items.
Remaining: 4. Take: 1
Remaining: 8. Take: 2
Remaining: {N}. Take:""",

    """Game of Nim (take 1 or 2).
State: 6 objects. Action: 2
State: 2 objects. Action: 1
State: {N} objects. Action:""",

    """Playing Nim. Take 1 or 2.
Left: 9 -> Play: 1
Left: 4 -> Play: 2
Left: {N} -> Play:"""
]

def main():
    logger.info("Starting uniform Nim dataset generation...")
    dataset = []
    
    random.seed(42) # Reproducibilidad

    # Para cada estado (número de objetos del 1 al 10), generamos PROMPTS_PER_STATE ejemplos
    for state in STATES:
        for _ in range(PROMPTS_PER_STATE):
            # Elegimos una plantilla al azar para meter variabilidad lingüística (ruido micro)
            template = random.choice(TEMPLATES)
            prompt = template.replace("{N}", str(state))
            
            dataset.append({
                "prompt": prompt,
                "current_state": state,
                "is_p_position": state % 3 == 0  # True si es múltiplo de 3 (posición fría/perdedora)
            })

    # Mezclar el dataset para que la inferencia sea aleatoria y no vaya en bloques
    logger.info("Shuffling dataset to ensure uniform random distribution...")
    random.shuffle(dataset)

    # Guardar a JSON
    logger.info(f"Saving {len(dataset)} total prompts to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    # Mostrar un ejemplo por terminal para verificación visual
    logger.info("\n--- Example Prompt Generated ---")
    logger.info(dataset[0]["prompt"])
    logger.info("--------------------------------\n")
    logger.info("Dataset generated successfully!")

if __name__ == "__main__":
    main()