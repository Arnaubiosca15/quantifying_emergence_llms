import os
import sys
import logging
import torch
import pandas as pd
from tqdm import tqdm

# Configuración de rutas (Igual que en tus scripts anteriores)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
if os.path.exists(EXTERNAL_DIR):
    sys.path.insert(0, EXTERNAL_DIR)

# --- PARCHE DE COMPATIBILIDAD HUGGINGFACE ---
import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
# --------------------------------------------

from easy_transformer.EasyTransformer import EasyTransformer

# Configurar logging y directorios
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, "adversarial_nim.log")
CSV_OUTPUT = os.path.join(RESULTS_DIR, "adversarial_nim_results.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Parámetros del experimento
NUM_GAMES = 1000
INITIAL_OBJECTS = 10

# Templates base para los dos agentes
PROMPT_NO_HINT = """Nim is a mathematical game. Players take turns removing 1 or 2 objects. The player who takes the last object wins.
Objects left: 5. Move: 2
Objects left: 3. Move: 1
Objects left: {N}. Move:"""

PROMPT_WITH_HINT = """Nim is a mathematical game. Players take turns removing 1 or 2 objects. The player who takes the last object wins. The best positions to look for are 3, 6 and 9.
Objects left: 5. Move: 2
Objects left: 3. Move: 1
Objects left: {N}. Move:"""

def get_model_move(model, prompt_template, current_objects, id_1, id_2):
    """
    Infiere el siguiente movimiento (1 o 2) restringiendo el espacio de logits
    y muestreando a partir de sus probabilidades (Softmax).
    """
    prompt = prompt_template.replace("{N}", str(current_objects))
    tokens = model.tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].cuda()
    attention_mask = tokens["attention_mask"].cuda()

    with torch.no_grad():
        logits = model(input_ids)
    
    # Obtener los logits del último token
    seq_len = attention_mask.sum(dim=1) - 1
    last_token_logits = logits[0, seq_len, :].squeeze()
    
    # Restringir espacio a ' 1' y ' 2'
    logit_1 = last_token_logits[id_1]
    logit_2 = last_token_logits[id_2]
    
    macro_logits = torch.stack([logit_1, logit_2])
    probs = torch.softmax(macro_logits, dim=-1)
    
    # Muestreo probabilístico basado en la confianza del modelo
    move_idx = torch.multinomial(probs, 1).item()
    
    return 1 if move_idx == 0 else 2

def main():
    logger.info("Initializing Adversarial Nim Experiment...")
    
    logger.info("Loading GPT-2 LARGE model on CUDA...")
    model = EasyTransformer.from_pretrained("gpt2-large").cuda()
    model.eval()

    # Identificar IDs de los tokens (espacio + número, por cómo tokeniza GPT-2)
    id_1 = model.tokenizer.encode(" 1")[0]
    id_2 = model.tokenizer.encode(" 2")[0]

    results = []

    logger.info(f"Starting {NUM_GAMES} games...")
    
    for game_id in tqdm(range(NUM_GAMES), desc="Playing NIM"):
        objects_left = INITIAL_OBJECTS
        
        # El 50% de las veces empieza el agente con Pista, el otro 50% el agente sin Pista
        hint_starts = (game_id % 2 == 0)
        
        current_turn_is_hint = hint_starts
        move_history = []
        
        while objects_left > 0:
            template = PROMPT_WITH_HINT if current_turn_is_hint else PROMPT_NO_HINT
            
            # Obtener el movimiento del modelo
            move = get_model_move(model, template, objects_left, id_1, id_2)
            
            # Asegurar que no coge más objetos de los que quedan
            if move > objects_left:
                move = objects_left
                
            move_history.append(f"{'Hint' if current_turn_is_hint else 'No-Hint'} takes {move}")
            
            objects_left -= move
            
            # Comprobar condición de victoria (el que toma el último objeto gana)
            if objects_left == 0:
                winner = "Hint" if current_turn_is_hint else "No-Hint"
                break
                
            # Cambiar turno
            current_turn_is_hint = not current_turn_is_hint

        # Registrar resultado de la partida
        results.append({
            "game_id": game_id,
            "hint_started": hint_starts,
            "winner": winner,
            "hint_won": 1 if winner == "Hint" else 0,
            "total_moves": len(move_history),
            "move_sequence": " | ".join(move_history)
        })

    # Guardar resultados y mostrar estadísticas básicas
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT, index=False)
    
    logger.info("=== ADVERSARIAL EXPERIMENT RESULTS ===")
    total_hint_wins = df["hint_won"].sum()
    win_rate = (total_hint_wins / NUM_GAMES) * 100
    
    wins_when_starting = df[(df["hint_started"] == True) & (df["hint_won"] == 1)].shape[0]
    wins_when_second = df[(df["hint_started"] == False) & (df["hint_won"] == 1)].shape[0]
    
    logger.info(f"Total Games Played: {NUM_GAMES}")
    logger.info(f"Agent 'HINT' Wins: {total_hint_wins} ({win_rate:.2f}%)")
    logger.info(f"Agent 'NO-HINT' Wins: {NUM_GAMES - total_hint_wins} ({100 - win_rate:.2f}%)")
    logger.info(f"Hint wins when playing FIRST: {wins_when_starting}/{NUM_GAMES//2}")
    logger.info(f"Hint wins when playing SECOND: {wins_when_second}/{NUM_GAMES//2}")
    logger.info(f"Results saved to: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
