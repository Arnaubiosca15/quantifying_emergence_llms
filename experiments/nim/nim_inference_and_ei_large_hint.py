import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de rutas
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

# Configurar logging
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "nim_inference_large_hint.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Directorios
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "nim")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Apuntamos al nuevo dataset con pistas
INPUT_JSON = os.path.join(DATA_DIR, "nim_dataset_uniform_hint.json")
BATCH_SIZE = 50

# ==========================================
# FUNCIONES MATEMÁTICAS (Hoel)
# ==========================================
def shannon_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 1e-9]
    return -np.sum(probs * np.log2(probs))

def get_macro_label(state):
    if state < 0: return "Error"
    if state % 3 == 0: return "Cold (P-pos)" 
    return "Hot (N-pos)"                     

def main():
    logger.info("Initializing Nim Inference & CE Pipeline (LARGE MODEL - WITH HINT)...")
    
    if not os.path.exists(INPUT_JSON):
        logger.error(f"Dataset not found at {INPUT_JSON}. Run generator first.")
        return

    logger.info("Loading GPT-2 LARGE model on CUDA...")
    model = EasyTransformer.from_pretrained("gpt2-large").cuda()
    model.eval()

    # Buscamos los Token IDs para " 1" y " 2"
    id_1 = model.tokenizer.encode(" 1")[0]
    id_2 = model.tokenizer.encode(" 2")[0]

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_prompts = len(dataset)
    logger.info(f"Loaded {total_prompts} prompts. Processing batches...")

    state_results = {s: {"p1": [], "p2": []} for s in range(1, 11)}

    for i in range(0, total_prompts, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        texts = [item["prompt"] for item in batch]
        current_states = [item["current_state"] for item in batch]

        tokens = model.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids, attention_mask = tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()

        with torch.no_grad():
            logits = model(input_ids)
            
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(batch)).cuda()
        last_token_logits = logits[batch_indices, sequence_lengths, :]

        # Máscara Estratégica
        macro_logits = torch.stack([last_token_logits[:, id_1], last_token_logits[:, id_2]], dim=-1)
        probs = torch.softmax(macro_logits, dim=-1)
        prob_1 = probs[:, 0].cpu().numpy()
        prob_2 = probs[:, 1].cpu().numpy()

        for j, state in enumerate(current_states):
            state_results[state]["p1"].append(prob_1[j])
            state_results[state]["p2"].append(prob_2[j])

        if (i + BATCH_SIZE) % 1000 == 0:
            logger.info(f"  Processed {min(i + BATCH_SIZE, total_prompts)}/{total_prompts} prompts")

    # ==========================================
    # 1. CONSTRUIR MATRIZ MICRO
    # ==========================================
    logger.info("Building Transition Probability Matrices (TPM)...")
    
    micro_past_states = list(range(1, 11))
    micro_future_states = list(range(9, -2, -1))
    
    micro_tpm = pd.DataFrame(0.0, index=micro_past_states, columns=micro_future_states)

    for s in micro_past_states:
        avg_p1 = np.mean(state_results[s]["p1"])
        avg_p2 = np.mean(state_results[s]["p2"])
        
        micro_tpm.loc[s, s - 1] = avg_p1
        micro_tpm.loc[s, s - 2] = avg_p2

    avg_micro_future = micro_tpm.mean(axis=0)
    H_E_micro = shannon_entropy(avg_micro_future.values)
    H_E_given_C_micro = micro_tpm.apply(lambda row: shannon_entropy(row.values), axis=1).mean()
    EI_micro = H_E_micro - H_E_given_C_micro

    # ==========================================
    # 2. CONSTRUIR MATRIZ MACRO
    # ==========================================
    macro_past_states = ["Cold (P-pos)", "Hot (N-pos)"]
    macro_future_states = ["Cold (P-pos)", "Hot (N-pos)", "Error"]
    
    macro_tpm = pd.DataFrame(0.0, index=macro_past_states, columns=macro_future_states)

    cold_starts = [3, 6, 9]
    hot_starts = [1, 2, 4, 5, 7, 8, 10]

    def calc_macro_transitions(start_states):
        trans = {"Cold (P-pos)": 0.0, "Hot (N-pos)": 0.0, "Error": 0.0}
        for s in start_states:
            avg_p1 = np.mean(state_results[s]["p1"])
            avg_p2 = np.mean(state_results[s]["p2"])
            
            trans[get_macro_label(s - 1)] += avg_p1
            trans[get_macro_label(s - 2)] += avg_p2
        
        return {k: v / len(start_states) for k, v in trans.items()}

    macro_trans_from_cold = calc_macro_transitions(cold_starts)
    macro_trans_from_hot = calc_macro_transitions(hot_starts)

    for k in macro_future_states:
        macro_tpm.loc["Cold (P-pos)", k] = macro_trans_from_cold[k]
        macro_tpm.loc["Hot (N-pos)", k] = macro_trans_from_hot[k]

    avg_macro_future = macro_tpm.mean(axis=0)
    H_E_macro = shannon_entropy(avg_macro_future.values)
    H_E_given_C_macro = macro_tpm.apply(lambda row: shannon_entropy(row.values), axis=1).mean()
    EI_macro = H_E_macro - H_E_given_C_macro
    CE = EI_macro - EI_micro

    # ==========================================
    # 3. REPORTAR RESULTADOS
    # ==========================================
    logger.info("\n=== RESULTS: CAUSAL EMERGENCE IN NIM (GPT-2 LARGE + HINT) ===")
    logger.info(f"Micro EI: {EI_micro:.4f} bits")
    logger.info(f"Macro EI: {EI_macro:.4f} bits")
    logger.info(f"Causal Emergence (CE): {CE:.4f} bits\n")

    # ==========================================
    # 4. VISUALIZACIONES
    # ==========================================
    sns.set_theme(style="white")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(micro_tpm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Nivel Micro (GPT-2 Large + Hint)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Estado Actual (Nº Objetos)")
    axes[0].set_xlabel("Estado Futuro (Nº Objetos)")

    sns.heatmap(macro_tpm, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title("Nivel Macro (GPT-2 Large + Hint)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Macro Estado Actual")
    axes[1].set_xlabel("Macro Estado Futuro")
    
    plt.tight_layout()
    plot_path_tpm = os.path.join(FIGURES_DIR, "nim_tpm_heatmaps_large_hint.png")
    plt.savefig(plot_path_tpm, dpi=300)
    
    plt.figure(figsize=(8, 6))
    metrics = ['Micro EI', 'Macro EI']
    values = [EI_micro, EI_macro]
    
    bars = plt.bar(metrics, values, color=['#4169E1', '#DC143C'])
    plt.title('Emergencia Causal (Nim) - GPT-2 Large + Hint', fontsize=15, fontweight='bold')
    plt.ylabel('Información Efectiva (bits)', fontsize=12)
    plt.ylim(0, max(values) * 1.2)
    
    color_ce = 'green' if CE > 0 else 'red'
    signo_ce = "+" if CE > 0 else ""
    plt.text(0.5, max(values)*1.05, f"ΔCE = {signo_ce}{CE:.4f} bits", ha='center', va='bottom', fontsize=14, fontweight='bold', color=color_ce)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - 0.05, f"{yval:.3f}", ha='center', va='top', color='white', fontweight='bold')

    plot_path_bar = os.path.join(FIGURES_DIR, "nim_causal_emergence_large_hint.png")
    plt.savefig(plot_path_bar, dpi=300)
    
    logger.info(f"Plots saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()