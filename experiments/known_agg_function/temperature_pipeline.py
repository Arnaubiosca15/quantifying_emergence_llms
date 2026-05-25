import os
import sys
import json
import csv
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Path Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external", "Easy-Transformer")
sys.path.insert(0, EXTERNAL_DIR)

import transformers
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

from easy_transformer.EasyTransformer import EasyTransformer

# Configure logging
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "temperature_pipeline.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "known_function")
CSV_DIR = os.path.join(PROJECT_ROOT, "results", "csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

for d in [CSV_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

IMBALANCE_RATIOS = [50, 60, 70, 80, 90]
TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
BATCH_SIZE = 50


def shannon_entropy(probs):
    """Calculates base-2 Shannon entropy for a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 1e-9]
    return -np.sum(probs * np.log2(probs))


def get_target_token_ids(tokenizer, target_names):
    target_ids = {}
    for name in target_names:
        space_name = f" {name}"
        target_ids[name] = tokenizer.encode(space_name)[0]
    return target_ids


def run_inference_with_temp(json_path, csv_path, model, target_ids, temperature):
    """Runs inference scaling logits by temperature before softmax."""
    if os.path.exists(csv_path):
        logger.info(f"Skipping inference. CSV exists: {os.path.basename(csv_path)}")
        return

    logger.info(f"Running inference for T={temperature}, Dataset={os.path.basename(json_path)}...")
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_john, id_mary = target_ids["John"], target_ids["Mary"]
    total_prompts = len(dataset)

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prompt_Index", "True_Target", "Micro_Prob_John", "Micro_Prob_Mary", "Macro_Prob_John", "Macro_Prob_Mary"])

    for i in range(0, total_prompts, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        true_targets = [item["IO"] for item in batch]

        tokens = model.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids, attention_mask = tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()

        with torch.no_grad():
            logits = model(input_ids)
            
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(batch)).cuda()
        
        # Extracción de logits y APLICACIÓN DE LA TEMPERATURA
        last_token_logits = logits[batch_indices, sequence_lengths, :]
        scaled_logits = last_token_logits / temperature

        # Micro probabilities
        micro_probs = torch.softmax(scaled_logits, dim=-1)
        micro_prob_john = micro_probs[:, id_john].cpu().numpy()
        micro_prob_mary = micro_probs[:, id_mary].cpu().numpy()

        # Macro probabilities (Aislamos los logits ya escalados)
        macro_logits = torch.stack([scaled_logits[:, id_john], scaled_logits[:, id_mary]], dim=-1)
        macro_probs = torch.softmax(macro_logits, dim=-1)
        macro_prob_john = macro_probs[:, 0].cpu().numpy()
        macro_prob_mary = macro_probs[:, 1].cpu().numpy()

        with open(csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for j in range(len(batch)):
                writer.writerow([
                    i + j, true_targets[j],
                    f"{micro_prob_john[j]:.6f}", f"{micro_prob_mary[j]:.6f}",
                    f"{macro_prob_john[j]:.6f}", f"{macro_prob_mary[j]:.6f}"
                ])

def calculate_ei_from_csv(csv_path):
    """Calculates Micro and Macro EI from a logits CSV."""
    df = pd.read_csv(csv_path)
    
    # Micro Calculation
    df["Micro_Prob_Other"] = (1.0 - (df["Micro_Prob_John"] + df["Micro_Prob_Mary"])).clip(lower=0.0)
    micro_H_E_given_C = df.apply(lambda row: shannon_entropy([row["Micro_Prob_John"], row["Micro_Prob_Mary"], row["Micro_Prob_Other"]]), axis=1).mean()
    micro_H_E = shannon_entropy([df["Micro_Prob_John"].mean(), df["Micro_Prob_Mary"].mean(), df["Micro_Prob_Other"].mean()])
    ei_micro = micro_H_E - micro_H_E_given_C

    # Macro Calculation
    grouped = df.groupby("True_Target")
    macro_H_E_given_C, avg_macro_john_total, avg_macro_mary_total = 0.0, 0.0, 0.0
    num_macro_states = len(grouped)
    
    for _, group in grouped:
        p_john, p_mary = group["Macro_Prob_John"].mean(), group["Macro_Prob_Mary"].mean()
        avg_macro_john_total += p_john * (1.0 / num_macro_states)
        avg_macro_mary_total += p_mary * (1.0 / num_macro_states)
        macro_H_E_given_C += shannon_entropy([p_john, p_mary]) * (1.0 / num_macro_states)

    macro_H_E = shannon_entropy([avg_macro_john_total, avg_macro_mary_total])
    ei_macro = macro_H_E - macro_H_E_given_C

    return ei_micro, ei_macro

def plot_temperature_emergence(results_df):
    """Generates a multi-line plot showing CE vs Temperature for each Imbalance Ratio."""
    plt.figure(figsize=(12, 7))
    
    # Colormap para que los distintos ratios se vean bien
    colors = plt.cm.viridis(np.linspace(0, 1, len(IMBALANCE_RATIOS)))
    
    for idx, ratio in enumerate(IMBALANCE_RATIOS):
        # Filtramos los datos solo para este ratio
        ratio_data = results_df[results_df["Ratio"] == ratio].sort_values("Temperature")
        
        temps = ratio_data["Temperature"].values
        ce_values = ratio_data["CE"].values
        
        plt.plot(
            temps, 
            ce_values, 
            marker='o', 
            linewidth=2, 
            color=colors[idx], 
            label=f'Ratio {ratio}/{100-ratio}'
        )

    plt.title('Causal Emergence vs Model Temperature across Degeneracy Profiles', fontsize=15, fontweight='bold')
    plt.xlabel('Temperature (Noise / Indeterminism)', fontsize=13)
    plt.ylabel('Causal Emergence ($\Delta$ EI in bits)', fontsize=13)
    plt.xticks(TEMPERATURES)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Dataset Imbalance", loc='upper left', fontsize=11)
    
    plot_path = os.path.join(FIGURES_DIR, "emergence_vs_temperature_multi.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Plot successfully saved to {plot_path}")


def main():
    logger.info("Initializing Temperature Pipeline (Noise vs Degeneracy)...")
    
    logger.info("Loading GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.eval()
    
    target_ids = get_target_token_ids(model.tokenizer, ["John", "Mary"])
    
    # Usaremos un DataFrame para guardar todos los cruces de variables
    results_list = []

    for temp in TEMPERATURES:
        for ratio in IMBALANCE_RATIOS:
            json_path = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
            
            # Nombre de archivo único para cada cruce
            csv_name = f"tpm_temp_{str(temp).replace('.', 'p')}_ratio_{ratio}_{100-ratio}.csv"
            csv_path = os.path.join(CSV_DIR, csv_name)
            
            # 1. Inference
            run_inference_with_temp(json_path, csv_path, model, target_ids, temp)
            
            # 2. Math
            ei_micro, ei_macro = calculate_ei_from_csv(csv_path)
            ce = ei_macro - ei_micro
            
            logger.info(f"Result -> Temp: {temp} | Ratio: {ratio}/{100-ratio} | CE: {ce:.4f}")
            results_list.append({
                "Temperature": temp,
                "Ratio": ratio,
                "EI_Micro": ei_micro,
                "EI_Macro": ei_macro,
                "CE": ce
            })

    # 3. Plot
    results_df = pd.DataFrame(results_list)
    logger.info("Generating multi-line plot...")
    plot_temperature_emergence(results_df)


if __name__ == "__main__":
    main()