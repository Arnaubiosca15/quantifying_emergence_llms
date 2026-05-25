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
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "model_size_pipeline.log")
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

# Experiment Parameters
IMBALANCE_RATIOS = [50, 60, 70, 80, 90]
# Validated models from your EasyTransformer installation
MODELS = ["solu-1l-old", "distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
BATCH_SIZE = 50


def shannon_entropy(probs):
    """Calculates base-2 Shannon entropy for a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 1e-9]
    return -np.sum(probs * np.log2(probs))


def get_target_token_ids(tokenizer, target_names):
    """Finds the exact token IDs for the target names."""
    target_ids = {}
    for name in target_names:
        space_name = f" {name}"
        target_ids[name] = tokenizer.encode(space_name)[0]
    return target_ids


def run_inference(json_path, csv_path, model, target_ids):
    """Runs standard inference for a specific model and dataset."""
    if os.path.exists(csv_path):
        logger.info(f"Skipping inference. CSV exists: {os.path.basename(csv_path)}")
        return

    logger.info(f"Running inference for Dataset={os.path.basename(json_path)}...")
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
        
        last_token_logits = logits[batch_indices, sequence_lengths, :]

        # Micro probabilities (over full vocab)
        micro_probs = torch.softmax(last_token_logits, dim=-1)
        micro_prob_john = micro_probs[:, id_john].cpu().numpy()
        micro_prob_mary = micro_probs[:, id_mary].cpu().numpy()

        # Macro probabilities (isolated to targets)
        macro_logits = torch.stack([last_token_logits[:, id_john], last_token_logits[:, id_mary]], dim=-1)
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

    # Macro Calculation (Uniform intervention 1/2)
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


def plot_models_for_ratio(ratio_data, ratio):
    """Generates a plot comparing models for a specific dataset imbalance ratio."""
    plt.figure(figsize=(10, 6))
    
    models = ratio_data["Model"].values
    ei_micros = ratio_data["EI_Micro"].values
    ei_macros = ratio_data["EI_Macro"].values
    ce_values = ratio_data["CE"].values

    plt.plot(models, ei_micros, marker='o', linestyle='-', color='blue', label='Micro EI')
    plt.plot(models, ei_macros, marker='s', linestyle='-', color='red', label='Macro EI')
    plt.plot(models, ce_values, marker='^', linestyle='--', color='green', label='Causal Emergence (CE)')

    plt.title(f'Causal Emergence vs Model Size (Dataset Ratio {ratio}/{100-ratio})', fontsize=14, fontweight='bold')
    plt.xlabel('Model Architecture (Complexity)', fontsize=12)
    plt.ylabel('Effective Information (bits)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=11)
    
    # Ensure X-axis is treated as categorical and ordered correctly
    plt.xticks(range(len(MODELS)), MODELS, rotation=15)
    
    plot_path = os.path.join(FIGURES_DIR, f"emergence_vs_models_ratio_{ratio}_{100-ratio}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Plot saved to {plot_path}")


def main():
    logger.info("Initializing Model Size Pipeline...")
    results_list = []

    # Loop over models
    for model_name in MODELS:
        logger.info(f"========== LOADING MODEL: {model_name} ==========")
        
        # Load model to GPU
        model = EasyTransformer.from_pretrained(model_name).cuda()
        model.eval()
        
        target_ids = get_target_token_ids(model.tokenizer, ["John", "Mary"])
        
        # Loop over imbalance ratios
        for ratio in IMBALANCE_RATIOS:
            json_path = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
            
            # Sanitize model name for file path
            safe_model_name = model_name.replace("/", "_").replace("-", "_")
            csv_name = f"tpm_model_{safe_model_name}_ratio_{ratio}_{100-ratio}.csv"
            csv_path = os.path.join(CSV_DIR, csv_name)
            
            # 1. Run Inference
            run_inference(json_path, csv_path, model, target_ids)
            
            # 2. Compute Math
            ei_micro, ei_macro = calculate_ei_from_csv(csv_path)
            ce = ei_macro - ei_micro
            
            logger.info(f"Result -> Model: {model_name} | Ratio: {ratio}/{100-ratio} | CE: {ce:.4f}")
            results_list.append({
                "Model": model_name,
                "Ratio": ratio,
                "EI_Micro": ei_micro,
                "EI_Macro": ei_macro,
                "CE": ce
            })

        # Free VRAM before loading the next model
        del model
        torch.cuda.empty_cache()

    # 3. Generate individual plots for each ratio
    results_df = pd.DataFrame(results_list)
    results_df['Model'] = pd.Categorical(results_df['Model'], categories=MODELS, ordered=True)
    
    logger.info("Generating individual plots for each ratio...")
    for ratio in IMBALANCE_RATIOS:
        ratio_data = results_df[results_df["Ratio"] == ratio].sort_values("Model")
        plot_models_for_ratio(ratio_data, ratio)

    logger.info("Pipeline completely finished!")


if __name__ == "__main__":
    main()