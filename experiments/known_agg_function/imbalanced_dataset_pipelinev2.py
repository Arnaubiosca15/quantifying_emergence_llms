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
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "imbalanced_pipeline.log")
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
BATCH_SIZE = 50

# Micro granularity. Mirrors Laura's cfg["top_n"]. Adjustable.
TOP_N = 10
MICRO_COLS = [f"micro_p{i}" for i in range(TOP_N)]


# ============================================================================
#  EI CORE  (Hoel et al. 2013) — shared by micro and macro, uniform intervention
# ============================================================================

def entropy_bits(dist):
    """Shannon entropy in bits of a distribution that sums to 1."""
    dist = np.asarray(dist, dtype=float)
    dist = dist[dist > 1e-12]
    return float(-np.sum(dist * np.log2(dist)))


def compute_ei(row_a, row_b):
    """
    Effective Information for a 2-cause system under UNIFORM intervention
    UC = [0.5, 0.5]  (Hoel et al. 2013).

        EI  = H(UE) - (1/2) * [ H(row_a) + H(row_b) ]
        UE  = (row_a + row_b) / 2          (marginal effect under uniform cause)
        Eff = EI / log2(n_effect_states)   (normalised — comparable across levels)

    row_a, row_b : effect distributions (each sums to 1), one averaged row per cause.
    """
    n_eff = len(row_a)
    ue = [(a + b) / 2.0 for a, b in zip(row_a, row_b)]
    ei = max(entropy_bits(ue) - 0.5 * (entropy_bits(row_a) + entropy_bits(row_b)), 0.0)
    eff = ei / np.log2(n_eff) if n_eff > 1 else 0.0
    return ei, eff


def get_target_token_ids(tokenizer, target_names):
    """Finds the exact token IDs for the target names."""
    target_ids = {}
    for name in target_names:
        space_name = f" {name}"
        target_ids[name] = tokenizer.encode(space_name)[0]
    return target_ids


def run_inference(json_path, csv_path, model, target_ids):
    """
    Runs GPT-2 inference and saves, per prompt:
      * the renormalised top-N distribution (MICRO, top_n states), rank-ordered
      * the macro collapse {Mary, John, Other} derived FROM that same top-N
        (Other = top-N mass not on either name), so macro is a true coarse-graining
        of micro — both come from the same renormalised top-N, matching Laura.
    """
    if os.path.exists(csv_path):
        logger.info(f"Skipping inference. CSV already exists: {os.path.basename(csv_path)}")
        return

    logger.info(f"Running inference for {os.path.basename(json_path)}...")
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_john, id_mary = target_ids["John"], target_ids["Mary"]
    total_prompts = len(dataset)

    header = ["Prompt_Index", "True_Target"] + MICRO_COLS + \
             ["Macro_Prob_Mary", "Macro_Prob_John", "Macro_Prob_Other"]
    with open(csv_path, mode="w", newline="") as csvfile:
        csv.writer(csvfile).writerow(header)

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
        last_token_logits = logits[batch_indices, sequence_lengths, :]   # [B, vocab]

        # Full-vocab softmax -> top-N -> renormalise (MICRO, rank-ordered)
        full_probs = torch.softmax(last_token_logits, dim=-1)            # [B, vocab]
        top_vals, top_idx = torch.topk(full_probs, TOP_N, dim=-1)        # [B, top_n]
        micro = top_vals / top_vals.sum(dim=-1, keepdim=True)            # [B, top_n], sums to 1

        # MACRO = collapse the SAME top-N into {Mary, John, Other}
        mask_mary = (top_idx == id_mary).float()
        mask_john = (top_idx == id_john).float()
        p_mary = (micro * mask_mary).sum(dim=-1)                         # [B]
        p_john = (micro * mask_john).sum(dim=-1)                         # [B]
        p_other = (1.0 - p_mary - p_john).clamp(min=0.0)                 # [B]

        micro_np  = micro.cpu().numpy()
        p_mary_np  = p_mary.cpu().numpy()
        p_john_np  = p_john.cpu().numpy()
        p_other_np = p_other.cpu().numpy()

        with open(csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for j in range(len(batch)):
                row = [i + j, true_targets[j]]
                row += [f"{v:.8f}" for v in micro_np[j]]
                row += [f"{p_mary_np[j]:.8f}", f"{p_john_np[j]:.8f}", f"{p_other_np[j]:.8f}"]
                writer.writerow(row)

        if (i + BATCH_SIZE) % 2000 == 0:
            logger.info(f"  Processed {i + BATCH_SIZE}/{total_prompts} prompts")


def calculate_ei_from_csv(csv_path):
    """
    Builds a per-cause TPM (one averaged effect row per cause) and applies a
    UNIFORM intervention over the two causes, for BOTH levels:

      MICRO : row = position-wise mean of the renormalised top-N distributions
              within the cause group (rank-aligned, as in Laura's pipeline).
      MACRO : row = mean of {Mary, John, Other} within the cause group.

    Emergence is reported on normalised effectiveness (Eff), since micro (top_n
    states) and macro (3 states) are not comparable in raw bits.
    """
    df = pd.read_csv(csv_path)

    rows_micro, rows_macro = {}, {}
    for cause, g in df.groupby("True_Target"):
        rows_micro[cause] = g[MICRO_COLS].mean().to_numpy()                       # [top_n]
        rows_macro[cause] = g[["Macro_Prob_Mary",
                               "Macro_Prob_John",
                               "Macro_Prob_Other"]].mean().to_numpy()             # [3]

    causes = list(rows_micro.keys())
    if len(causes) != 2:
        raise ValueError(f"Expected exactly 2 causes (John, Mary), got: {causes}")
    c0, c1 = causes

    ei_micro, eff_micro = compute_ei(rows_micro[c0], rows_micro[c1])
    ei_macro, eff_macro = compute_ei(rows_macro[c0], rows_macro[c1])

    return ei_micro, ei_macro, eff_micro, eff_macro


def plot_emergence(results_list):
    """Generates the final plot (normalised effectiveness) and saves it as a PNG."""
    ratios     = [r[0] for r in results_list]
    eff_micros = [r[3] for r in results_list]
    eff_macros = [r[4] for r in results_list]
    ce_eff     = [r[5] for r in results_list]

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, eff_micros, marker='o', linestyle='-',  color='blue',  label=f'Micro Eff (top-{TOP_N} tokens)')
    plt.plot(ratios, eff_macros, marker='s', linestyle='-',  color='red',   label='Macro Eff ({Mary, John, Other})')
    plt.plot(ratios, ce_eff,     marker='^', linestyle='--', color='green', label='Causal Emergence (ΔEff)')

    plt.title('Causal Emergence vs Dataset Imbalance (Degeneracy)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset Imbalance (% ABBA Templates)', fontsize=12)
    plt.ylabel('Effectiveness  (EI / log2 n_states)', fontsize=12)
    plt.xticks(ratios, [f"{r}/{100-r}" for r in ratios])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=11)

    plot_path = os.path.join(FIGURES_DIR, "emergence_vs_imbalance.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Plot successfully saved to {plot_path}")


def main():
    logger.info("Initializing Imbalanced Pipeline (Inference -> Math -> Plot)...")

    logger.info("Loading GPT-2 model on cuda...")
    model = EasyTransformer.from_pretrained("gpt2").cuda()
    model.eval()

    target_ids = get_target_token_ids(model.tokenizer, ["John", "Mary"])
    final_results = []

    for ratio in IMBALANCE_RATIOS:
        json_path = os.path.join(DATA_DIR, f"ioi_imbalanced_{ratio}_{100-ratio}.json")
        csv_path  = os.path.join(CSV_DIR, f"tpm_logits_{ratio}_{100-ratio}.csv")

        run_inference(json_path, csv_path, model, target_ids)

        ei_micro, ei_macro, eff_micro, eff_macro = calculate_ei_from_csv(csv_path)
        ce_raw = ei_macro - ei_micro     # raw bits — different scales, reference only
        ce_eff = eff_macro - eff_micro   # normalised — the meaningful comparison

        logger.info(
            f"Results for {ratio}/{100-ratio}: "
            f"EI_micro={ei_micro:.4f}  EI_macro={ei_macro:.4f}  | "
            f"Eff_micro={eff_micro:.4f}  Eff_macro={eff_macro:.4f}  | "
            f"CE_raw={ce_raw:+.4f}  CE_eff={ce_eff:+.4f}"
        )
        final_results.append((ratio, ei_micro, ei_macro, eff_micro, eff_macro, ce_eff))

    logger.info("Generating plots...")
    plot_emergence(final_results)


if __name__ == "__main__":
    main()