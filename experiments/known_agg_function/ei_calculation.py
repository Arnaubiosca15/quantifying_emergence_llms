import os
import sys
import logging
import pandas as pd
import numpy as np

# 1. Path Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "results", "csv", "tpm_logits_comparison.csv")
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "ei_calculation.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
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


def shannon_entropy(probs):
    """Calculates base-2 Shannon entropy for a probability distribution."""
    # Filter out zeros to avoid log2(0) warnings/NaNs
    probs = np.array(probs)
    probs = probs[probs > 1e-9]
    return -np.sum(probs * np.log2(probs))


def main():
    logger.info("Starting Effective Information (EI) calculation...")

    if not os.path.exists(INPUT_CSV):
        logger.error(f"CSV file not found at {INPUT_CSV}")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)
    num_prompts = len(df)
    logger.info(f"Loaded {num_prompts} prompt records.")

    # ---------------------------------------------------------
    # 2. MICRO EI CALCULATION
    # Intervention: Uniform across all N prompts (weight 1/N)
    # ---------------------------------------------------------
    # Calculate the probability of "everything else" in the vocab
    df["Micro_Prob_Other"] = 1.0 - (df["Micro_Prob_John"] + df["Micro_Prob_Mary"])
    # Clip to 0 just in case floating point math gives -0.000001
    df["Micro_Prob_Other"] = df["Micro_Prob_Other"].clip(lower=0.0)

    # H(E|C): Average entropy of individual prompts
    micro_entropies = df.apply(
        lambda row: shannon_entropy([row["Micro_Prob_John"], row["Micro_Prob_Mary"], row["Micro_Prob_Other"]]), 
        axis=1
    )
    micro_H_E_given_C = micro_entropies.mean()

    # H(E): Entropy of the average distribution
    avg_micro_john = df["Micro_Prob_John"].mean()
    avg_micro_mary = df["Micro_Prob_Mary"].mean()
    avg_micro_other = df["Micro_Prob_Other"].mean()
    
    micro_H_E = shannon_entropy([avg_micro_john, avg_micro_mary, avg_micro_other])

    ei_micro = micro_H_E - micro_H_E_given_C

    # ---------------------------------------------------------
    # 3. MACRO EI CALCULATION
    # Intervention: Uniform across the 2 Macro States (weight 1/2)
    # ---------------------------------------------------------
    # Group by the true target to simulate forcing the macro state
    grouped = df.groupby("True_Target")
    
    # Check if we have exactly our two macro states
    if set(grouped.groups.keys()) != {"John", "Mary"}:
        logger.warning(f"Unexpected macro states found: {grouped.groups.keys()}")

    # H(E|C): Entropy given the macro state intervention
    macro_H_E_given_C = 0.0
    avg_macro_john_total = 0.0
    avg_macro_mary_total = 0.0

    num_macro_states = len(grouped)
    
    for name, group in grouped:
        # Average probability distribution for this specific macro state
        p_john = group["Macro_Prob_John"].mean()
        p_mary = group["Macro_Prob_Mary"].mean()
        
        # Add to the total average (weighted evenly by 1/2 as per Hoel's definition)
        avg_macro_john_total += p_john * (1.0 / num_macro_states)
        avg_macro_mary_total += p_mary * (1.0 / num_macro_states)
        
        # Entropy for this specific macro intervention
        macro_H_E_given_C += shannon_entropy([p_john, p_mary]) * (1.0 / num_macro_states)

    # H(E): Entropy of the average macro distribution
    macro_H_E = shannon_entropy([avg_macro_john_total, avg_macro_mary_total])

    ei_macro = macro_H_E - macro_H_E_given_C
    causal_emergence = ei_macro - ei_micro

    # ---------------------------------------------------------
    # 4. RESULTS REPORTING
    # ---------------------------------------------------------
    logger.info("=== CALCULATION RESULTS ===")
    
    logger.info("--- MICRO LEVEL ---")
    logger.info(f"Avg Distribution: John={avg_micro_john:.4f}, Mary={avg_micro_mary:.4f}, Other={avg_micro_other:.4f}")
    logger.info(f"H(E_micro)      = {micro_H_E:.4f} bits")
    logger.info(f"H(E_micro|C)    = {micro_H_E_given_C:.4f} bits")
    logger.info(f"EI_micro        = {ei_micro:.4f} bits")

    logger.info("--- MACRO LEVEL ---")
    logger.info(f"Avg Distribution: John={avg_macro_john_total:.4f}, Mary={avg_macro_mary_total:.4f}")
    logger.info(f"H(E_macro)      = {macro_H_E:.4f} bits")
    logger.info(f"H(E_macro|C)    = {macro_H_E_given_C:.4f} bits")
    logger.info(f"EI_macro        = {ei_macro:.4f} bits")

    logger.info("--- CAUSAL EMERGENCE ---")
    if causal_emergence > 0:
        logger.info(f"CE = +{causal_emergence:.4f} bits (MACRO BEATS MICRO)")
    else:
        logger.info(f"CE = {causal_emergence:.4f} bits (Reductionism holds)")

if __name__ == "__main__":
    main()