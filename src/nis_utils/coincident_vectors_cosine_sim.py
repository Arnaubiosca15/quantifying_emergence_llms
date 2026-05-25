import os
import glob
import random
import logging
import numpy as np
from itertools import combinations

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data/processed")


def cosine_similarity(a, b):
    """Calculate cosine similarity between two flattened arrays."""
    a_flat, b_flat = a.flatten(), b.flatten()
    
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a_flat, b_flat) / (norm_a * norm_b)


def check_cosine_similarity():
    """Perform statistical and similarity checks on attention head tensors."""
    files_v = sorted(glob.glob(os.path.join(DATA_DIR, "*_input_V_dim64.npy")))
    files_z = sorted(glob.glob(os.path.join(DATA_DIR, "*_output_Z_dim64.npy")))
    
    if not files_v or not files_z:
        logging.error(f"Missing required tensor files in {DATA_DIR}")
        return

    # 1. Basic tensor health checks
    logging.info("--- Tensor Statistics (Health Check) ---")
    sample_v = np.load(files_v[0])
    logging.info(f"Sample file: {os.path.basename(files_v[0])}")
    logging.info(f"Shape: {sample_v.shape}")
    logging.info(f"Mean: {np.mean(sample_v):.6f} | Std: {np.std(sample_v):.6f}")
    logging.info(f"Min/Max: {np.min(sample_v):.4f} / {np.max(sample_v):.4f}\n")

    # 2. Intra-head similarity (V vs Z)
    logging.info("--- Intra-head Similarity (Input V vs Output Z) ---")
    for i in range(min(5, len(files_v), len(files_z))):
        v_tensor, z_tensor = np.load(files_v[i]), np.load(files_z[i])
        sim = cosine_similarity(v_tensor, z_tensor)
        
        head_name = os.path.basename(files_v[i]).split('_input')[0]
        logging.info(f"[{head_name}] V-Z Cosine similarity: {sim:.4f}")

    # 3. Inter-head similarity (Cross-check)
    logging.info("\n--- Inter-head Similarity (Cross-checking Z outputs) ---")
    random.seed(42)
    sample_files = random.sample(files_z, min(5, len(files_z)))
    
    for f1, f2 in combinations(sample_files, 2):
        t1, t2 = np.load(f1), np.load(f2)
        sim = cosine_similarity(t1, t2)
        
        n1 = os.path.basename(f1).replace('_output_Z_dim64.npy', '')
        n2 = os.path.basename(f2).replace('_output_Z_dim64.npy', '')
        
        if sim > 0.99:
            logging.warning(f"High similarity alert: {n1} vs {n2} (Sim: {sim:.4f})")
        else:
            logging.info(f"Normal variance: {n1} vs {n2} (Sim: {sim:.4f})")


if __name__ == "__main__":
    check_cosine_similarity()