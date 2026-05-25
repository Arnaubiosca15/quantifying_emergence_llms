import os
import glob
import logging
import numpy as np

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data/processed")


def check_exact_matches(file_pattern):
    """Check for exact duplicate tensors across the dataset."""
    search_path = os.path.join(DATA_DIR, file_pattern)
    files = sorted(glob.glob(search_path))
    
    if not files:
        logging.error(f"No files found matching pattern: {file_pattern}")
        return

    ref_file = files[0]
    logging.info(f"Analyzing subset: {file_pattern.replace('*_', '')}")
    logging.info(f"Reference file: {os.path.basename(ref_file)}")
    
    ref_tensor = np.load(ref_file)
    matches = []
    
    # Compare against a small subset to evaluate systemic errors
    for file in files[1:21]: 
        tensor = np.load(file)
        short_name = os.path.basename(file)
        
        if ref_tensor.shape != tensor.shape:
            logging.error(f"Shape mismatch in {short_name}: expected {ref_tensor.shape}, got {tensor.shape}")
            continue
            
        match_pct = np.mean(ref_tensor == tensor) * 100
        matches.append(match_pct)
        
        if match_pct == 100.0:
            logging.warning(f"EXACT CLONE DETECTED: {short_name} ({match_pct:.2f}% match)")
        elif match_pct > 90.0:
            logging.warning(f"High similarity anomaly: {short_name} ({match_pct:.2f}% match)")
        else:
            logging.info(f"Standard deviation: {short_name} ({match_pct:.2f}% match)")

    avg_match = np.mean(matches) if matches else 0
    logging.info(f"Summary for {file_pattern}: Average match = {avg_match:.2f}%\n")


if __name__ == "__main__":
    check_exact_matches("*_input_V_dim64.npy")
    check_exact_matches("*_output_Z_dim64.npy")