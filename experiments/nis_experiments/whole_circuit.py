import os
import csv
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian

# Configure absolute paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "circuit")
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "experiment_circuit.log")
CSV_FILE = os.path.join(PROJECT_ROOT, "results", "csv", "whole_circuit.csv")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def approx_ei(input_size, output_size, sigmas_matrix, func, num_samples, L, easy=True):
    """Approximate Effective Information using log-determinant (slogdet) for 768d stability."""
    device = torch.device("cpu")
    sigmas_matrix = sigmas_matrix.to(device)
    
    rho = 1 / (2 * L)**input_size 
    
    dd = torch.diag(sigmas_matrix)
    dett = torch.log(dd).sum()
    
    term1 = - (output_size + output_size * np.log(2 * np.pi) + dett) / 2 
    
    xx = L * 2 * (torch.rand(num_samples, input_size, device=device) - 0.5)
    
    logdets = 0
    for i in range(xx.size()[0]):
        jac = jacobian(func, xx[i, :]) 
        sign, logabsdet = torch.slogdet(jac)
        
        # Guard against zero or infinite determinants in high-dimensional spaces
        if sign != 0 and not torch.isinf(logabsdet):
            logdets += logabsdet.item()
        else:
            logdet_fallback = -(output_size + output_size * np.log(2 * np.pi) + dett)
            logdets += logdet_fallback.item()
    
    int_jacobian = logdets / xx.size()[0]
    term2 = -np.log(rho) + int_jacobian
    ei = max(term1 + term2, 0)
    
    return ei / output_size, -ei / np.log(rho), ei


# Experiment Configuration
DIM_MICRO = 768
DIM_MACRO = 32
EPOCHS = 3000
BATCH_SIZE = 500 
DEVICE_TRAIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CircuitDataset(Dataset):
    """Dataset for full IOI circuit residual stream processing (768 dimensions)."""
    def __init__(self, data_dir):
        f_in = os.path.join(data_dir, "circuit_INPUT_dim768.npy")
        f_out = os.path.join(data_dir, "circuit_OUTPUT_dim768.npy")
        
        if not os.path.exists(f_in) or not os.path.exists(f_out):
            raise FileNotFoundError(f"Circuit tensors not found in {data_dir}")

        x_in_raw = torch.tensor(np.load(f_in), dtype=torch.float32)
        x_out_raw = torch.tensor(np.load(f_out), dtype=torch.float32)
        
        # Z-score normalization
        self.x_in = (x_in_raw - x_in_raw.mean(dim=0)) / (x_in_raw.std(dim=0) + 1e-6)
        self.x_out = (x_out_raw - x_out_raw.mean(dim=0)) / (x_out_raw.std(dim=0) + 1e-6)

    def __len__(self): 
        return self.x_in.shape[0]

    def __getitem__(self, idx): 
        return self.x_in[idx], self.x_out[idx]


class NIS(nn.Module):
    """Neural Information Squeezer adapted for 768-dimensional residual streams."""
    def __init__(self, dim_micro=768, dim_macro=32):
        super().__init__()
        
        hidden_enc = max(dim_micro // 2, 256)
        hidden_dyn = max(dim_macro * 2, 64)
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_micro, hidden_enc), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_enc, dim_macro)
        )
        self.macro_dynamics = nn.Sequential(
            nn.Linear(dim_macro, hidden_dyn), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dyn, dim_macro)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_macro, hidden_enc), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_enc, dim_micro)
        )
        
    def forward(self, x):
        m_t = self.encoder(x)
        m_t_next = self.macro_dynamics(m_t)
        return self.decoder(m_t_next)


def main():
    logger.info(f"Execution started at {time.ctime()}")
    logger.info("Evaluating Causal Emergence for Full IOI Circuit (768d -> 768d)")
    
    # Check if CSV exists to write headers later
    csv_exists = os.path.isfile(CSV_FILE)
    
    dataset = CircuitDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = NIS(dim_micro=DIM_MICRO, dim_macro=DIM_MACRO).to(DEVICE_TRAIN)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    logger.info(f"Training NIS with {len(dataset)} samples on {DEVICE_TRAIN} for {EPOCHS} epochs")
    model.train()
    
    final_loss = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_t, x_t_next in loader:
            x_t, x_t_next = x_t.to(DEVICE_TRAIN), x_t_next.to(DEVICE_TRAIN)
            optimizer.zero_grad()
            loss = criterion(model(x_t), x_t_next)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 200 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} - MSE Loss: {total_loss/len(loader):.6f}")
        if epoch == EPOCHS - 1:
            final_loss = total_loss / len(loader)

    logger.info("Moving model to CPU for formal EI evaluation (Jacobian computation)")
    model.cpu()
    model.eval()

    # Evaluate Micro Metrics
    logger.info("Computing Micro Effective Information (This may take a moment due to 768d Jacobians)...")
    sigmas_micro = torch.eye(DIM_MICRO) * 0.01
    def micro_func(x): return model(x.unsqueeze(0)).squeeze(0)
    _, _, ei_micro_tensor = approx_ei(DIM_MICRO, DIM_MICRO, sigmas_micro, micro_func, num_samples=50, L=1.0)

    # Evaluate Macro Metrics
    logger.info("Computing Macro Effective Information...")
    sigmas_macro = torch.eye(DIM_MACRO) * 0.01
    def macro_func(m): return model.macro_dynamics(m.unsqueeze(0)).squeeze(0)
    _, _, ei_macro_tensor = approx_ei(DIM_MACRO, DIM_MACRO, sigmas_macro, macro_func, num_samples=50, L=1.0)

    # Clean variables for logging and CSV
    ei_micro = ei_micro_tensor.item() if isinstance(ei_micro_tensor, torch.Tensor) else float(ei_micro_tensor)
    ei_macro = ei_macro_tensor.item() if isinstance(ei_macro_tensor, torch.Tensor) else float(ei_macro_tensor)
    emergence_ce = ei_macro - ei_micro

    logger.info("--- Results (Intrinsic Emergence) ---")
    logger.info(f"EI Micro ({DIM_MICRO}d -> {DIM_MICRO}d dynamics): {ei_micro:.6f}")
    logger.info(f"EI Macro ({DIM_MACRO}d -> {DIM_MACRO}d dynamics): {ei_macro:.6f}")
    logger.info(f"Causal Emergence (Macro - Micro): {emergence_ce:.6f}")
    
    if emergence_ce > 0:
        logger.info("Positive causal emergence detected! The IOI circuit operates reliably in a lower-dimensional macro-state.")
    else:
        logger.info("No emergence detected. The circuit requires full dimensionality or the bottleneck lost critical information.")

    # Save to CSV
    logger.info(f"Saving results to {CSV_FILE}")
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(CSV_FILE, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(['Timestamp', 'Experiment', 'Q_Dim', 'EI_Micro', 'EI_Macro', 'Emergence_CE', 'MSE_Loss'])
        writer.writerow([
            current_timestamp, 
            "Full_IOI_Circuit_768", 
            DIM_MACRO, 
            ei_micro, 
            ei_macro, 
            emergence_ce, 
            final_loss
        ])
        
    logger.info(f"Execution finished at {time.ctime()}")


if __name__ == "__main__":
    main()