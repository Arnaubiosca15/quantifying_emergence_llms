import os
import sys
import csv
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian

# Configure absolute paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw","DATASET_INPUT_NIS_v2")
RESULT_FILE = os.path.join(PROJECT_ROOT, "results", "csv", "results_loop_heads.csv")

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

# Configure logging format and targets (Console and File)
LOG_FILE = os.path.join(PROJECT_ROOT, "results", "q_search_experiment.log")
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


def approx_ei(input_size, output_size, sigmas_matrix, func, num_samples, L, easy=True):
    """Approximate Effective Information using torch.slogdet for stability."""
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
        
        if sign != 0 and not torch.isinf(logabsdet):
            logdets += logabsdet.item()
        else:
            logdet_fallback = -(output_size + output_size * np.log(2 * np.pi) + dett)
            logdets += logdet_fallback.item()
            
    int_jacobian = logdets / xx.size()[0]
    term2 = -np.log(rho) + int_jacobian
    ei = max(term1 + term2, 0)
    
    return ei / output_size, -ei / np.log(rho), ei


# Experiment Configurations
IMPORTANT_HEADS = [
    (9, 9), (10, 0), (9, 6), (10, 10), (10, 6), (10, 2), 
    (10, 7), (11, 10),                               
    (7, 3), (7, 9), (8, 6), (8, 10),                   
    (5, 5), (5, 8), (5, 9), (6, 9),                    
    (3, 0), (0, 1), (0, 10),                          
    (2, 2), (4, 11),                                 
    (7, 8)                                          
]

RANDOM_HEADS = [
    (0, 0), (0, 5), (1, 8), (2, 5), (2, 10), 
    (3, 9), (4, 1), (5, 0), (6, 11), (7, 1), 
    (8, 0), (8, 5), (9, 1), (11, 0), (11, 7)
]

EXPERIMENT_HEADS = [(l, h, "IOI_Rockstar") for l, h in IMPORTANT_HEADS] + \
                   [(l, h, "Random_Noise") for l, h in RANDOM_HEADS]

Q_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
EPOCHS = 2000
BATCH_SIZE = 100
DEVICE_TRAIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IOIIntrinsicDataset(Dataset):
    """Dataset class for loading standardized attention head features."""
    def __init__(self, data_dir, layer, head):
        f_in = os.path.join(data_dir, f"layer_{layer}_head_{head}_input_V_dim64.npy")
        f_out = os.path.join(data_dir, f"layer_{layer}_head_{head}_output_Z_dim64.npy")
        
        if not os.path.exists(f_in) or not os.path.exists(f_out):
            raise FileNotFoundError(f"Missing required tensor files for head {layer},{head}")

        x_in_raw = torch.tensor(np.load(f_in)[:, -1, :], dtype=torch.float32)
        x_out_raw = torch.tensor(np.load(f_out)[:, -1, :], dtype=torch.float32)
        
        self.x_in = (x_in_raw - x_in_raw.mean(dim=0)) / (x_in_raw.std(dim=0) + 1e-6)
        self.x_out = (x_out_raw - x_out_raw.mean(dim=0)) / (x_out_raw.std(dim=0) + 1e-6)

    def __len__(self): 
        return self.x_in.shape[0]
        
    def __getitem__(self, idx): 
        return self.x_in[idx], self.x_out[idx]


class NIS(nn.Module):
    """Neural Information Squeezer with smooth activations for stable Jacobian calculation."""
    def __init__(self, dim_micro=64, dim_macro=4):
        super().__init__()
        hidden_enc = max(dim_micro, dim_macro * 2, 128) 
        hidden_dyn = max(dim_macro * 2, 32)
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_micro, hidden_enc), 
            nn.ELU(), 
            nn.Linear(hidden_enc, dim_macro)
        )
        
        self.macro_dynamics = nn.Sequential(
            nn.Linear(dim_macro, hidden_dyn), 
            nn.ELU(), 
            nn.Linear(hidden_dyn, dim_macro)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dim_macro, hidden_enc), 
            nn.ELU(), 
            nn.Linear(hidden_enc, dim_micro)
        )
        
    def forward(self, x):
        return self.decoder(self.macro_dynamics(self.encoder(x)))


def main():
    logger.info("=== STARTING OPTIMIZED Q-SEARCH EXPERIMENT ===")
    
    completed_runs = set()
    saved_ei_micros = {} 
    
    # Check for existing records to resume execution
    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Layer', 'Head', 'Type', 'Q_Dim', 'EI_Micro', 'EI_Macro', 'Emergence_CE', 'MSE_Loss'])
    else:
        with open(RESULT_FILE, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  
            for row in reader:
                if row:  
                    l, h, q = int(row[1]), int(row[2]), int(row[4])
                    completed_runs.add((l, h, q))
                    saved_ei_micros[(l, h)] = float(row[5])
        logger.info(f"Resume state: found {len(completed_runs)} completed Q dimensions.")

    # Outer loop: Iterating over specified attention heads
    for layer, head, head_type in EXPERIMENT_HEADS:
        
        if all((layer, head, q) in completed_runs for q in Q_VALUES):
            continue
            
        logger.info(f"Processing Head ({layer}, {head}) - Type: {head_type}")
        
        try:
            dataset = IOIIntrinsicDataset(DATA_DIR, layer, head)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        except Exception as e:
            logger.error(f"Error loading dataset for head ({layer},{head}): {e}")
            continue

        # Baseline training / recovery (EI Micro)
        if (layer, head) in saved_ei_micros:
            baseline_ei_micro = saved_ei_micros[(layer, head)]
            logger.info(f"  -> EI_Micro retrieved from cache: {baseline_ei_micro:.4f}")
        else:
            logger.info("  -> Training Baseline (Q=64) for EI_Micro...")
            model_base = NIS(dim_micro=64, dim_macro=64).to(DEVICE_TRAIN)
            opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
            crit_base = nn.MSELoss()
            
            model_base.train()
            for epoch in range(EPOCHS):
                for x_t, x_t_next in loader:
                    x_t, x_t_next = x_t.to(DEVICE_TRAIN), x_t_next.to(DEVICE_TRAIN)
                    opt_base.zero_grad()
                    loss = crit_base(model_base(x_t), x_t_next)
                    loss.backward()
                    opt_base.step()
                    
            model_base.cpu()
            model_base.eval()
            sigmas_micro = torch.eye(64) * 0.01
            def micro_func(x): return model_base(x.unsqueeze(0)).squeeze(0)
            
            _, _, ei_micro_tensor = approx_ei(64, 64, sigmas_micro, micro_func, num_samples=50, L=1.0)
            baseline_ei_micro = ei_micro_tensor.item() if isinstance(ei_micro_tensor, torch.Tensor) else float(ei_micro_tensor)
            saved_ei_micros[(layer, head)] = baseline_ei_micro
            logger.info(f"  -> Baseline computation finished: {baseline_ei_micro:.4f}")

        # Inner loop: Optimization across all Q values
        for q in Q_VALUES:
            if (layer, head, q) in completed_runs:
                continue
                
            model = NIS(dim_micro=64, dim_macro=q).to(DEVICE_TRAIN)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
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
                if epoch == EPOCHS - 1:
                    final_loss = total_loss / len(loader)

            model.cpu()
            model.eval()
            
            # Evaluate Macro dynamics metrics exclusively
            sigmas_macro = torch.eye(q) * 0.01
            def macro_func(m): return model.macro_dynamics(m.unsqueeze(0)).squeeze(0)
            _, _, ei_macro_tensor = approx_ei(q, q, sigmas_macro, macro_func, num_samples=50, L=1.0)

            ei_macro_val = ei_macro_tensor.item() if isinstance(ei_macro_tensor, torch.Tensor) else float(ei_macro_tensor)
            emergencia_val = ei_macro_val - baseline_ei_micro
            
            logger.info(f"  -> Q={q:<2} | Loss={final_loss:.4f} | E_CE={emergencia_val:.2f}")

            # Persist results to CSV file
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(RESULT_FILE, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    current_timestamp, 
                    layer, 
                    head, 
                    head_type, 
                    q, 
                    baseline_ei_micro, 
                    ei_macro_val, 
                    emergencia_val, 
                    final_loss
                ])

    logger.info("=== EXPERIMENT LOOP COMPLETED ===")


if __name__ == "__main__":
    main()