import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian

# Configure logger to output to both console and file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE = os.path.join(PROJECT_ROOT, "results", "resultados_intrinsicos_8_6.txt")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw","DATASET_INPUT_NIS_v2")

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(RESULT_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def approx_ei(input_size, output_size, sigmas_matrix, func, num_samples, L, easy=True):
    """Approximate Effective Information on CPU."""
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
        det = torch.abs(torch.det(jac))
        if det > 1e-9:
            logdets += torch.log(det).item()
        else:
            logdet = -(output_size + output_size * np.log(2 * np.pi) + dett)
            logdets += logdet.item()
    
    int_jacobian = logdets / xx.size()[0]
    term2 = -np.log(rho) + int_jacobian
    ei = max(term1 + term2, 0)
    
    return ei / output_size, -ei / np.log(rho), ei


# Experiment configuration
LAYER, HEAD = 8, 6
DIM_MACRO = 4
EPOCHS = 2000
BATCH_SIZE = 100 
DEVICE_TRAIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IOIIntrinsicDataset(Dataset):
    """Dataset for attention head intrinsic inputs and outputs."""
    def __init__(self, data_dir, layer, head):
        f_in = os.path.join(data_dir, f"layer_{layer}_head_{head}_input_V_dim64.npy")
        f_out = os.path.join(data_dir, f"layer_{layer}_head_{head}_output_Z_dim64.npy")
        
        x_in_raw = torch.tensor(np.load(f_in)[:, -1, :], dtype=torch.float32)
        x_out_raw = torch.tensor(np.load(f_out)[:, -1, :], dtype=torch.float32)
        
        # Data normalization
        self.x_in = (x_in_raw - x_in_raw.mean(dim=0)) / (x_in_raw.std(dim=0) + 1e-6)
        self.x_out = (x_out_raw - x_out_raw.mean(dim=0)) / (x_out_raw.std(dim=0) + 1e-6)

    def __len__(self): 
        return self.x_in.shape[0]

    def __getitem__(self, idx): 
        return self.x_in[idx], self.x_out[idx]


class NIS(nn.Module):
    """Neural Information Squeezer model architecture."""
    def __init__(self, dim_micro=64, dim_macro=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_micro, 32), 
            nn.LeakyReLU(0.2), 
            nn.Linear(32, dim_macro)
        )
        self.macro_dynamics = nn.Sequential(
            nn.Linear(dim_macro, 16), 
            nn.LeakyReLU(0.2), 
            nn.Linear(16, dim_macro)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_macro, 32), 
            nn.LeakyReLU(0.2), 
            nn.Linear(32, dim_micro)
        )
        
    def forward(self, x):
        m_t = self.encoder(x)
        m_t_next = self.macro_dynamics(m_t)
        return self.decoder(m_t_next)


def main():
    if os.path.exists(RESULT_FILE): 
        os.remove(RESULT_FILE)
        
    logger.info(f"Execution started at {time.ctime()}")
    logger.info(f"Evaluating Intrinsic Emergence for Head ({LAYER}, {HEAD})")
    
    dataset = IOIIntrinsicDataset(DATA_DIR, LAYER, HEAD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = NIS(dim_micro=64, dim_macro=DIM_MACRO).to(DEVICE_TRAIN)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    logger.info(f"Training NIS with {len(dataset)} samples on {DEVICE_TRAIN} for {EPOCHS} epochs")
    model.train()
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

    logger.info("Moving model to CPU for formal EI evaluation (Jacobian computation)")
    model.cpu()
    model.eval()

    # Evaluate Micro Metrics
    sigmas_micro = torch.eye(64) * 0.01
    def micro_func(x): return model(x.unsqueeze(0)).squeeze(0)
    _, _, ei_micro = approx_ei(64, 64, sigmas_micro, micro_func, num_samples=50, L=1.0)

    # Evaluate Macro Metrics
    sigmas_macro = torch.eye(DIM_MACRO) * 0.01
    def macro_func(m): return model.macro_dynamics(m.unsqueeze(0)).squeeze(0)
    _, _, ei_macro = approx_ei(DIM_MACRO, DIM_MACRO, sigmas_macro, macro_func, num_samples=50, L=1.0)

    logger.info("--- Results (Intrinsic Emergence) ---")
    logger.info(f"EI Micro (64d -> 64d dynamics): {ei_micro:.6f}")
    logger.info(f"EI Macro ({DIM_MACRO}d -> {DIM_MACRO}d dynamics): {ei_macro:.6f}")
    logger.info(f"Causal Emergence (Macro - Micro): {ei_macro - ei_micro:.6f}")
    
    if ei_macro > ei_micro:
        logger.info("Positive causal emergence detected. Attention head processes features at macro level.")
    else:
        logger.info("No emergence detected. Head utilizes full dimensionality or network collapsed.")
        
    logger.info(f"Execution finished at {time.ctime()}")


if __name__ == "__main__":
    main()