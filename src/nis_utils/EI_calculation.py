import logging
import numpy as np
import torch
from torch.autograd.functional import jacobian

# Configure logger for mathematical warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def approx_ei(
    input_size, output_size, sigmas_matrix, func, num_samples, L, easy=True, device=None
):
    """
    Approximate Effective Information (EI) using Monte Carlo sampling.
    Returns normalized EI (d_EI), efficiency (eff), and raw components.
    """
    if device is not None:
        sigmas_matrix = sigmas_matrix.to(device)
        
    # Log-space calculation to prevent float underflow/overflow: 
    # ln((2L)^-n) = -n * ln(2L)
    log_rho = -input_size * np.log(2 * L)
    
    if easy:
        diag_elements = torch.diag(sigmas_matrix)
        log_det = torch.log(diag_elements).sum().item()
    else:
        _, logdet_s = torch.slogdet(sigmas_matrix)
        log_det = logdet_s.item()
        
    # First term of EI (entropy related)
    term1 = -(output_size + output_size * np.log(2 * np.pi) + log_det) / 2.0 
    
    # Monte Carlo sampling uniformly distributed across [-L, L]
    samples = L * 2 * (torch.rand(num_samples, input_size, device=sigmas_matrix.device) - 0.5)
    
    log_dets = 0.0
    all_zeros = True 
    
    # Evaluate the Jacobian determinant for each sample
    for i in range(samples.size(0)):
        jac = jacobian(func, samples[i, :]) 
        sign, log_abs_det = torch.slogdet(jac)
        
        if sign != 0 and not torch.isinf(log_abs_det):
            log_dets += log_abs_det.item() 
            all_zeros = False
        else:
            # Fallback penalization if determinant is 0 or infinite
            log_det_fallback = -(output_size + output_size * np.log(2 * np.pi) + log_det)
            log_dets += log_det_fallback
            
    avg_log_det_jacobian = log_dets / samples.size(0) 
    term2 = -log_rho + avg_log_det_jacobian 
    
    if all_zeros:
        logger.warning("Jacobian determinants collapsed to zero. Model might be inactive.")
        term2 = -term1
        
    # Ensure EI is strictly non-negative
    ei = max(term1 + term2, 0.0)
    
    eff = -ei / log_rho if log_rho != 0 else 0.0
    d_ei = ei / output_size
    
    return d_ei, eff, ei, term1, term2, -log_rho