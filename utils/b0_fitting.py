import numpy as np
import torch
from scipy.special import sph_harm
import sys
import os
from utils.mri import fft


def _load_bart():
    """Import the BART Python wrapper (bart ecalib is used for ESPIRiT). Needs TOOLBOX_PATH."""
    toolbox = os.environ.get('TOOLBOX_PATH')
    if toolbox is None:
        raise RuntimeError('set TOOLBOX_PATH to your BART installation directory')
    if os.path.join(toolbox, 'python') not in sys.path:
        sys.path.append(os.path.join(toolbox, 'python'))
    from bart import bart
    return bart


def real_sph_harm(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2) * (-1)**m * sph_harm(m, l, phi, theta).real
    elif m < 0:
        return np.sqrt(2) * (-1)**m * sph_harm(-m, l, phi, theta).imag
    else:
        return sph_harm(0, l, phi, theta).real

def create_design_matrix(theta, phi, l_max):
    basis_list = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = real_sph_harm(l, m, theta, phi)
            basis_list.append(Y_lm)
    return np.stack(basis_list, axis=1)

def fit_spherical_harmonics(b0, mask, l_max, lambda_reg=1e-6):
    nx, ny = b0.shape
    
    # Generate theta and phi grids
    theta_vals = np.linspace(0, np.pi, nx)
    phi_vals = np.linspace(0, 2 * np.pi, ny)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    
    # Flatten arrays
    mask_flat = mask.flatten()
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    b0_flat = b0.flatten()
    
    # Apply mask to select ROI
    theta_roi = theta_flat[mask_flat]
    phi_roi = phi_flat[mask_flat]
    b0_roi = b0_flat[mask_flat]
    
    # Construct design matrix
    A_np = create_design_matrix(theta_roi, phi_roi, l_max)
    
    # Normalize columns to improve conditioning
    norms = np.linalg.norm(A_np, axis=0, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    A_np_normalized = A_np / norms
    
    # Convert to PyTorch tensors
    A = torch.from_numpy(A_np_normalized).double()
    b0_torch = torch.from_numpy(b0_roi).double().unsqueeze(1)
    
    # Add regularization (Ridge regression)
    ATA = A.T @ A
    reg_matrix = lambda_reg * torch.eye(ATA.shape[0], dtype=A.dtype)
    coeffs_normalized = torch.linalg.solve(ATA + reg_matrix, A.T @ b0_torch)
    
    # Rescale coefficients to original basis
    coeffs = coeffs_normalized / torch.from_numpy(norms.T).double()
    
    # Reconstruct fitted b0 over the entire grid
    theta_all = theta_grid.flatten()
    phi_all = phi_grid.flatten()
    A_full_np = create_design_matrix(theta_all, phi_all, l_max)
    A_full = torch.from_numpy(A_full_np).double()
    fitted_full = A_full @ coeffs
    
    return fitted_full.reshape(nx, ny).detach().numpy()

def create_polynomial_features(x, y, degree):
    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append((x ** i) * (y ** j))
    return np.stack(features, axis=1 if x.ndim == 1 else 2)

def fit_polynomial(b0, mask, degree=2):
    b0 = np.asarray(b0)
    mask = np.asarray(mask, dtype=bool)
    nx, ny = b0.shape
    
    # Generate x and y grids
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    
    # Flatten and filter by mask
    mask_flat = mask.flatten()
    xx_flat, yy_flat, b0_flat = xx.flatten(), yy.flatten(), b0.flatten()
    xx_roi, yy_roi, b0_roi = xx_flat[mask_flat], yy_flat[mask_flat], b0_flat[mask_flat]
    
    # Construct polynomial features for ROI
    X_roi = create_polynomial_features(xx_roi, yy_roi, degree)
    X_roi = torch.from_numpy(X_roi).float()
    b0_roi = torch.from_numpy(b0_roi).float().unsqueeze(1)
    
    # Solve for polynomial coefficients
    coeffs = torch.linalg.lstsq(X_roi, b0_roi, rcond=None).solution
    
    # Construct polynomial features for the entire grid
    X_full = create_polynomial_features(xx, yy, degree)
    X_full = torch.from_numpy(X_full).float()
    
    # Compute fitted b0
    b0_fitted = torch.einsum("ijk,k->ij", X_full, coeffs.squeeze())
    
    return b0_fitted.numpy()


def fit_espirit(img, b0, threshold=0.0005, calib_size=(100, 100), kernel_size=(21, 21), base_norm_factor=5):
    """
    Perform B0 fitting using ESPIRiT calibration.

    Parameters:
        img (numpy.ndarray): Input image. (nx, ny, nz)
        b0 (numpy.ndarray): B0 field for the specific case. (nx, ny, nz)
    
    Returns:
        numpy.ndarray: ESPIRiT-calibrated B0 field.
    """
    # Normalize factor based on max absolute value of b0_all
    normalize_factor = np.abs(b0).max() / np.pi * base_norm_factor
    
    # Create complex images for ESPIRiT calibration
    img1 = np.abs(img) * np.exp(1j * (b0 / normalize_factor))
    img2 = np.abs(img) + 1j * np.zeros_like(img)
    img_for_espirit = np.stack([img2, img1], axis=-1)  # [nx, ny, 1, 2]
    
    # Define BART command
    bart_cmd = f'ecalib -t {threshold} -m 1 -r {calib_size[0]}:{calib_size[1]} -k {kernel_size[0]}:{kernel_size[1]} -c 0'
    
    # Perform FFT along spatial dimensions
    k_for_espirit = fft(torch.from_numpy(img_for_espirit), dim=(0, 1)).numpy()
    
    # Apply BART ESPIRiT calibration
    bart = _load_bart()
    SEs_tmp, emaps_lr = bart(2, bart_cmd, k_for_espirit)
    
    # Compute ESPIRiT-calibrated B0 field
    b0_espirit = np.angle(SEs_tmp[:, :, :, 1] / SEs_tmp[:, :, :, 0]) * normalize_factor
    b0_espirit = np.nan_to_num(b0_espirit, nan=0)
    
    return b0_espirit