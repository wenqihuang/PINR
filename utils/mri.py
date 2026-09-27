import torch
import numpy as np


def fft(x, dim, norm='ortho'):
    """Centered FFT over `dim` (int or tuple): ifftshift -> fftn -> fftshift, orthonormal."""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm=norm)
    return torch.fft.fftshift(x, dim=dim)


def ifft(x, dim, norm='ortho'):
    """Centered inverse FFT over `dim`: ifftshift -> ifftn -> fftshift, orthonormal."""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm=norm)
    return torch.fft.fftshift(x, dim=dim)


def forward_k(img, B0, esp):
    r"""
    Forward model with field map (undistorted --> distorted)
    :param img: input image, [nx, ny, nz]
    :param B0: B0 field map, [nx, ny, nz]
    """
    nx, ny, nz = img.shape

    t = esp * (torch.linspace(0, ny-1, ny).to(img.device)[None,None,None] - ny//2)     # [1, 1, 1, ny], last dimension for time each y line
    phasemap = torch.exp(1j * 2 * np.pi * B0[...,None] * t)            # [nx, ny, nz, ny]
    image_phase_compensation = img[...,None] * phasemap    # [nx, ny, nz, ny]
    k = fft(image_phase_compensation, dim=(1))           # [kx, ky, nz, ny]

    # Build the indices for advanced indexing
    kx_idx = torch.arange(nx)[:, None, None]   # shape [kx, 1, 1]
    nz_idx = torch.arange(nz)[None, None, :]   # shape [1, 1, nz]
    ky_idx = torch.arange(ny)[None, :, None]   # shape [1, ky, 1]

    # I is broadcast to match [1, ky, 1]
    I = torch.arange(ny).to(img.device)             # shape [ny]
    I_expanded = I[None, :, None]                   # shape [1, ky, 1]

    # Now gather the correct values
    k_correct = k[kx_idx, ky_idx, nz_idx, I_expanded]  # [kx, ky, nz]
    image_correct = ifft(k_correct, dim=(1))         # [nx, ny, nz]
    return image_correct #.real


def undersample_kspace(imgs_epi, csms, R, calib_size=(20, 20)):
        r"""
        Perform undersampling on input images.
        :param imgs_epi: input images, [n_angle, nx, ny]
        :param csms: coil sensitivity maps, [n_angle, nx, ny, ncoil]
        :param R: acceleration factor
        """
        n_angle = imgs_epi.shape[0]
        epi_ku_list = []
        calib_list = []
        epi_zf_coil_list = []
        epi_mask_list = []
        
        for i in range(n_angle):
            img_epi = imgs_epi[i]
            csm = csms[i]
            img_epi_coil = img_epi[...,None] * csm
            k_epi = fft(torch.from_numpy(img_epi_coil), dim=(0, 1)).numpy()
            # get the center calib_size of kspace as the calibration region
            kx, ky, ncoil = k_epi.shape
            calib = k_epi[kx//2-calib_size[0]//2:kx//2+calib_size[0]//2, ky//2-calib_size[1]//2:ky//2+calib_size[1]//2]
            calib_list.append(calib)
            mask_epi = np.zeros_like(k_epi, dtype=np.float32)
            mask_epi[:, i*(R//n_angle)::R] = 1
            k_epi = k_epi * mask_epi
            img_epi_zf_coil = ifft(torch.from_numpy(k_epi), dim=(0, 1)).numpy()
            epi_ku_list.append(k_epi)
            epi_zf_coil_list.append(img_epi_zf_coil)
            epi_mask_list.append(mask_epi)
        calib = np.stack(calib_list, axis=0)    # n_angle, kx, ky, ncoil
        epi_ku = np.stack(epi_ku_list, axis=0)    # n_angle, kx, ky, ncoil
        epi_zf_coil = np.stack(epi_zf_coil_list, axis=0)    # n_angle, nx, ny
        epi_mask = np.stack(epi_mask_list, axis=0)  # n_angle, nx, ny
        
        return epi_ku, calib, epi_zf_coil, epi_mask