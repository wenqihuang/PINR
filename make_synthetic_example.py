"""Build a rotated-view EPI example in the format main.py expects, from the bundled brain phantom.

`demo/mni152_phantom.npz` holds a T2*-like axial slice of the ICBM 152 template and a B0 map
derived from a susceptibility model of the head (see demo/build_mni_phantom.py). This script adds
a smooth synthetic phase and coil sensitivities, rotates image, B0 and coils to the requested view
angles and distorts every view with the same EPI forward model as the real simulation
(`prepare_simulated_data_PD.py`). The output is one .mat file, so the training script can be run
right away:

    python make_synthetic_example.py --n_angle 3
    python main.py --config configs/config_simulated_rot3.yml --subject_id synthetic --slice_idx 0 --R 6

Usage:
    python make_synthetic_example.py [--n_angle 3] [--n_coil 8] [--b0_scale 1.0] [--esp_fs 0.25e-3]
                                     [--t2star_ms 0] [--phantom demo/mni152_phantom.npz]
                                     [--save_dir simulated_data/multi_angle]

`--t2star_ms` adds signal decay along the echo train (off by default, like the paper's simulation and
the forward model in main.py); a global scale above 1 or a longer echo spacing makes the distortion
larger, but beyond |dB0/dPE| * N_y * esp = 1 the displacement folds over and signal piles up.
"""
import os
from argparse import ArgumentParser

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import binary_erosion, gaussian_filter

from utils.image_processing import rotate_images
from utils.mri import fft, forward_k, ifft


def forward_k_t2star(img, B0, esp, t2star):
    """forward_k from utils.mri with exp(-t / T2*) decay along the echo train (t = 0 at the first line)."""
    nx, ny, nz = img.shape
    idx = torch.arange(ny, device=img.device, dtype=torch.float32)
    t = esp * (idx - ny // 2)[None, None, None]
    weight = torch.exp(1j * 2 * np.pi * B0[..., None] * t) * torch.exp(-esp * idx / t2star)[None, None, None]
    k = fft(img[..., None] * weight, dim=(1))
    kx_idx = torch.arange(nx)[:, None, None]
    ky_idx = torch.arange(ny)[None, :, None]
    nz_idx = torch.arange(nz)[None, None, :]
    return ifft(k[kx_idx, ky_idx, nz_idx, ky_idx], dim=(1))


def coil_sensitivities(support, n_coil, seed=0):
    """Gaussian loop profiles on a ring around the object with a smooth random phase."""
    rng = np.random.default_rng(seed)
    n = support.shape[0]
    xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing='ij')
    csm = np.zeros((*support.shape, n_coil), dtype=np.complex64)
    for c in range(n_coil):
        ang = 2 * np.pi * c / n_coil
        cx, cy = 1.3 * np.cos(ang), 1.3 * np.sin(ang)
        prof = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2)
        ph = gaussian_filter(rng.standard_normal(support.shape), 25)
        ph = ph / np.abs(ph).max() * np.pi
        csm[..., c] = prof * np.exp(1j * ph)
    csm /= np.sqrt((np.abs(csm) ** 2).sum(-1, keepdims=True)).max()
    return csm * support[..., None]


def main():
    ap = ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phantom', type=str, default='demo/mni152_phantom.npz')
    ap.add_argument('--n_angle', type=int, default=3)
    ap.add_argument('--n_coil', type=int, default=8)
    ap.add_argument('--b0_scale', type=float, default=1.0, help='multiplies the phantom B0 map')
    ap.add_argument('--esp_fs', type=float, default=0.25e-3, help='echo spacing (s) of the fully sampled EPI')
    ap.add_argument('--t2star_ms', type=float, default=0.0, help='T2* decay along the echo train (ms); 0 = none')
    ap.add_argument('--save_dir', type=str, default='simulated_data/multi_angle')
    ap.add_argument('--subject_id', type=str, default='synthetic')
    ap.add_argument('--device', type=str, default='0')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ph = np.load(args.phantom)
    mag, b0, head = ph['image'].astype(np.float32), ph['b0'].astype(np.float32) * args.b0_scale, ph['head']
    nx, ny = mag.shape

    # smooth synthetic phase, same construction as prepare_simulated_data_PD.py
    gx, gy = np.meshgrid(np.linspace(-np.pi, np.pi, nx), np.linspace(-np.pi, np.pi, ny), indexing='ij')
    phase = np.cos(np.sqrt((gx - 0.9) ** 2 + (gy + 1.1) ** 2) / 1.5) * 0.5 * np.pi
    img = (mag * np.exp(1j * phase * head)).astype(np.complex64)
    csm = coil_sensitivities(head, args.n_coil)

    thetas = np.linspace(0, 360, args.n_angle, endpoint=False)
    rot = rotate_images([img, b0, csm], thetas=thetas)
    imgs_rot = np.stack([r[0] for r in rot], 0)   # (n_angle, nx, ny)
    b0s = np.stack([r[1] for r in rot], 0)        # (n_angle, nx, ny)
    csms = np.stack([r[2] for r in rot], 0)       # (n_angle, nx, ny, n_coil)

    img_t, b0_t = torch.tensor(imgs_rot, device=device).permute(1, 2, 0), torch.tensor(b0s, device=device).permute(1, 2, 0)
    if args.t2star_ms > 0:
        imgs_epi = forward_k_t2star(img_t, b0_t, args.esp_fs, args.t2star_ms * 1e-3)
    else:
        imgs_epi = forward_k(img_t, b0_t, esp=args.esp_fs)
    imgs_epi = imgs_epi.permute(2, 0, 1).cpu().numpy()

    save_dir = f'{args.save_dir}_rot{args.n_angle}/{args.subject_id}'
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'simulated_data_Slice0.mat')
    sio.savemat(save_file, {
        'img': img,           # distortion-free image, (nx, ny)
        'imgs_epi': imgs_epi, # distorted EPI images per view, (n_angle, nx, ny)
        'B0': b0s,            # B0 per view, (n_angle, nx, ny), Hz
        'csm': csms,          # coil sensitivities per view, (n_angle, nx, ny, n_coil)
        'esp': args.esp_fs,   # echo spacing of the fully sampled EPI (s)
        'thetas': thetas,     # view angles (degrees)
    })
    tau = ny * args.esp_fs
    shift = np.abs(b0[head]).max() * tau
    gx, gy = np.gradient(b0, axis=0), np.gradient(b0, axis=1)
    inner = binary_erosion(head, iterations=2)
    fold = max(np.abs(gy * np.cos(th) - gx * np.sin(th))[inner].max() for th in np.deg2rad(thetas)) * tau
    print(f'saved {save_file}: {nx}x{ny}, {args.n_angle} views, {args.n_coil} coils, '
          f'|B0| <= {np.abs(b0[head]).max():.0f} Hz, readout {ny * args.esp_fs * 1e3:.0f} ms, '
          f'max distortion {shift:.1f} px, max |dB0/dPE| * tau {fold:.2f} (fold-over above 1)')


if __name__ == '__main__':
    main()
