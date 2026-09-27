"""Build demo/mni152_phantom.npz: an axial brain slice with a susceptibility-derived B0 map.

Anatomy comes from the ICBM 152 2009a nonlinear symmetric T1 template and its GM/WM probability
maps (bundled with nilearn, 1 mm). The slice is rendered with a T2*-like contrast (CSF bright,
GM > WM, scalp fat bright, bone and air dark) to resemble a b=0 EPI image.

The template is brain-extracted, so a head is built around it: a 10 mm skull/scalp shell plus the
face and neck below (every column filled down to the bottom of the volume). Air cavities
(frontal and sphenoid sinuses, nasal cavity, mastoid cells) are placed relative to landmarks of
the brain mask, so they sit against the brain surface as in a real head.

B0 model (3 T, B0 along z): susceptibility with air as the zero reference, tissue at -9.4 ppm
and the cavities at 0 ppm, plus a small intensity-driven texture inside the brain. The field
is the dipole convolution dB/B0 = F^-1[(1/3 - kz^2/|k|^2) F[chi]], converted with 127.7 Hz/ppm, and
an in-plane polynomial fitted over the brain is removed (first order by default, i.e. linear shimming).
The field is smoothed (2 mm) and scaled (0.8) so that the EPI displacement of the default example stays
invertible: |dB0/dPE| * N_y * esp < 1 everywhere, which is what keeps the distorted views free of pile-up.

Usage:
    python demo/build_mni_phantom.py [--z -28] [--shim_order 1] [--b0_smooth_mm 2] [--b0_scale 0.8] [--size 224]
"""
import argparse
import os

import nibabel as nib
import numpy as np
from nilearn import datasets
from scipy import ndimage as ndi

HZ_PER_PPM_3T = 42.577e6 * 3.0 * 1e-6   # 127.7 Hz per ppm
CHI_TISSUE = -9.4                       # ppm relative to air

SHELL_MM = 10        # skull + scalp thickness around the brain mask
SKULL_MM = 7.5       # inner part of the shell rendered dark (bone)


def landmarks(brain, affine):
    """Helpers giving brain-surface positions in MNI mm: lowest z at (x, y), most anterior y at (x, z)."""
    inv = np.linalg.inv(affine)
    kk = np.arange(brain.shape[2])[None, None, :]
    zmin = np.where(brain.any(2), np.where(brain, kk, brain.shape[2]).min(2), -1)      # (i, j) -> k
    jj = np.arange(brain.shape[1])[None, :, None]
    ymax = np.where(brain.any(1), np.where(brain, jj, -1).max(1), -1)                 # (i, k) -> j

    def z_inf(x, y, r=6):
        i, j = np.round(inv @ [x, y, 0, 1])[:2].astype(int)
        block = zmin[i - r:i + r + 1, j - r:j + r + 1]
        return nib.affines.apply_affine(affine, [i, j, block[block >= 0].min()])[2]

    def y_ant(x, z, r=6):
        i, k = np.round(inv @ [x, 0, z, 1])[[0, 2]].astype(int)
        block = ymax[i - r:i + r + 1, k - r:k + r + 1]
        return nib.affines.apply_affine(affine, [i, block.max(), k])[1]
    return z_inf, y_ant


def cavity_list(z_inf, y_ant):
    """Air cavities as (centre xyz in mm, semi-axes in mm), placed against the brain surface."""
    return [
        ((0.0, y_ant(0.0, 8.0) + 8.0, 8.0), (18.0, 7.0, 14.0)),               # frontal sinus
        ((0.0, 12.0, z_inf(0.0, 12.0) - 18.0), (14.0, 13.0, 12.0)),           # sphenoid sinus
        ((0.0, 48.0, z_inf(0.0, 48.0) - 22.0), (14.0, 20.0, 15.0)),           # nasal cavity / ethmoid
        ((58.0, -28.0, z_inf(58.0, -28.0) - 12.0), (12.0, 14.0, 12.0)),       # mastoid cells, right
        ((-58.0, -28.0, z_inf(-58.0, -28.0) - 12.0), (12.0, 14.0, 12.0)),     # mastoid cells, left
    ]


def load_volumes():
    t1_img = datasets.load_mni152_template(resolution=1)
    data_dir = os.path.join(os.path.dirname(datasets.__file__), 'data')
    gm = nib.load(os.path.join(data_dir, 'mni_icbm152_gm_tal_nlin_sym_09a_converted.nii.gz')).get_fdata()
    wm = nib.load(os.path.join(data_dir, 'mni_icbm152_wm_tal_nlin_sym_09a_converted.nii.gz')).get_fdata()
    brain = datasets.load_mni152_brain_mask(resolution=1).get_fdata() > 0
    t1 = t1_img.get_fdata().astype(np.float32)
    gm, wm = gm / gm.max(), wm / wm.max()   # probability maps are stored on a 0..255 scale
    return t1, gm.astype(np.float32), wm.astype(np.float32), brain, t1_img.affine


def ellipsoid_mask(shape, affine, centre, axes):
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    xyz = nib.affines.apply_affine(affine, np.stack([ii, jj, kk], -1).reshape(-1, 3)).reshape(*shape, 3)
    d = ((xyz - np.asarray(centre)) / np.asarray(axes)) ** 2
    return d.sum(-1) <= 1.0


def dipole_field_ppm(chi, pad=256):
    """Field shift (ppm) of a susceptibility volume (ppm), B0 along the third axis."""
    shape = chi.shape
    padded = np.zeros((pad, pad, pad), dtype=np.float32)
    off = [(pad - n) // 2 for n in shape]
    padded[off[0]:off[0] + shape[0], off[1]:off[1] + shape[1], off[2]:off[2] + shape[2]] = chi
    k = [np.fft.fftfreq(pad) for _ in range(3)]
    kx, ky, kz = np.meshgrid(*k, indexing='ij')
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        D = 1.0 / 3.0 - kz ** 2 / k2
    D[0, 0, 0] = 0.0
    field = np.real(np.fft.ifftn(D * np.fft.fftn(padded))).astype(np.float32)
    return field[off[0]:off[0] + shape[0], off[1]:off[1] + shape[1], off[2]:off[2] + shape[2]]


def remove_poly(f, mask, order):
    """Subtract the polynomial of the given order fitted to f over mask (in-plane shimming)."""
    x, y = np.meshgrid(np.linspace(-1, 1, f.shape[0]), np.linspace(-1, 1, f.shape[1]), indexing='ij')
    terms = [np.ones_like(x), x, y, x * x, y * y, x * y][: (1, 3, 6)[order]]
    A = np.stack(terms, -1)
    coef, *_ = np.linalg.lstsq(A[mask], f[mask], rcond=None)
    return f - A @ coef


def crop_pad(s, size):
    """Centre-crop or zero-pad a 2D array to size x size."""
    out = np.zeros((size, size), dtype=s.dtype)
    ox, oy = (size - s.shape[0]) // 2, (size - s.shape[1]) // 2
    lx, ly = min(s.shape[0], size), min(s.shape[1], size)
    out[max(ox, 0):max(ox, 0) + lx, max(oy, 0):max(oy, 0) + ly] = s[max(-ox, 0):max(-ox, 0) + lx, max(-oy, 0):max(-oy, 0) + ly]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', type=float, default=-28.0, help='axial slice position in MNI mm')
    ap.add_argument('--shim_order', type=int, default=1, choices=[0, 1, 2],
                    help='polynomial order removed from the field over the brain (shimming)')
    ap.add_argument('--size', type=int, default=224, help='output matrix (square, 1 mm voxels)')
    ap.add_argument('--b0_smooth_mm', type=float, default=2.0,
                    help='Gaussian smoothing of the field map (mm); keeps the displacement field continuous')
    ap.add_argument('--b0_scale', type=float, default=0.8,
                    help='global scale of the field; 0.8 keeps the EPI displacement invertible for the default example')
    ap.add_argument('--out', type=str, default='demo/mni152_phantom.npz')
    args = ap.parse_args()

    t1, gm, wm, brain, affine = load_volumes()
    shape = t1.shape

    # head: brain + shell (skull, scalp) + face/neck (columns filled down to the volume bottom)
    dist = ndi.distance_transform_edt(~brain)                       # mm from the brain surface
    shell = (dist > 0) & (dist <= SHELL_MM)
    head = brain | shell
    head |= np.cumsum(head[:, :, ::-1], axis=2)[:, :, ::-1] > 0     # fill downwards (z axis points up)

    z_inf, y_ant = landmarks(brain, affine)
    cav = np.zeros(shape, dtype=bool)
    for centre, axes in cavity_list(z_inf, y_ant):
        cav |= ellipsoid_mask(shape, affine, centre, axes)
    cav &= head & (dist > 2.0)
    tissue = head & ~cav

    # susceptibility (ppm, air = 0) with a small intensity-driven texture inside the brain
    chi = CHI_TISSUE * tissue.astype(np.float32)
    chi += 0.15 * (t1 - t1[brain].mean()) * brain
    field_hz = dipole_field_ppm(chi) * HZ_PER_PPM_3T

    # axial slice at z, cropped / padded to size x size (axis 0: L-R, axis 1: P-A)
    k = int(round(np.linalg.solve(affine, [0, 0, args.z, 1])[2]))
    t1s, gms, wms, dists = (crop_pad(v[:, :, k], args.size) for v in (t1, gm, wm, dist))
    brains, heads, cavs, tissues = (crop_pad(v[:, :, k], args.size) for v in (brain, head, cav, tissue))
    support = brains | (heads & (dists <= SHELL_MM))               # brain + skull/scalp shell carry signal
    b0 = ndi.gaussian_filter(crop_pad(field_hz[:, :, k], args.size), args.b0_smooth_mm)
    b0 = remove_poly(b0, brains, args.shim_order) * args.b0_scale

    # T2*-like magnitude: CSF bright, GM > WM, bone dark, scalp fat bright, soft tissue below the
    # skull base at a muscle/fat level, air dark
    rng = np.random.default_rng(0)
    texture = 1.0 + 0.12 * ndi.gaussian_filter(rng.standard_normal((args.size, args.size)), 3)
    csf = np.clip(brains - gms - wms, 0, 1)
    img = (0.95 * csf + 0.60 * gms + 0.42 * wms) * (0.9 + 0.1 * (1.0 - t1s))
    skull = (dists > 1.5) & (dists <= SKULL_MM)
    scalp = (dists > SKULL_MM) & (dists <= SHELL_MM)
    face = heads & (dists > SHELL_MM)
    img += 0.06 * skull + 0.70 * scalp * texture + 0.35 * face * texture
    img = ndi.gaussian_filter(img, 0.6) * (tissues & support)
    img /= img.max()

    # fold-over check: the EPI displacement y -> y + tau * B0(y) stays invertible iff |dB0/dPE| < 1/tau;
    # evaluated along the phase-encoding direction of each of the three default views
    tau = args.size * 0.25e-3                                      # readout of the default example (s)
    gx, gy = np.gradient(b0, axis=0), np.gradient(b0, axis=1)
    inner = ndi.binary_erosion(support, iterations=2)
    worst = max(np.abs(gy * np.cos(th) - gx * np.sin(th))[inner].max() for th in np.deg2rad([0.0, 120.0, 240.0]))
    print(f'fold-over check: max |dB0/dPE| * tau = {worst * tau:.2f} over the three views '
          f'(limit 1 for {args.size} lines at 0.25 ms); max |B0| in support {np.abs(b0[support]).max():.0f} Hz '
          f'= {np.abs(b0[support]).max() * tau:.1f} px')
    # no step at the support edge: the field is kept as is inside the support and tapered to zero
    # smoothly outside it (window = 1 within 8 px of the support, Gaussian roll-off beyond)
    window = ndi.gaussian_filter(ndi.binary_dilation(support, iterations=8).astype(np.float32), 2.5)
    b0 = b0 * window
    np.savez_compressed(args.out, image=img.astype(np.float32), b0=b0.astype(np.float32),
                        head=support, brain=brains, z_mm=args.z, field_strength_T=3.0,
                        source='ICBM 152 2009a nonlinear symmetric template (nilearn)')
    print(f'saved {args.out}: {img.shape}, B0 over brain [{b0[brains].min():.0f}, {b0[brains].max():.0f}] Hz, '
          f'std {b0[brains].std():.0f} Hz, 99th pct |B0| {np.percentile(np.abs(b0[brains]), 99):.0f} Hz')

    # diagnostic figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(t1s.T, cmap='gray', origin='lower'); ax[0].set_title(f'T1 template, z = {args.z:g} mm')
    ax[1].imshow(img.T, cmap='gray', origin='lower'); ax[1].set_title('T2*-like phantom')
    v = np.percentile(np.abs(b0[brains]), 99.5)
    im = ax[2].imshow(b0.T, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower'); ax[2].set_title('B0 (Hz)')
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    xs = int(round(np.linalg.solve(affine, [0, 0, 0, 1])[0]))
    sag = np.where(cav[xs], 2, np.where(tissue[xs], 1, 0)).T
    ax[3].imshow(sag, cmap='viridis', origin='lower'); ax[3].axhline(k, color='r', lw=1)
    ax[3].set_title('sagittal x = 0: tissue / cavities, slice')
    for a in ax:
        a.set_axis_off()
    plt.tight_layout(); plt.savefig(os.path.splitext(args.out)[0] + '_preview.png', dpi=110); plt.close()


if __name__ == '__main__':
    main()
