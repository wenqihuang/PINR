import numpy as np
import torch
import os
import scipy.io as sio
from utils.b0_fitting import fit_espirit
from utils.image_processing import rotate_images, center_zero_pad
from utils.mri import forward_k
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Load configurations for the main function.")
    parser.add_argument("--data_file", type=str, default="./data/raw/subject01.mat", help="Path to the data file")
    parser.add_argument("--esp_fs", type=float, default=0.25e-3, help="Equivalent echo spacing time assuming fully sampled")
    parser.add_argument("--n_angle", type=int, default=3, help="Number of angles to rotate the data")
    parser.add_argument("--save_dir", type=str, default="simulated_data/multi_angle", help="Path to save the simulated data")
    parser.add_argument("--device", type=str, default="1", help="Device to use for computations")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    data_file = args.data_file
    subject_id = data_file.split('/')[-1].split('.')[0]
    n_angles = args.n_angle
    esp = args.esp_fs

    # load data
    data = sio.loadmat(data_file)
    print(data.keys())

    # get data
    b0_all = data['B0']
    R2s_all = data['R2s_rec']
    csm_all = data['SEs']
    img_all = data['im']

    # permute first two dimensions
    b0_all = np.transpose(b0_all, (1,0,2))      # (nx, ny, nz)
    R2s_all = np.transpose(R2s_all, (1,0,2))    # (nx, ny, nz)
    csm_all = np.transpose(csm_all, (1,0,2,3))  # (nx, ny, nz, nc)
    img_all = np.transpose(img_all, (1,0,2))    # (nx, ny, nz)

    # check if b0_fitted exists
    save_dir = f'simulated_data/b0_fitted/'
    save_file = os.path.join(save_dir, f'b0_fitted_{subject_id}.mat')
    if os.path.exists(save_file):
        print(f'fitted b0 already exists, load from {save_file}')
        b0_fitted_espirit = sio.loadmat(save_file)['b0_fitted']
        print(f'loaded fitted b0 from {save_file}')
    else:
        print(f'fitted b0 does not exist, fit b0 with espirit...')
        # fit b0 with espirit
        b0_fitted_espirit = fit_espirit(img_all, b0_all, calib_size=(80, 80), kernel_size=(15, 15), threshold=0, base_norm_factor=5)
        # save b0_fitted_espirit
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sio.savemat(save_file, {'b0_fitted': b0_fitted_espirit})
        print(f'saved fitted b0 to {save_file}')
    
    b0_all = b0_fitted_espirit * 0.7

    # get n_slice
    nx, ny, n_slice, nc = csm_all.shape

    # get max value of img_all
    max_v = np.max(np.abs(img_all))
    # normalize data
    img_all /= max_v

    # process each slice
    for slice_idx in range(n_slice):
        print(f'processing slice {slice_idx} of data{data_file}...')
        # get slice data
        b0 = b0_all[:, :, slice_idx]
        R2s = R2s_all[:, :, slice_idx]
        csm = csm_all[:, :, slice_idx, :]
        img = img_all[:, :, slice_idx]
        b0 = b0_all[:, :, slice_idx]

        # padding the data
        b0, R2s, csm, img = center_zero_pad([b0, R2s, csm, img])

        # get nx, ny, nz, nc again after padding
        nx, ny, nc = csm.shape

        # remove phase information of img
        img = np.abs(img)
        # simulate a sine wave phase

        grid = np.meshgrid(np.linspace(-np.pi, np.pi, nx), np.linspace(-np.pi, np.pi, ny), indexing='ij')
        phase = np.cos(np.sqrt((grid[0]-0.9)**2 + (grid[1]+1.1)**2)/1.5) * 0.5 * np.pi

        # create mask for phase
        mask = np.abs(img) > 0.25 * np.max(np.abs(img))
        # diaolate mask
        mask = np.logical_or(mask, np.roll(mask, 1, axis=0))
        mask = np.logical_or(mask, np.roll(mask, -1, axis=0))
        mask = np.logical_or(mask, np.roll(mask, 1, axis=1))
        mask = np.logical_or(mask, np.roll(mask, -1, axis=1))

        phase = phase * mask
        img = img * np.exp(1j * phase)
        R2s = R2s * mask

        img_orig = img.copy()
        
        # rotate b0, img, csm
        thetas = np.linspace(0, 360, n_angles, endpoint=False)
        print(f'rotating data with angles: {thetas}')
        rot_img_list = rotate_images([img, b0, csm], thetas=thetas)

        # generate distorted images
        imgs_rotated_no_distortion = np.stack([rot_img_list[i][0] for i in range(n_angles)], 0)   # (n_angle, nx, ny)
        B0s = np.stack([rot_img_list[i][1] for i in range(n_angles)], 0)    # (n_angle, nx, ny)
        csms = np.stack([rot_img_list[i][2] for i in range(n_angles)], 0)   # (n_angle, nx, ny, ncoil)

        imgs_epi = forward_k(torch.tensor(imgs_rotated_no_distortion).to(device).permute(1,2,0), torch.tensor(B0s).to(device).permute(1,2,0), esp=esp)
        imgs_epi = imgs_epi.permute(2,0,1).cpu().numpy()    # (n_angle, nx, ny)
            

        # stack up the images, csms and b0s
        thetas = np.array(thetas)

        # save data
        save_dir = f'{args.save_dir}_rot{n_angles}/{subject_id}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, f'simulated_data_Slice{slice_idx}.mat')
        data_dict = {
            'img': img_orig,        # distortion-free image, (nx, ny)
            'imgs_epi': imgs_epi,   # epi images with different angles, (n_angle, nx, ny, 1)
            'B0': B0s,              # B0 with different angles, (n_angle, nx, ny)
            'csm': csms,            # sensitivity maps with different angles, (n_angle, nx, ny, ncoil)
            'esp': esp,             # echo spacing time (assume fully sampled)
            'thetas': thetas        # angles of epi images
        }
        sio.savemat(save_file, data_dict)
        print(f'saved data to {save_file}')

    print('done!')

if __name__ == "__main__":
    main()