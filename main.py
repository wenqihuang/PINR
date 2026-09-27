import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from model import create_model
import scipy.io as sio
from utils.mri import fft, ifft
from utils.image_processing import rotate_grid
from utils.mri import forward_k, undersample_kspace
from utils.parser import load_config, namespace_to_dict
import argparse

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

torch.set_num_threads(10)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load configurations for the main function.")
    parser.add_argument("--config", type=str, default="configs/config_simulated_rot2.yml", help="Path to the config YAML file")
    parser.add_argument("--device", type=str, default="0", help="Device to use for computations")
    parser.add_argument("--subject_id", type=str, default="-1", help="Subject ID")
    parser.add_argument("--slice_idx", type=int, default=-1, help="Slice index to reconstruct, -1 for default slice in config file")
    parser.add_argument("--R", type=int, default=-1, help="Per-view acceleration factor, -1 for the value in the config file")
    return parser.parse_args()


def main():
    """Main function that loads configurations and executes tasks."""
    args = parse_arguments()
    config_dict, config = load_config(args.config)

    # Overwrite configurations with command-line arguments
    config.device = args.device
    config.data_config.slice_idx = args.slice_idx if args.slice_idx >= 0 else config.data_config.slice_idx
    config.data_config.subject_id = args.subject_id if args.subject_id != "-1" else config.data_config.subject_id
    config.data_config.R = args.R if args.R >= 0 else config.data_config.R
    config_dict["data_config"]["slice_idx"] = config.data_config.slice_idx
    config_dict["data_config"]["subject_id"] = config.data_config.subject_id
    config_dict["data_config"]["R"] = config.data_config.R
    config_dict["device"] = config.device

    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    device = torch.device('cuda')
    

    subject_id = config.data_config.subject_id

    if config.data_config.data_type == "simulated":
        # Load simulated data
        file_path = os.path.join(config.data_config.data_dir, subject_id, f'simulated_data_Slice{config.data_config.slice_idx}.mat')
        data = sio.loadmat(file_path)
        img_orig = data['img']          # distortion-free image, (nx, ny)
        imgs_epi = data['imgs_epi']     # list of epi images with different angles, (n_angle, nx, ny, 1)
        B0s = data['B0']                # B0 with different angles, (n_angle, nx, ny)
        esp = data['esp'][0, 0]         # echo spacing time (assume fully sampled)
        csms = data['csm']              # sensitivity maps with different angles, (n_angle, nx, ny, ncoil)
        thetas = data['thetas'][0]      # angles of epi images

        # print shapes

        # Perform undersampling
        _, _, imgs_epi_zf_coil, masks_epi = undersample_kspace(imgs_epi, csms, config.data_config.R)

        # Normalization
        max_mag = np.max(np.abs(imgs_epi_zf_coil))
        img_orig /= max_mag
        imgs_epi_zf_coil /= max_mag
        imgs_epi /= max_mag

    elif config.data_config.data_type == "blipupdown":
        raise NotImplementedError("Data type 'blipupdown' is not implemented yet.")

    n_angle = len(thetas)
    masks_roi = np.sum(np.abs(csms), 3) > 0     # (n_angle, nx, ny)


    # Generate grid
    nx, ny = imgs_epi[0].shape
    nz = 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.zeros(1)
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')   # (nx, ny) * 2 

    # rotate the grid according to thetas
    grids = np.zeros((n_angle, nx, ny, 3))      # rotated grids
    for i in range(n_angle):
        theta = thetas[i]
        grid_x_rot, grid_y_rot = rotate_grid(grid_x, grid_y, theta, 0.5, 0.5)
        grid_rot = np.stack([grid_x_rot, grid_y_rot], -1)  # (nx, ny, 2)
        grids[i,:,:,0:2] = grid_rot


    # create input tensor
    grids = torch.tensor(grids, dtype=torch.float32, device=device)  # (n_angle, nx, ny, 3)
    imgs_epi = torch.tensor(imgs_epi, dtype=torch.complex64, device=device)  # (n_angle, nx, ny)
    imgs_epi_zf_coil = torch.tensor(imgs_epi_zf_coil, dtype=torch.complex64, device=device)  # (n_angle, nx, ny, nc)
    csms = torch.from_numpy(csms).to(device)
    masks_epi = torch.from_numpy(masks_epi).to(device)  # (n_angle, nx, ny, nc)
    masks_roi = torch.tensor(masks_roi, dtype=torch.float32, device=device)  # (n_angle, nx, ny)

    # mask epi images and img_orig
    imgs_epi = imgs_epi * masks_roi
    img_orig = img_orig * masks_roi[0].cpu().numpy()


    # create models
    model_b0 = create_model(
        namespace_to_dict(config.model_config.b0.encoding), 
        config.model_config.b0.in_dim,
        config.model_config.b0.hidden_dims,
        config.model_config.b0.out_dim
    ).to(device)

    model_img = create_model(
        namespace_to_dict(config.model_config.img.encoding), 
        config.model_config.img.in_dim,
        config.model_config.img.hidden_dims,
        config.model_config.img.out_dim
    ).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model_b0.parameters(), "lr": float(config.train_config.lr)},
            {"params": model_img.parameters(), "lr": float(config.train_config.lr)},
        ]
    )

    # training
    smooth_factor = float(config.train_config.b0_smooth_factor) 
    for iter in range(config.train_config.n_iter):
        # Get B0 and distortion-free images for each angle
        grid_vec = grids.reshape(-1, 3)  # (n_angle*nx*ny, 3)
        b0 = model_b0(grid_vec) * config.data_config.b0_factor  # convert network output to reasonable frequency range
        b0 = b0.reshape(n_angle, nx, ny)  # (n_angle, nx, ny)

        b0 = b0 * masks_roi

        imgs_no_distortion = torch.view_as_complex(model_img(grid_vec)).reshape(n_angle, nx, ny) * masks_roi
        
        # Generate distorted images
        imgs_distortion = forward_k(imgs_no_distortion.permute(1,2,0), b0.permute(1,2,0), esp).permute(2,0,1)

        # Get predicted zf coil images for loss calculation further
        imgs_distortion_zf_coil = ifft(fft(imgs_distortion[..., None] * csms, dim=(1, 2)) * masks_epi, dim=(1, 2))

        # Regularizations
        if (iter + 1) % 1000 == 0:
            smooth_factor = smooth_factor * 0.1
        elif config.train_config.n_iter - iter < 1000:
            smooth_factor = 0
        reg_b0_smooth = smooth_factor * \
            (torch.mean((b0[:,1:] - b0[:,:-1])**2) + torch.mean((b0[:,:,1:] - b0[:,:,:-1])**2))     # TV regularization for B0 field
            
        reg_out_roi_b0 = float(config.train_config.out_roi_supression_factor) * \
            torch.mean((b0 * (1 - masks_roi)).abs()**2)  # suppress B0 outside ROI
        reg_out_roi_img = float(config.train_config.out_roi_supression_factor) * \
            torch.mean((imgs_no_distortion * (1 - masks_roi)).abs()**2)  # suppress image outside ROI
        
        reg_b0 = reg_b0_smooth + reg_out_roi_b0
        reg_img = reg_out_roi_img

        loss_dc = criterion(torch.view_as_real(imgs_distortion_zf_coil), torch.view_as_real(imgs_epi_zf_coil))

        loss = loss_dc + reg_b0 + reg_img
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


                

    
    # save the results
    save_dir = f'{config.log_config.save_dir}_rot{n_angle}/{subject_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f'reconstructed_data_Slice{config.data_config.slice_idx}.mat')
    result_dict = {
        'img_orig': img_orig,     # distortion-free image, (nx, ny)
        'b0': b0[0].detach().cpu().numpy(),  # B0 field for the angle=0 case, (nx, ny)
        'recon': imgs_no_distortion[0].detach().cpu().numpy(),  # distortion-free image for the angle=0 case, (nx, ny)
    }
    sio.savemat(save_file, result_dict)

    # save a figure
    imgs_distortion = imgs_distortion.detach().cpu().numpy()          # (n_angle, nx, ny)
    b0_pred = b0[0].detach().cpu().numpy()        # (nx, ny)
    diff_b0 = np.abs(B0s[0] - b0_pred)
    diff_epi = np.abs(imgs_distortion - imgs_epi.detach().cpu().numpy())       # (n_angle, nx, ny)
    img_pred = imgs_no_distortion.detach().cpu().numpy()
    diff_img = np.abs(img_orig - img_pred[0])
    d_vmax = np.abs(B0s[0]).max()
    v_min = 0
    v_max = np.abs(img_orig).max()
    fig, axs = plt.subplots(n_angle+1, 5, figsize=(15, 12))
    for i in range(n_angle):
        axs[i, 0].imshow(np.abs(imgs_epi[i].detach().cpu().numpy()), cmap='gray', vmin=v_min, vmax=v_max)
        axs[i, 0].set_title(f'epi_{i}')
        axs[i, 1].imshow(np.abs(imgs_distortion[i]), cmap='gray', vmin=v_min, vmax=v_max)
        axs[i, 1].set_title(f'pred_epi_{i}')
        axs[i, 2].imshow(diff_epi[i], cmap='gray', vmin=0, vmax=0.1*v_max)
        axs[i, 2].set_title(f'diff_epi_{i} (10x)')
        axs[i, 3].imshow(B0s[i], cmap='viridis', vmin=-d_vmax, vmax=d_vmax)
        axs[i, 3].set_title(f'groundtruth b0_{i}')
        axs[i, 4].imshow(b0_pred, cmap='viridis', vmin=-d_vmax, vmax=d_vmax)
        axs[i, 4].set_title(f'pred b0_{i}')
    
    axs[-1, 0].imshow(np.abs(img_orig), cmap='gray', vmin=v_min, vmax=v_max)
    axs[-1, 0].set_title('img_groundtruth')
    axs[-1, 1].imshow(np.abs(img_pred[0]), cmap='gray', vmin=v_min, vmax=v_max)
    axs[-1, 1].set_title('img_pred')
    axs[-1, 2].imshow(diff_img, cmap='gray', vmin=0, vmax=0.1*v_max)
    axs[-1, 2].set_title('diff_img (10x)')
    axs[-1, 3].imshow(np.abs(b0_pred - B0s[0]), cmap='viridis', vmin=0, vmax=d_vmax*0.2)
    axs[-1, 3].set_title('diff_b0 (10x)')
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_Slice{config.data_config.slice_idx}.png'))
    plt.close()
    print(f'saved data to {save_file}')
    print('done!')
    
if __name__ == "__main__":
    main()