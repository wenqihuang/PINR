import numpy as np
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity
import scipy.io as sio
import os
from utils.image_processing import center_zero_pad


def compute_image_metrics(img1, img2):
    """
    Compute NRMSE, PSNR, and SSIM between two 2D images.

    Parameters:
        img1 (numpy.ndarray): Original image (2D array).
        img2 (numpy.ndarray): Reconstructed or noisy image (2D array).
        
    Returns:
        dict: A dictionary with keys 'NRMSE', 'PSNR', and 'SSIM' corresponding
              to the computed metrics.
              
    Raises:
        ValueError: If the two input images do not have the same shape.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    nrmse_value = normalized_root_mse(img2, img1)
    psnr_value = peak_signal_noise_ratio(img2, img1, data_range=1)
    ssim_value = structural_similarity(img2, img1, data_range=1)
    
    return {'NRMSE': nrmse_value, 'PSNR': psnr_value, 'SSIM': ssim_value}


def calculate_scale_bias(img: np.ndarray, target: np.ndarray):
    # Flatten the arrays to 1D in case they are not already.
    x = img.ravel()
    y = target.ravel()
    
    # Compute the means of x and y.
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Compute the numerator and denominator for the scale.
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    # Check to avoid division by zero.
    if denominator == 0:
        raise ValueError("Variance of img is zero, cannot compute scale.")
    
    scale = numerator / denominator
    bias = y_mean - scale * x_mean
    
    return scale, bias


if __name__ == "__main__":
    subject_list = ['subject01', 'subject02', 'subject03']
    slice_start_end_list = [[50, 140], [50, 140], [50, 140]]
    slice_list = []
    for i in range(len(subject_list)):
        slice_i = [j for j in range(slice_start_end_list[i][0], slice_start_end_list[i][1])]
        slice_list.append(slice_i)


    topup_metrics_list = []
    rot2_metrics_list = []
    rot3_metrics_list = []
    for i in range(len(subject_list)):
        subject_id = subject_list[i]
        print(f'processing subject {subject_id}...')
    
        # load topup result (whole volume)
        topup_data = sio.loadmat(f'results/simulated_rot2/{subject_id}/topup_corrected_bs240.mat')
        img_topup = topup_data['topup_corrected']
        b0_topup = topup_data['fieldmap']

        for slc in slice_list[i]:
            print(f'processing subject {subject_id}, slice {slc}...')

            # load csm to get ROI mask
            mat = sio.loadmat(os.path.join('./simulated_data/multi_angle_rot2', subject_id, f'simulated_data_Slice{slc}.mat'))
            csm = mat['csm'][0]     # get the rot=0 csm, (nx, ny, nc)
            masks_roi = np.sum(np.abs(csm), -1) > 0 

            # load topup result (single slice)
            img_topup_slice = img_topup[:, :, slc, :].mean(-1)  # average over ap unwrap and pa unwrap

            # load original image
            # load b0 ground truth
            b0 = sio.loadmat(f'simulated_data/b0_fitted/b0_fitted_{subject_id}.mat')['b0_fitted']
            b0_orig = center_zero_pad([b0[:, :, slc]])[0] * 0.7   # same 0.7 scaling as in prepare_simulated_data_PD.py
            b0_topup_slice = -b0_topup[:, :, slc]

            # load rot2 results
            mat = sio.loadmat(f'results/simulated_rot2/{subject_id}/reconstructed_data_Slice{slc}.mat')
            img_orig = mat['img_orig']
            img_recon_rot2 = mat['recon']
            b0_recon_rot2 = mat['b0']

            # load rot3 results
            mat = sio.loadmat(f'results/simulated_rot3/{subject_id}/reconstructed_data_Slice{slc}.mat')
            img_recon_rot3 = mat['recon']
            b0_recon_rot3 = mat['b0']

            # apply roi mask to all images and b0
            img_topup_slice = img_topup_slice * masks_roi
            img_orig = img_orig * masks_roi
            img_recon_rot2 = img_recon_rot2 * masks_roi
            img_recon_rot3 = img_recon_rot3 * masks_roi
            b0_topup_slice = b0_topup_slice * masks_roi
            b0_recon_rot2 = b0_recon_rot2 * masks_roi
            b0_recon_rot3 = b0_recon_rot3 * masks_roi
            b0_orig = b0_orig * masks_roi

            b0_max = np.max(np.abs(b0_orig))

            # minimize bias and scale of topup, rot2 and rot3
            img_orig /= np.max(np.abs(img_orig))
            scale, bias = calculate_scale_bias(np.abs(img_topup_slice), np.abs(img_orig))
            img_topup_slice = img_topup_slice * scale + bias
            scale, bias = calculate_scale_bias(np.abs(img_recon_rot2), np.abs(img_orig))
            img_recon_rot2 = img_recon_rot2 * scale + bias
            scale, bias = calculate_scale_bias(np.abs(img_recon_rot3), np.abs(img_orig))
            img_recon_rot3 = img_recon_rot3 * scale + bias

            # compute metrics
            metrics_topup = compute_image_metrics(np.abs(img_topup_slice), np.abs(img_orig))
            metrics_rot2 = compute_image_metrics(np.abs(img_recon_rot2), np.abs(img_orig))
            metrics_rot3 = compute_image_metrics(np.abs(img_recon_rot3), np.abs(img_orig))

            # compute B0 nrmse
            metrics_topup['B0_diff'] = np.abs(b0_orig - b0_topup_slice).mean()
            metrics_rot2['B0_diff'] = np.abs(b0_orig - b0_recon_rot2).mean()
            metrics_rot3['B0_diff'] = np.abs(b0_orig - b0_recon_rot3).mean()


            topup_metrics_list.append(metrics_topup)
            rot2_metrics_list.append(metrics_rot2)
            rot3_metrics_list.append(metrics_rot3)


        
    # calculate mean and std of metrics
    topup_metrics = {
        'NRMSE_mean': np.mean([m['NRMSE'] for m in topup_metrics_list]),
        'NRMSE_std': np.std([m['NRMSE'] for m in topup_metrics_list]),
        'PSNR_mean': np.mean([m['PSNR'] for m in topup_metrics_list]),
        'PSNR_std': np.std([m['PSNR'] for m in topup_metrics_list]),
        'SSIM_mean': np.mean([m['SSIM'] for m in topup_metrics_list]),
        'SSIM_std': np.std([m['SSIM'] for m in topup_metrics_list]),
        'B0_diff_mean': np.mean([m['B0_diff'] for m in topup_metrics_list]),
        'B0_diff_std': np.std([m['B0_diff'] for m in topup_metrics_list])
    }
    rot2_metrics = {
        'NRMSE_mean': np.mean([m['NRMSE'] for m in rot2_metrics_list]),
        'NRMSE_std': np.std([m['NRMSE'] for m in rot2_metrics_list]),
        'PSNR_mean': np.mean([m['PSNR'] for m in rot2_metrics_list]),
        'PSNR_std': np.std([m['PSNR'] for m in rot2_metrics_list]),
        'SSIM_mean': np.mean([m['SSIM'] for m in rot2_metrics_list]),
        'SSIM_std': np.std([m['SSIM'] for m in rot2_metrics_list]),
        'B0_diff_mean': np.mean([m['B0_diff'] for m in rot2_metrics_list]),
        'B0_diff_std': np.std([m['B0_diff'] for m in rot2_metrics_list])
    }
    rot3_metrics = {
        'NRMSE_mean': np.mean([m['NRMSE'] for m in rot3_metrics_list]),
        'NRMSE_std': np.std([m['NRMSE'] for m in rot3_metrics_list]),
        'PSNR_mean': np.mean([m['PSNR'] for m in rot3_metrics_list]),
        'PSNR_std': np.std([m['PSNR'] for m in rot3_metrics_list]),
        'SSIM_mean': np.mean([m['SSIM'] for m in rot3_metrics_list]),
        'SSIM_std': np.std([m['SSIM'] for m in rot3_metrics_list]),
        'B0_diff_mean': np.mean([m['B0_diff'] for m in rot3_metrics_list]),
        'B0_diff_std': np.std([m['B0_diff'] for m in rot3_metrics_list])
    }
    
    print('-----------------------------------')
    print('Image Metrics:')
    print('NRMSE, PSNR, SSIM')
    print(f'TOPUP: {topup_metrics["NRMSE_mean"]:.4f} +/- {topup_metrics["NRMSE_std"]:.4f}, {topup_metrics["PSNR_mean"]:.4f} +/- {topup_metrics["PSNR_std"]:.4f}, {topup_metrics["SSIM_mean"]:.4f} +/- {topup_metrics["SSIM_std"]:.4f}')
    print(f'ROT2: {rot2_metrics["NRMSE_mean"]:.4f} +/- {rot2_metrics["NRMSE_std"]:.4f}, {rot2_metrics["PSNR_mean"]:.4f} +/- {rot2_metrics["PSNR_std"]:.4f}, {rot2_metrics["SSIM_mean"]:.4f} +/- {rot2_metrics["SSIM_std"]:.4f}')
    print(f'ROT3: {rot3_metrics["NRMSE_mean"]:.4f} +/- {rot3_metrics["NRMSE_std"]:.4f}, {rot3_metrics["PSNR_mean"]:.4f} +/- {rot3_metrics["PSNR_std"]:.4f}, {rot3_metrics["SSIM_mean"]:.4f} +/- {rot3_metrics["SSIM_std"]:.4f}')
    print('-----------------------------------')
    print('B0 Metrics:')
    print('B0_diff')
    print(f'TOPUP: {topup_metrics["B0_diff_mean"]:.4f} +/- {topup_metrics["B0_diff_std"]:.4f}')
    print(f'ROT2: {rot2_metrics["B0_diff_mean"]:.4f} +/- {rot2_metrics["B0_diff_std"]:.4f}')
    print(f'ROT3: {rot3_metrics["B0_diff_mean"]:.4f} +/- {rot3_metrics["B0_diff_std"]:.4f}')
    print('-----------------------------------')