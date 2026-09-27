import numpy as np
import nibabel as nib
import torch
import os
import scipy.io as sio
from utils.mri import ifft, undersample_kspace
from utils.parser import load_config
from pygrappa import grappa
import argparse
from glob import glob


# add fsl path
FSLDIR = os.environ.get('FSLDIR', '/usr/local/fsl')
os.environ['FSLDIR'] = FSLDIR
os.environ['PATH'] += os.pathsep + os.path.join(FSLDIR, 'bin')
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load configurations for the main function.")
    parser.add_argument("--config", type=str, default="configs/config_simulated_rot2.yml", help="Path to the config YAML file")
    parser.add_argument("--subject_id", type=str, default="-1", help="Subject ID")
    parser.add_argument("--device", type=str, default="0", help="Device to use for computations")
    parser.add_argument("--R", type=int, default=-1, help="Per-view acceleration factor, -1 for the value in the config file")
    return parser.parse_args()


def main():
    """Main function that loads configurations and executes tasks."""
    args = parse_arguments()
    config_dict, config = load_config(args.config)

    # Overwrite configurations with command-line arguments
    config.device = args.device
    config.data_config.subject_id = args.subject_id if args.subject_id != "-1" else config.data_config.subject_id
    config.data_config.R = args.R if args.R >= 0 else config.data_config.R
    config_dict["data_config"]["subject_id"] = config.data_config.subject_id
    config_dict["data_config"]["R"] = config.data_config.R
    config_dict["device"] = config.device

    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    device = torch.device('cuda')
    
    print("Loaded Configuration:", config_dict)

    subject_id = config.data_config.subject_id

    if config.data_config.data_type == "simulated":
        # get the total number of slices from the file folder
        n_slices = len(glob(os.path.join(config.data_config.data_dir, subject_id, "simulated_data_Slice*.mat")))
        print(f"Total number of slices: {n_slices}")

        ku_list = []
        imgs_epi_zf_coil_list = []
        calib_list = []
        B0s_list = []
        csms_list = []

        for slice_idx in range(n_slices):
            # Load simulated data
            file_path = os.path.join(config.data_config.data_dir, subject_id, f'simulated_data_Slice{slice_idx}.mat')
            print(f"Loading data from {file_path}...")
            data = sio.loadmat(file_path)
            imgs_epi = data['imgs_epi']     # list of epi images with different angles, (n_angle, nx, ny)
            B0s = data['B0']                # B0 with different angles, (n_angle, nx, ny)
            esp = data['esp'][0, 0]         # echo spacing time (assume fully sampled)
            csms = data['csm']              # sensitivity maps with different angles, (n_angle, nx, ny, ncoil)
            thetas = data['thetas'][0]      # angles of epi images

            assert imgs_epi.shape[0] == 2, "Only support 2 EPI images for TOPUP."

            B0s_list.append(B0s)
            csms_list.append(csms)
        

            # Perform undersampling
            kus_epi, calib, imgs_epi_zf_coil, masks_epi = undersample_kspace(imgs_epi, csms, config.data_config.R, calib_size=(10, 10))

            # Normalization
            max_mag = np.max(np.abs(imgs_epi_zf_coil))
            if max_mag == 0:
                max_mag = 1
            calib /= max_mag
            kus_epi /= max_mag
            imgs_epi_zf_coil /= max_mag
            imgs_epi /= max_mag

            ku_list.append(kus_epi)
            calib_list.append(calib)

        ku_list = np.stack(ku_list, axis=-2)    # (n_angle, nx, ny, nslice, coil)
        calib_list = np.stack(calib_list, axis=-2)    # (n_angle, nxc, nyc, nslice, coil)
        csms_list = np.stack(csms_list, axis=-2)    # (n_angle, nx, ny, nslice, ncoil)


    # plot kus_epi

    # grappa reconstruction for blip-up and blip-down images
    grappa_img_file = f"results/grappa_appa_{subject_id}_R{config.data_config.R}.mat"

    if os.path.exists(grappa_img_file):
        print(f"GRAPPA reconstruction already exists for {subject_id}.")
        imgs_grappa_all = sio.loadmat(grappa_img_file)["imgs_grappa"]
        print(f"Loaded GRAPPA reconstruction from {grappa_img_file}.")
    else:
        print("Performing GRAPPA reconstruction...")
        imgs_grappa_all = []
        for slc in range(n_slices):
            print(f"GRAPPA reconstruction for slice {slc}...")
            imgs_grappa = []
            for i in range(2):
                calib_i = calib_list[i,:,:,slc,:]      # (kx_c, ky_c, ncoil)
                ku_i = ku_list[i,:,:,slc,:]       # (kx, ky, ncoil)
                csm_i = csms_list[i,:,:,slc,:]    # (nx, ny, ncoil)
                k_grappa = grappa(ku_i, calib_i, kernel_size=(5, 5))
                img_grappa = np.sum(ifft(torch.from_numpy(k_grappa), dim=(0, 1)).numpy() * csm_i.conj(), axis=-1)
                
                if i == 1:
                    img_grappa = img_grappa[::-1, ::-1]     # rotate 180 degrees
                imgs_grappa.append(img_grappa)

            imgs_grappa = np.abs(np.stack(imgs_grappa, axis=-1))    # (nx, ny, n_angle)
            
            imgs_grappa_all.append(imgs_grappa)
        imgs_grappa_all = np.stack(imgs_grappa_all, axis=-2)    # (nx, ny, nslice, n_angle)

        # save grappa images
        sio.savemat(grappa_img_file, {"imgs_grappa": imgs_grappa_all})

        print("GRAPPA reconstruction completed.")
    

    # multiprocessing to run TOPUP for batch of slices
    print("Running TOPUP...")
    batch_size = 240
    n_batches = (n_slices + batch_size - 1) // batch_size
    fieldmap_list = []
    topup_corrected_list = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_slices)
        print(f"Running TOPUP for batch {batch_idx + 1}/{n_batches}...")
        
        # convert grappa images to nibabel
        img_grappa_nii = nib.Nifti1Image(imgs_grappa_all[:,:,start_idx:end_idx,:], np.eye(4))

        # create a temporary directory for TOPUP
        temp_dir = f"tmp_{subject_id}_2"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # save grappa images to nifti in the temporary directory
        img_grappa_nii_path = os.path.join(temp_dir, f"grappa_images.nii.gz")
        nib.save(img_grappa_nii, img_grappa_nii_path)

        # save acqparams to a text file
        acqparams = [[0, 1, 0, esp*(imgs_grappa_all.shape[1]-1)],  # AP phase encoding (y-axis)
                    [0, -1, 0, esp*(imgs_grappa_all.shape[1]-1)]]     # PA phase encoding (y-axis)
        acqparams_path = os.path.join(temp_dir, "acqparams.txt")
        np.savetxt(acqparams_path, acqparams)

        # run TOPUP
        topup_cmd = f"topup --imain={img_grappa_nii_path} --datain={acqparams_path} --config=./configs/b02b0_topup.cnf --out={temp_dir}/topup_results --fout={temp_dir}/fieldmap --iout={temp_dir}/topup_corrected"
        os.system(topup_cmd)

        # load corrected images and b0 field
        topup_corrected = nib.load(f"{temp_dir}/topup_corrected.nii").get_fdata()
        fieldmap = nib.load(f"{temp_dir}/fieldmap.nii").get_fdata()

        topup_corrected_list.append(topup_corrected)
        fieldmap_list.append(fieldmap)
        print("TOPUP completed for batch", batch_idx + 1)
    
    # concatenate results
    topup_corrected = np.concatenate(topup_corrected_list, axis=-2)
    fieldmap = np.concatenate(fieldmap_list, axis=-2)

    # save corrected images and b0 field
    result_file = f"results/simulated_rot2/{subject_id}/topup_corrected_bs{batch_size}.mat"
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    result_dict = {
        "topup_corrected": topup_corrected,
        "fieldmap": fieldmap
    }
    sio.savemat(result_file, result_dict)

    # remove temporary directory
    os.system(f"rm -rf {temp_dir}")

    print("TOPUP completed. Results saved to", result_file)

    
if __name__ == "__main__":
    main()