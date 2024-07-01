import os
import argparse
import multiprocessing
import reorient_register_smooth
import generate_pix2pix_datasets
import FLowGAN_inference
import FLowGAN_inference_ensemble
import reconstruct_volumes
import reconstruct_volumes_ensemble
import coregister_reconstructed_volumes
import coregister_reconstructed_volumes_ensemble
import anisotropic_diffusion_smoothing
import anisotropic_diffusion_smoothing_ensemble
import coregister_smoothed_outputs_ensemble
import averaging_smoothed_ensemble

"""
This script runs the full FLowGAN pipeline on data in BIDS format. The pipeline includes the following steps:

1. Reorient, register, and smooth (gaussian filter for CBF maps) each volume
2. Generate pix2pix datasets
3. Run inference with FLowGAN pix2pix networks
4. Reconstruct and reshape volumes from pix2pix outputs
6. Coregister reshaped reconstructed volumes
7. Smooth volumes using anisotropic diffusion


The required arguments are:

-subs_file: Path to file containing list of subjects
-data: Path to directory with subjects in BIDS format (each with T1w and ASL CBF volumes)
-output_dir: Path to directory to output all outputs from FLowGAN

The optional arguments are:

-parallel: Run the pipeline in parallel
-ensemble: Use an ensemble of 12 models and average the outputs. This generates more robust results but takes longer
-intermediates: Keep intermediate outputs from the pipeline. This is useful for debugging

"""


def get_subject_list(subs_file):
    # file containing subjects
    subject_file = open(subs_file, 'r')

    # list of all subjects
    full_subject_list = []

    # convert to list
    for line in subject_file:
        line = line.split('\n')[0]
        full_subject_list.append(line)

    return full_subject_list

# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Run full FLowGAN pipeline on data in BIDS format')

    parser.add_argument('-subs_file','--subs_file', help='File containing list of subjects', type=str, required=True)
    parser.add_argument('-data','--data', help='Directory with subjects in BIDS format (each with T1w and ASL CBF volumes)', type=str, required=True)
    parser.add_argument('-output_dir','--output_dir', help='Directory to output all outputs from FLowGAN', type=str, required=True)
    parser.add_argument('-parallel','--parallel', help='(OPTIONAL) Run in parallel', action='store_true')
    parser.add_argument('-ensemble', '--ensemble', help='(OPTIONAL) Use ensemble of 12 models and average', action='store_true')
    parser.add_argument('-intermediates', '--intermediates', help='(OPTIONAL) Keep intermediate outputs', action='store_true')
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    # path from where script is run
    # important to save because the test script changes the working directory
    original_path = os.getcwd()

    # list of subjects
    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    print('Number of subjects: ', len(full_subject_list))

    print('Reorienting, registering, and smoothing')

    # reorient, register, and smooth (ASL CBF maps) each volume
    reorient_register_smooth.reorient_register_and_smooth_each_subject(
        full_subject_list=full_subject_list,
        data_dir=os.path.abspath(args.data)
    )

    print('Finished reorienting, registering, and smoothing')

    print('Generating pix2pix datasets')

    parallel = args.parallel

    ensemble = args.ensemble

    intermediates = args.intermediates

    # process in parallel
    if parallel:
        print('Generating pix2pix datasets in parallel')
        generate_pix2pix_datasets.create_dataset_parallel(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir)
        )

    # run in series
    else:
        generate_pix2pix_datasets.create_dataset_series(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir)
        )
    
    print('Finished generating pix2pix datasets')

    print('Running inference with FLowGAN pix2pix networks')

    # run inference

    if ensemble:
        print('Ensemble')

        FLowGAN_inference_ensemble.test_all_folds(
            model_dir=os.path.abspath('checkpoints/FLowGAN_ensemble/'),
            model_name='FLowGAN_ensemble',
            data_source=os.path.abspath(args.output_dir),
            output_dir=os.path.abspath(args.output_dir),
            pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath('pytorch-CycleGAN-and-pix2pix/'),
            direction='BtoA',
            n_splits=12
        )

    else:
        print('Normal')

        FLowGAN_inference.test_model(
            model_name='FLowGAN',
            data_source=os.path.abspath(args.output_dir),
            checkpoints_dir=os.path.abspath('checkpoints/'),
            output_dir=os.path.abspath(args.output_dir),
            pytorch_CycleGAN_and_pix2pix_dir=os.path.abspath('pytorch-CycleGAN-and-pix2pix/'),
            direction='BtoA'
        )

    print('Finished inference with FLowGAN pix2pix networks')

    # change back to original path
    os.chdir(original_path)

    print('Reconstructing and reshape volumes from pix2pix outputs')

    # reconstruct images from pix2pix outputs

    if ensemble:
        print('Ensemble')
        # process in parallel
        if parallel:
            print('Reconstructing volumes from pix2pix outputs in parallel')
            
            reconstruct_volumes_ensemble.reconstruct_volumes_parallel(
                full_subject_list=full_subject_list,
                data_dir=os.path.abspath(args.output_dir),
                t1_dir=os.path.abspath(args.data),
                output_dir=os.path.abspath(args.output_dir),
                model_name_stem='FLowGAN_ensemble',
                phase='test',
                epoch='latest',
                n_splits=12
            )

        else:
            reconstruct_volumes_ensemble.reconstruct_volumes_series(
                full_subject_list=full_subject_list,
                data_dir=os.path.abspath(args.output_dir),
                t1_dir=os.path.abspath(args.data),
                output_dir=os.path.abspath(args.output_dir),
                model_name_stem='FLowGAN_ensemble',
                phase='test',
                epoch='latest',
                n_splits=12
            )

    else:
        print('Normal')
        # process in parallel
        if parallel:
            print('Reconstructing volumes from pix2pix outputs in parallel')
            
            reconstruct_volumes.reconstruct_in_parallel(
                full_subject_list=full_subject_list,
                model_name_stem='FLowGAN',
                data_dir=os.path.join(os.path.abspath(args.output_dir), 'results_FLowGAN'),
                t1_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis'),
                phase='test',
                epoch='latest'
            )

        # run in series
        else:
            reconstruct_volumes.reconstruct_in_series(
                full_subject_list=full_subject_list,
                model_name_stem='FLowGAN',
                data_dir=os.path.join(os.path.abspath(args.output_dir), 'results_FLowGAN'),
                t1_dir=os.path.abspath(args.data),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis'),
                phase='test',
                epoch='latest'
            )
    
    print('Finished reconstructing and reshaping volumes from pix2pix outputs')

    print('Coregister reshaped reconstructed volumes')

    # coregister reshaped reconstructed volumes
    if ensemble:
        print('Ensemble')

        coregister_reconstructed_volumes_ensemble.coregister_all_subjects(
            full_subject_list=full_subject_list,
            recon_dir=os.path.abspath(args.output_dir),
            out_dir=os.path.abspath(args.output_dir),
            n_splits=12
        )

    else:
        print('Normal')

        coregister_reconstructed_volumes.coregister_all_subjects(
            full_subject_list=full_subject_list,
            recon_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis'),
            out_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_coregistered')
        )

    print('Finished coregistering reshaped reconstructed volumes')

    print('Smoothing volumes using anisotropic diffusion')

    if ensemble:
        if parallel:
            print('Ensemble')

            anisotropic_diffusion_smoothing_ensemble.smooth_in_parallel(
                full_subject_list=full_subject_list,
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                voxel_size=(1, 1, 1),
                n_splits=12
            )
        
        else:
            anisotropic_diffusion_smoothing_ensemble.smooth_in_series(
                full_subject_list=full_subject_list,
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.abspath(args.output_dir),
                voxel_size=(1, 1, 1),
                n_splits=12
            )

    else:
        print('Normal')

        if parallel:
            anisotropic_diffusion_smoothing.smooth_all_subs_in_parallel(
                full_subject_list=full_subject_list,
                input_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_coregistered'),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'FLowGAN_outputs'),
                voxel_size=(1, 1, 1)
            )
        
        else:
            anisotropic_diffusion_smoothing.smooth_all_subs_in_series(
                full_subject_list=full_subject_list,
                input_dir=os.path.join(os.path.abspath(args.output_dir), 'recon_niftis_coregistered'),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'FLowGAN_outputs'),
                voxel_size=(1, 1, 1)
            )

    print('Finished smoothing volumes using anisotropic diffusion')

    if ensemble:
        print('Coregistering smoothed outputs')

        coregister_smoothed_outputs_ensemble.coregister_all_subjects(
            full_subject_list=full_subject_list,
            ensemble_output_dir=os.path.abspath(args.output_dir),
            n_splits=12
        )

        print('Finished coregistering smoothed outputs')

        # average the outputs across each fold to increase signal-to-noise ratio
        print('Averaging smoothed volumes')

        if parallel:
            print('Averaging in parallel')

            averaging_smoothed_ensemble.average_in_parallel(
                full_subject_list=full_subject_list,
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'FLowGAN_outputs'),
                n_splits=12
            )

        else:
            print('Averaging in series')

            averaging_smoothed_ensemble.average_in_series(
                full_subject_list=full_subject_list,
                input_dir=os.path.abspath(args.output_dir),
                output_dir=os.path.join(os.path.abspath(args.output_dir), 'FLowGAN_outputs'),
                n_splits=12
            )

        print('Finished averaging smoothed volumes')

    if intermediates:
        print('Moving intermediate outputs to intermediates directory')

        os.makedirs(os.path.join(os.path.abspath(args.output_dir), 'intermediates'))

        # we will move only the specific directories that were generated by FLowGAN to the intermediates directory
        # to make sure that we don't mess with any other files that may be in the output directory

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        if ensemble:
            folds = [f'fold_{fold}' for fold in range(0, 12)]

            for fold in folds:
                os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), fold)} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        else:
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "results_FLowGAN")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')
            os.system(f'mv {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_coregistered")} {os.path.join(os.path.abspath(args.output_dir), "intermediates")}')

        print('Finished moving intermediate outputs to intermediates directory')

    else:
        print('Removing intermediate outputs')

        # we will move only the specific directories that were generated by FLowGAN to the intermediates directory
        # to make sure that we don't mess with any other files that may be in the output directory

        # remove generated files from reorient_register_skullstrip.py
        for sub in full_subject_list:
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_cbf_LPI.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_T1_LPI.nii.gz")}')
            os.system(f'rm {os.path.join(os.path.abspath(args.data), sub, "derivatives", "registered_images", sub + "_cbf_in_T1.nii.gz")}')

        planes = ['axial', 'coronal', 'sagittal']

        for plane in planes:
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), f"dataset_{plane}_pix2pix")}')

        if ensemble:
            folds = [f'fold_{fold}' for fold in range(0, 12)]

            for fold in folds:
                os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), fold)}')

        else:
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "results_FLowGAN")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis")}')
            os.system(f'rm -r {os.path.join(os.path.abspath(args.output_dir), "recon_niftis_coregistered")}')

        print('Finished removing intermediate outputs')


    print('Finished running FLowGAN')
