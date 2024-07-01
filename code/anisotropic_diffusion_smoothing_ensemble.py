import os
import numpy as np
import nibabel as nib
import argparse
import multiprocessing
import anisotropic_diffusion_smoothing

# get list of subjects from subjects file
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

# smooth for each fold in series
def smooth_in_series(full_subject_list, input_dir, output_dir, voxel_size=(1, 1, 1), n_splits=12):
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # process each fold in series
    for fold in folds:
        input_dir_fold = os.path.join(input_dir, fold, 'recon_niftis_coregistered')
        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_smoothed')

        anisotropic_diffusion_smoothing.smooth_all_subs_in_series(full_subject_list, input_dir_fold, output_dir_fold, voxel_size)

# smooth for each fold in parallel
def smooth_in_parallel(full_subject_list, input_dir, output_dir, voxel_size=(1, 1, 1), n_splits=12):
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    max_processes = n_splits

    list_of_arguments = []

    # create list of argument tuples
    for fold in folds:
        input_dir_fold = os.path.join(input_dir, fold, 'recon_niftis_coregistered')
        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis_smoothed')

        list_of_arguments.append((full_subject_list, input_dir_fold, output_dir_fold, voxel_size))

    # spawn up to n_splits processes at a time
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(anisotropic_diffusion_smoothing.smooth_all_subs_in_series, list_of_arguments)

    print('Finished parallel smoothing')

# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Smooth coregistered reshaped reconstructed volumes from ensemble model using anisotropic diffusion')

    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with where coregistered reshaped reconstructed volumes from each plane are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the smoothed volumes',
                        required=True,
                        )
    
    # (OPTIONAL) voxel size
    parser.add_argument('-voxel_size','--voxel_size',
                        help='Voxel size',
                        default=1,
                        required=False,
                        )
    
    # (OPTIONAL) number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of folds',
                        required=False,
                        default=12,
                        )
    
    # (OPTIONAL) parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel, 10 at a time',
                        default=False,
                        required=False,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    # run in parallel
    if bool(args.parallel) == True:
        smooth_all_subs_in_parallel(
            full_subject_list=full_subject_list,
            input_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            voxel_size=(int(args.voxel_size), int(args.voxel_size), int(args.voxel_size)),
            n_splits=int(args.n_splits)
        )

        print('Finished parallel smoothing')
    
    else:
        smooth_all_subs_in_series(
            full_subject_list=full_subject_list,
            input_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            voxel_size=(int(args.voxel_size), int(args.voxel_size), int(args.voxel_size)),
            n_splits=int(args.n_splits)
        )
    
    print('Finished')

