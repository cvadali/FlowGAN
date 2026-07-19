import os
import numpy as np
import cv2
import nibabel as nib
import argparse
import multiprocessing
import math
import reconstruct_volumes

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

# reconstruct volumes from each fold in series
def reconstruct_volumes_series(full_subject_list, data_dir, t1_dir, output_dir, model_name_stem, phase='test', epoch='latest', n_splits=12):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    for fold in folds:
        print(f'Reconstructing volumes for {fold}')

        data_dir_fold = os.path.join(data_dir, fold, 'results_FlowGAN')

        output_dir_fold = os.path.join(output_dir, fold, 'recon_niftis')

        model_name_stem_fold = f'{model_name_stem}_{fold}'

        reconstruct_volumes.reconstruct_in_series(
            full_subject_list=full_subject_list,
            data_dir=data_dir_fold,
            t1_dir=t1_dir,
            output_dir=output_dir_fold,
            model_name_stem=model_name_stem_fold,
            phase=phase,
            epoch=epoch
        )
    
    print('Finished reconstructing volumes in series')

# reconstruct volumes from each fold in parallel
def reconstruct_volumes_parallel(full_subject_list, data_dir, t1_dir, output_dir, model_name_stem, phase='test', epoch='latest', n_splits=12):
    max_processes = n_splits

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # get list of folds
    folds = [f'fold_{fold}' for fold in range(0, n_splits)]

    list_of_args = []

    for fold in folds:
        list_of_args.append((full_subject_list, 
                             os.path.join(data_dir, fold, 'results_FlowGAN'), 
                             t1_dir, 
                             os.path.join(output_dir, fold, 'recon_niftis'), 
                             f'{model_name_stem}_{fold}', 
                             phase, 
                             epoch))

    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(reconstruct_volumes.reconstruct_in_series, list_of_args)
    
    print('Finished reconstructing volumes in parallel')

# if script is actually run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reconstruct nifti PET volumes from generated ensemble test set images')

    # subjects file
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all subjects',
                        required=True,
                        )

    # model name stem
    parser.add_argument('-model_name_stem','--model_name_stem',
                        help='Stem of the name of the model',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory where outputs from ensemble model are stored',
                        required=True,
                        )
    
    # t1 directory
    parser.add_argument('-t1_dir','--t1_dir',
                        help='Directory with original T1 images',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the reconstructed volumes',
                        required=True,
                        )

    # (OPTIONAL) number of folds
    parser.add_argument('-n_splits','--n_splits',
                        help='Number of folds',
                        default=12,
                        required=False,
                        )
    
    # (OPTIONAL) phase (train, val, test)
    parser.add_argument('-phase','--phase',
                        help='Phase (test, train, val)',
                        default='test',
                        required=False,
                        )
    
    # (OPTIONAL) epoch of model used to generate images
    parser.add_argument('-epoch','--epoch',
                        help='Epoch of model used to generate images',
                        default='latest',
                        required=False,
                        )
    
    # (OPTIONAL) process in parallel
    parser.add_argument('-parallel', '--parallel',
                        help='Run reconstruction in parallel',
                        default=False,
                        required=False,
                        )
    

    args = parser.parse_args()

    print(args)

    print('Starting')

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    # process in parallel, 10 at a time
    if bool(args.parallel) == True:
        reconstruct_volumes_parallel(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            t1_dir=os.path.abspath(args.t1_dir),
            output_dir=os.path.abspath(args.output_dir),
            model_name_stem=args.model_name_stem,
            phase=args.phase,
            epoch=args.epoch,
            n_splits=int(args.n_splits)
        )

    # run in series
    else:
        reconstruct_volumes_series(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            t1_dir=os.path.abspath(args.t1_dir),
            output_dir=os.path.abspath(args.output_dir),
            model_name_stem=args.model_name_stem,
            phase=args.phase,
            epoch=args.epoch,
            n_splits=int(args.n_splits)
        )

    print('Finished')
