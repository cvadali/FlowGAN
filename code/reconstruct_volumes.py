import os
import numpy as np
import cv2
import nibabel as nib
import argparse
import multiprocessing
import math

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

# create numpy array from generated images in one plane
def get_volume_predicted_single_plane(subject, pix2pix_out_dir, model_name, plane, phase='test', epoch='latest'):
    # all image outputs
    all_images = os.listdir(os.path.join(pix2pix_out_dir, model_name, f'{phase}_{str(epoch)}', 'images'))

    # image outputs for subject
    images_sub = [image for image in all_images if subject in image]

    # find number of images that were output for this subject
    largest_number = 0

    for image in images_sub:
        number = int(image.split(f'{subject}_')[-1].split('_')[0])
        if number > largest_number:
            largest_number = number

    # list to store volume
    dicom = np.zeros((largest_number+1, largest_number+1, largest_number+1))
    
    # iterate for each image
    for i in range(largest_number+1):
        try:
            # filename stem
            file_stem = f'{subject}_{str(i)}'

            # image file name
            # generated images end in _fake_B.png
            recon_filename = os.path.join(pix2pix_out_dir, model_name, f'{phase}_{str(epoch)}', 'images', f'{file_stem}_fake_B.png')

            # add image to list
            image = cv2.imread(recon_filename)[:,:,0]

            if plane == 'axial':
                dicom[:,:,i] = image
            elif plane == 'sagittal':
                dicom[i,:,:] = image
            elif plane == 'coronal':
                dicom[:,i,:] = image

        except Exception as e:
            print(e)
            continue
    return np.array(dicom)

# if exactly 2 dimensions are the maximum
def crop_to_original_dimensions_2(pet_volume, t1_dim_sagittal, t1_dim_coronal, t1_dim_axial):
    max_dim = max(t1_dim_sagittal, t1_dim_coronal, t1_dim_axial)

    # if sagittal and coronal are the maximum dimensions, crop axial
    if t1_dim_sagittal == t1_dim_coronal:
        shift_axial = (max_dim - t1_dim_axial) / 2

        pet = pet_volume[:,:, math.floor(shift_axial):-math.floor(shift_axial)]
    
    # if sagittal and axial are the maximum dimensions, crop coronal
    elif t1_dim_sagittal == t1_dim_axial:
        shift_coronal = (max_dim - t1_dim_coronal) / 2

        pet = pet_volume[:, math.floor(shift_coronal):-math.floor(shift_coronal), :]
    
    # if coronal and axial are the maximum dimensions, crop sagittal
    elif t1_dim_coronal == t1_dim_axial:
        shift_sagittal = (max_dim - t1_dim_sagittal) / 2

        pet = pet_volume[math.floor(shift_sagittal):-math.floor(shift_sagittal), :, :]
    
    else:
        print('Error: 2 dimensions are not the maximum')

    return pet

# if exactly 1 dimension is the maximum
def crop_to_original_dimensions_1(pet_volume, t1_dim_sagittal, t1_dim_coronal, t1_dim_axial):
    max_dim = max(t1_dim_sagittal, t1_dim_coronal, t1_dim_axial)

    # if sagittal is the maximum dimension, crop coronal and axial
    if t1_dim_sagittal == max_dim:
        shift_coronal = (max_dim - t1_dim_coronal) / 2
        shift_axial = (max_dim - t1_dim_axial) / 2

        pet = pet_volume[:, math.floor(shift_coronal):-math.floor(shift_coronal), math.floor(shift_axial):-math.floor(shift_axial)]
    
    # if coronal is the maximum dimension, crop sagittal and axial
    elif t1_dim_coronal == max_dim:
        shift_sagittal = (max_dim - t1_dim_sagittal) / 2
        shift_axial = (max_dim - t1_dim_axial) / 2

        pet = pet_volume[math.floor(shift_sagittal):-math.floor(shift_sagittal), :, math.floor(shift_axial):-math.floor(shift_axial)]
    
    # if axial is the maximum dimension, crop sagittal and coronal
    elif t1_dim_axial == max_dim:
        shift_sagittal = (max_dim - t1_dim_sagittal) / 2
        shift_coronal = (max_dim - t1_dim_coronal) / 2

        pet = pet_volume[math.floor(shift_sagittal):-math.floor(shift_sagittal), math.floor(shift_coronal):-math.floor(shift_coronal), :]
    
    else:
        print('Error: 1 dimension is not the maximum')
    
    return pet

# crop to original dimensions
def crop_to_original_dimensions(pet_volume, t1_dim_sagittal, t1_dim_coronal, t1_dim_axial):
    # find maximum dimension
    max_dim = max(t1_dim_sagittal, t1_dim_coronal, t1_dim_axial)

    # see if exactly two dimensions are the maximum
    count_max = sum(1 for number in (t1_dim_sagittal, t1_dim_coronal, t1_dim_axial) if number == max_dim)

    # if all dimensions are the same, no need to crop
    if count_max == 3:
        return pet_volume

    # 2 are the maximum dimensions
    elif count_max == 2:
        pet = crop_to_original_dimensions_2(pet_volume, t1_dim_sagittal, t1_dim_coronal, t1_dim_axial)

    # 1 maximum dimension
    elif count_max == 1:
        pet = crop_to_original_dimensions_1(pet_volume, t1_dim_sagittal, t1_dim_coronal, t1_dim_axial)
    
    else:
        print('Error: no maximum dimension')
    
    return pet

# save reconstructed nifti image
def save_volume_predicted_single_plane(subject, data_dir, t1_dir, output_dir, model_name_stem, plane, phase='test', epoch='latest'):
    print(f'Processing: {subject}_{plane}')

    # name of model used to generate images
    model_name = f'{model_name_stem}_{plane}'

    # numpy array containing image
    volume = get_volume_predicted_single_plane(subject, data_dir, model_name, plane, phase, epoch)

    # original T1 image
    t1_image = nib.load(os.path.join(t1_dir, subject, 'derivatives', 'registered_images', f'{subject}_T1_in_T1.nii.gz'))

    # get original T1 dimensions
    t1_dimensions = t1_image.shape

    # remove zero padding and crop to original dimensions
    pet = crop_to_original_dimensions(volume, t1_dimensions[0], t1_dimensions[1], t1_dimensions[2])

    # output directory for this plane
    output_dir_plane = os.path.join(output_dir, f'recon_{plane}')
    
    # output
    pet_filepath = os.path.join(output_dir_plane, f'{subject}_recon_pet.nii.gz')

    # use T1 affine
    pet_affine = t1_image.affine

    # save image
    nib.save(nib.Nifti1Image(pet, pet_affine), pet_filepath)

    save_success = os.path.exists(pet_filepath)

    return save_success

# iterate over a list of subjects
def iterate_for_each_sub(full_subject_list, data_dir, t1_dir, output_dir, model_name_stem, plane, phase='test', epoch='latest'):

    # iterate for each subject
    for sub in full_subject_list:
        success = save_volume_predicted_single_plane(sub, data_dir, t1_dir, output_dir, model_name_stem, plane, phase, epoch)

        if success == True:
            print(f'Completed: {sub}_{plane}')
        
        else:
            print(f'Failed: {sub}_{plane}')

# iterate over each plane
def reconstruct_in_series(full_subject_list, data_dir, t1_dir, output_dir, model_name_stem, phase='test', epoch='latest'):
    if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)

    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:     
        # output directory for this plane
        output_dir_plane = os.path.join(output_dir, f'recon_{plane}')

        if os.path.exists(output_dir_plane) == False:
            os.makedirs(output_dir_plane)
        
        iterate_for_each_sub(full_subject_list, data_dir, t1_dir, output_dir, model_name_stem, plane, phase, epoch)

# reconstruct in parallel
def reconstruct_in_parallel(full_subject_list, model_name_stem, data_dir, t1_dir, output_dir, phase='test', epoch='latest'):
    print('Reconstructing in parallel')

    max_processes = 10

    planes = ['axial', 'coronal', 'sagittal']
    
    list_of_arguments = []

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    for plane in planes:
        output_dir_plane = os.path.join(output_dir, f'recon_{plane}')

        if os.path.exists(output_dir_plane) == False:
            os.makedirs(output_dir_plane)

        for sub in full_subject_list:
            list_of_arguments.append((sub, 
                                        data_dir, 
                                        t1_dir, 
                                        output_dir, 
                                        model_name_stem, 
                                        plane, 
                                        phase, 
                                        epoch))
    
    # spawn up 10 processes at once
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(save_volume_predicted_single_plane, list_of_arguments)

# if script is actually run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reconstruct nifti PET volumes from generated test set images')

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
                        help='Directory where generated images in each plane are stored',
                        required=True,
                        )
    
    # t1 directory
    parser.add_argument('-t1_dir','--t1_dir',
                        help='Directory with original T1 images',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the reconstructed niftis',
                        required=True,
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
        reconstruct_in_parallel(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            t1_dir=os.path.abspath(args.t1_dir),
            output_dir=os.path.abspath(args.output_dir),
            model_name_stem=args.model_name_stem,
            phase=args.phase,
            epoch=args.epoch
        )

    # run in series
    else:
        reconstruct_in_series(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            t1_dir=os.path.abspath(args.t1_dir),
            output_dir=os.path.abspath(args.output_dir),
            model_name_stem=args.model_name_stem,
            phase=args.phase,
            epoch=args.epoch
        )

    print('Finished')


