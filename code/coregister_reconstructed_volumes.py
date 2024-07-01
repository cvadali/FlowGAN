import os
import numpy as np
import ants
import sys
import argparse
import multiprocessing

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

# register single image to target
def register_image(path_source, path_target):

    # Load the source and target images
    source_image = ants.image_read(path_source)
    target_image = ants.image_read(path_target)

    # Perform image registration
    registration_result = ants.registration(
        fixed=target_image,
        moving=source_image,
        type_of_transform="DenseRigid",
        verbose=False
    )

    # Get the registered image
    registered_image = registration_result['warpedmovout']
    
    return registered_image

# save registered image
def save_registered_image(path_source, path_target, out_path):

    registered_image = register_image(path_source, path_target)

    ants.image_write(registered_image, out_path)

    success = os.path.exists(out_path)

    return success

# coregister images for a single subject in a single modality
def coregister_planes_single_subject(subject, recon_dir, out_dir):
    planes = ['coronal', 'sagittal']

    reshaped_axial_name = f'{subject}_recon_pet.nii.gz'

    reshaped_axial_path = os.path.join(recon_dir, 'recon_axial', reshaped_axial_name)

    registered_axial_path = os.path.join(out_dir, 'recon_axial', reshaped_axial_name)

    os.system(f'cp {reshaped_axial_path} {registered_axial_path}')

    for plane in planes:
        reshaped_image_name = f'{subject}_recon_pet.nii.gz'

        reshaped_image_path = os.path.join(recon_dir, f'recon_{plane}', reshaped_image_name)

        print(f'Processing {subject} {plane}')

        registered_image_path = os.path.join(out_dir, f'recon_{plane}', reshaped_image_name)

        successful_registration = save_registered_image(reshaped_image_path, reshaped_axial_path, registered_image_path)

        if successful_registration == True:
            print(f'Successfully registered {subject} {plane}')
        
        else:
            print(f'Failed registration of {subject} {plane}')
            sys.exit()

# iterate over list of subjects
def coregister_all_subjects(full_subject_list, recon_dir, out_dir):
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    out_dir_axial = os.path.join(out_dir, 'recon_axial')
    if os.path.exists(out_dir_axial) == False:
        os.makedirs(out_dir_axial)
    
    out_dir_coronal = os.path.join(out_dir, 'recon_coronal')
    if os.path.exists(out_dir_coronal) == False:
        os.makedirs(out_dir_coronal)

    out_dir_sagittal = os.path.join(out_dir, 'recon_sagittal')
    if os.path.exists(out_dir_sagittal) == False:
        os.makedirs(out_dir_sagittal)
    
    for sub in full_subject_list:
        coregister_planes_single_subject(sub, recon_dir, out_dir)



# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Coregister reshaped reconstructed volumes from FLowGAN inference')

    # file with all subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File with all of the subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with where reshaped reconstructed volumes from each plane are stored',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the coregistered reconstructed volumes',
                        required=True,
                        )

    args = parser.parse_args()

    print(args)

    print('Starting')

    full_subject_list = get_subject_list(args.subs_file)

    coregister_all_subjects(
        full_subject_list=full_subject_list,
        recon_dir=os.path.abspath(args.data),
        out_dir=os.path.abspath(args.output_dir)
    )
    
    print('Finished')

