import numpy as np
import os
import cv2
import sys
import ants
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

# pad into a cube
def make_cube(img):
    # this function zero pads the image until it becomes a cube
    x,y,z = img.shape
    max_dim = np.max(img.shape)
    
    to_add_x = (max_dim - x) / 2
    to_add_y = (max_dim - y) / 2
    to_add_z = (max_dim -  z) / 2
    
    zero_padded = np.ones((int(x+(to_add_x*2)), 
                           int(y+(to_add_y*2)), 
                           int(z+(to_add_z*2))))*img[0,0,0]
    
    zero_padded[math.floor(to_add_x):x+math.floor(to_add_x), 
                math.floor(to_add_y):y+math.floor(to_add_y), 
                math.floor(to_add_z):z+math.floor(to_add_z)] = img
    
    return zero_padded

# normalize and pad to a cube
def norm_zero_to_one(vol3D):
    '''Normalize a 3D volume to zero to one range

    Parameters:
      vol3D (3D numpy array): 3D image volume
    '''

    # normalize to 0 to 1 range
    vol3D = vol3D - np.min(vol3D) # set lower bound
    vol3D = vol3D / ( np.max(vol3D) - np.min(vol3D) ) # set upper bound

    return make_cube(vol3D)

# load subject images
def load_subject_images(data_source, subject, plane):
    # load T1 and ASL CBF
    cbf_sigma_3 = ants.image_read(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_smoothed_cbf_sigma_3_in_T1.nii.gz'))
    cbf_sigma_1 = ants.image_read(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_smoothed_cbf_sigma_1_in_T1.nii.gz'))
    t1 = ants.image_read(os.path.join(data_source, subject, 'derivatives', 'registered_images', f'{subject}_T1_in_T1.nii.gz'))

    subject_dict = []

    # multiply by 255 for RGB conversion
    subject_dict.append(norm_zero_to_one((cbf_sigma_3).numpy()) * 255)
    subject_dict.append(norm_zero_to_one((cbf_sigma_1).numpy()) * 255)
    subject_dict.append(norm_zero_to_one((t1).numpy()) * 255)

    subject_dict = np.array(subject_dict)

    if plane == 'axial':
        subject_dict = subject_dict
    
    elif plane == 'coronal':
        subject_dict = subject_dict
    
    elif plane == 'sagittal':
        subject_dict = subject_dict
    
    else:
        print('Must enter valid plane')
        sys.exit()

    return subject_dict


# Function for concatenating three images
def concat_three_images(image1,image2,image3):
    return np.concatenate([image1[:,:,np.newaxis],image2[:,:,np.newaxis],image3[:,:,np.newaxis]],axis=2)

# concatenate asl + t1 and asl + t1 images next to each other (for pix2pix)
def concat_asl_asl(asl_array):
    asl_and_asl_array = np.concatenate([asl_array, asl_array], 1)

    return asl_and_asl_array

# create dataset for a single sub for a given plane
def create_dataset_single_sub_single_plane(subID, data_source, outdir, plane):
    print(f'Processing: {subID} {plane}')

    # preprocess and save out paired asl/asl datasets of PNG files
    # select appropriate volume
    array_asl = load_subject_images(data_source, subID, plane)

    output_path = os.path.join(outdir, f'dataset_{plane}_pix2pix', 'test')

    z = array_asl.shape[2] # process along the z-axis

    for i in range(0,z):
        filename = subID + '_' + str(i) + '.png'

        filepath = os.path.join(output_path, filename)

        # slice differently depending on plane
        if plane == 'sagittal':
            asl_png_array = concat_three_images(array_asl[0,i,:,:], array_asl[1,i,:,:], array_asl[2,i,:,:])
        elif plane == 'coronal':
            asl_png_array = concat_three_images(array_asl[0,:,i,:], array_asl[1,:,i,:], array_asl[2,:,i,:])
        elif plane == 'axial':
            asl_png_array = concat_three_images(array_asl[0,:,:,i], array_asl[1,:,:,i], array_asl[2,:,:,i])
        else:
            print('Must specify a plane')
            sys.exit()
        
        # combine asl and asl images side by side
        # asl + t1 on left, asl + t1 on right
        combined_asl_asl_png_array = concat_asl_asl(asl_png_array)

        # save combined asl asl image
        cv2.imwrite(filepath, combined_asl_asl_png_array)

# create dataset for a given plane (axial, coronal, or sagittal)
def create_dataset_single_plane(subs_file, data_source, outdir, plane):

    full_subject_list = get_subject_list(subs_file)

    # process all subjects
    for subID in full_subject_list:
        create_dataset_single_sub_single_plane(subID, data_source, outdir, plane)

# iterate over each plane
def create_dataset_series(subs_file, data_source, outdir):
    planes = ['axial', 'coronal', 'sagittal']

    # Check if the folder where the .png files will be output exists 
    # if not, create it
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    for plane in planes:
        print(plane)

        output_path = os.path.join(outdir, f'dataset_{plane}_pix2pix')

        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        
        output_path_test = os.path.join(outdir, f'dataset_{plane}_pix2pix', 'test')

        if os.path.exists(output_path_test) == False:
            os.makedirs(output_path_test)
        
        create_dataset_single_plane(subs_file, data_source, outdir, plane)
    
    print('Finished')

# create pix2pix dataset in parallel
def create_dataset_parallel(full_subject_list, data_dir, output_dir):
    print('Processing in parallel')

    max_processes = 10
    
    list_of_arguments = []

    planes = ['axial', 'coronal', 'sagittal']

    # Check if the folder where the .png files will be output exists 
    # if not, create it
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    # iterate for each subject
    for sub in full_subject_list:
        for plane in planes:
            output_path = os.path.join(output_dir, f'dataset_{plane}_pix2pix')

            if os.path.exists(output_path) == False:
                os.makedirs(output_path)
            
            output_path_test = os.path.join(output_dir, f'dataset_{plane}_pix2pix', 'test')

            if os.path.exists(output_path_test) == False:
                os.makedirs(output_path_test)

            list_of_arguments.append((sub, data_dir, output_dir, plane))
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(create_dataset_single_sub_single_plane, list_of_arguments)
    
    print('Finished parallel processing')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Create test pix2pix dataset for FLowGAN inference')
    
    # file with subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing list of subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with subjects (each with ASL CBF and T1 volumes)',
                        required=True,
                        )
    
    # output directory
    parser.add_argument('-output_dir','--output_dir',
                        help='Directory to output the pix2pix images',
                        required=True,
                        )
    
    # (OPTIONAL) run in parallel
    parser.add_argument('-parallel','--parallel',
                        help='Run in parallel',
                        required=False,
                        default=False,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    if bool(args.parallel) == True:
        full_subject_list = get_subject_list(os.path.abspath(args.subs_file))
        
        create_dataset_parallel(
            full_subject_list=full_subject_list,
            data_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir)
        )

    else:
        create_dataset_series(
            subs_file=os.path.abspath(args.subs_file),
            data_source=os.path.abspath(args.data),
            outdir=os.path.abspath(args.output_dir)
        )

    print('Finished')
