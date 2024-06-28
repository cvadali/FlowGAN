import os
import ants
import nibabel as nib
import argparse
import scipy

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

# reorient images to LPI
def reorient_images(path_source):
    # Load the image data
    image = ants.image_read(path_source)

    # Reorient the image to LPI
    reoriented_image = image.reorient_image2('LPI')

    return reoriented_image

# register images
def register_images(source_image, target_image, path_out):

    # Perform image registration
    registration_result = ants.registration(
        fixed=target_image,
        moving=source_image,
        type_of_transform="DenseRigid",
        verbose=False
    )

    # Get the registered image
    registered_image = registration_result['warpedmovout']

    # Save the registered image
    ants.image_write(registered_image, path_out)
    
    xfm = registration_result['fwdtransforms']
    
    return registered_image, xfm

# resample image to 1mm isotropic voxels
def resample_image(img):
    # use 1mm isotropic resampled image
    image = img.copy()

    # Calculate the new voxel spacing.
    new_spacing = [1, 1, 1]
  
    # Resample the image to 1mm isotropic.
    resampled_image = ants.resample_image(image, new_spacing, interp_type=0)

    return resampled_image

# smooth with gaussian filter
def smooth_with_gaussian(image, sigma):
    # get image data
    image_data = image.get_fdata()

    # smooth image with gaussian filter
    smoothed_image = scipy.ndimage.gaussian_filter(image_data, sigma=sigma)

    # create smoothed image
    smoothed_image = nib.Nifti1Image(smoothed_image, affine=image.affine)

    return smoothed_image

def reorient_register_and_smooth_each_subject(full_subject_list, data_dir):
    
    modalities = ['T1', 'cbf']

    for sub in full_subject_list:
        try:

            print(f'Processing: {sub}')

            # directory for subject
            sub_dir = os.path.join(data_dir, sub)

            # derivatives for subject
            derivatives_dir = os.path.join(sub_dir, 'derivatives')

            if os.path.exists(derivatives_dir) == False:
                os.makedirs(derivatives_dir)

            # directory within derivatives containing registered images
            registered_images_dir = os.path.join(derivatives_dir, 'registered_images')

            if os.path.exists(registered_images_dir) == False:
                os.makedirs(registered_images_dir)
            
            for modality in modalities:
                print(f'Processing: {sub} {modality}')

                print('reorienting to LPI')
                
                # reorient to LPI
                reoriented_image = resample_image(reorient_images(os.path.join(sub_dir, f'{sub}_{modality}.nii.gz')))

                print('save LPI image')

                # save LPI image
                ants.image_write(reoriented_image, os.path.join(registered_images_dir,f'{sub}_{modality}_LPI.nii.gz'))

                print('registering to T1')

                if modality == 'T1':
                    ants.image_write(reoriented_image, os.path.join(registered_images_dir,f'{sub}_{modality}_in_T1.nii.gz'))

                elif modality.lower() == 'cbf':
                    # register to T1
                    reg_image, xfm = register_images(reoriented_image,
                                                            ants.image_read(os.path.join(registered_images_dir, f'{sub}_T1_in_T1.nii.gz')),
                                os.path.join(registered_images_dir,f'{sub}_{modality}_in_T1.nii.gz'))

                    print('Smoothing with gaussian filter')
                    # smooth cbf in T1 space with gaussian filter
                    # once at sigma=1
                    # and once at sigma=3
                    registered_cbf = nib.load(os.path.join(registered_images_dir, f'{sub}_cbf_in_T1.nii.gz'))

                    smoothed_image_sigma_1 = smooth_with_gaussian(registered_cbf, sigma=1)

                    smoothed_image_sigma_3 = smooth_with_gaussian(registered_cbf, sigma=3)

                    nib.save(smoothed_image_sigma_1, os.path.join(registered_images_dir, f'{sub}_smoothed_cbf_sigma_1_in_T1.nii.gz'))

                    nib.save(smoothed_image_sigma_3, os.path.join(registered_images_dir, f'{sub}_smoothed_cbf_sigma_3_in_T1.nii.gz'))


                else:
                    print(f'Unknown modality: {modality}')
                    
            
        except Exception as e:
            print(e)

# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Reorient, register, and smooth T1 and ASL CBF in preparation for running through FLowGAN')

     # file containing name of subjects
    parser.add_argument('-subs_file','--subs_file',
                        help='File containing subjects',
                        required=True,
                        )
    
    # data source
    parser.add_argument('-data','--data',
                        help='Directory with source data',
                        required=True,
                        )
    
    args = parser.parse_args()

    # print arguments
    print(args)

    print('Starting')

    full_subject_list = get_subject_list(os.path.abspath(args.subs_file))

    reorient_register_and_smooth_each_subject(
        full_subject_list=full_subject_list,
        data_dir=os.path.abspath(args.data)
    )

    print('Finished')


