import os
import numpy as np
import nibabel as nib
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

# anisotropic diffusion smoothing for a given image
def anisotropic_diffusion(img, niter=1, kappa=80, gamma=0.1, voxelspacing=(0.8,0.8,0.8), option=3):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = tuple([slice(None, -1) if j == i else slice(None) for j in range(out.ndim)])
            deltas[i][slicer] = np.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = tuple([slice(1, None) if j == i else slice(None) for j in range(out.ndim)])
            matrices[i][slicer] = np.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (np.sum(matrices, axis=0))

    return out

# average numpy arrays of 3 planes
def average_planes(axial, coronal, sagittal):
    averaged_image = (axial + coronal + sagittal) / 3

    return averaged_image

# generate and save smoothed image for one subject for one sequence
def smooth_one_sub(subject, input_dir, output_dir, voxel_size=(1,1,1)):
    print(f'Processing: {subject}')

    # path to axial image
    axial_image_path = os.path.join(input_dir, 'recon_axial', f'{subject}_recon_pet.nii.gz')

    # axial image
    axial_image = nib.load(axial_image_path)

    # smoothed axial image
    smoothed_axial_image = anisotropic_diffusion(axial_image.get_fdata(), niter=40, voxelspacing=voxel_size)

    # path to coronal image
    coronal_image_path = os.path.join(input_dir, 'recon_coronal', f'{subject}_recon_pet.nii.gz')

    # coronal image
    coronal_image = nib.load(coronal_image_path)

    # smoothed coronal image
    smoothed_coronal_image = anisotropic_diffusion(coronal_image.get_fdata(), niter=40, voxelspacing=voxel_size)

    # path to sagittal image
    sagittal_image_path = os.path.join(input_dir, 'recon_sagittal', f'{subject}_recon_pet.nii.gz')

    # sagittal image
    sagittal_image = nib.load(sagittal_image_path)

    # smoothed sagittal image
    smoothed_sagittal_image = anisotropic_diffusion(sagittal_image.get_fdata(), niter=40, voxelspacing=voxel_size)

    # averaged image
    averaged_image = average_planes(smoothed_axial_image, smoothed_coronal_image, smoothed_sagittal_image)

    # smoothed averaged image path
    smoothed_averaged_path = os.path.join(output_dir, f'{subject}_FlowGAN_pet.nii.gz')

    # create nifti image
    smoothed_averaged_image = nib.Nifti1Image(averaged_image, axial_image.affine, axial_image.header)

    # save image
    nib.save(smoothed_averaged_image, smoothed_averaged_path)

    if os.path.exists(smoothed_averaged_path) == True:
        print(f'Successfully smoothed {subject}')
    
    else:
        print(f'Failed to smooth {subject}')

# iterate for all subs
def smooth_all_subs_in_series(full_subject_list, input_dir, output_dir, voxel_size=(1,1,1)):
    # create output directory if it doesn't already exist
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    for sub in full_subject_list:
        smooth_one_sub(sub, input_dir, output_dir, voxel_size)

def smooth_all_subs_in_parallel(full_subject_list, input_dir, output_dir, voxel_size=(1,1,1)):
    max_processes = 10

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    list_of_arguments = []

    # create list of argument tuples
    for sub in full_subject_list:
        list_of_arguments.append((sub, input_dir, output_dir, voxel_size))
    
    # spawn up to 10 processes at a time
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(smooth_one_sub, list_of_arguments)

    print('Finished parallel smoothing')


# if script is run
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Smooth coregistered reshaped reconstructed volumes using anisotropic diffusion')

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
            voxel_size=(int(args.voxel_size), int(args.voxel_size), int(args.voxel_size))
        )

        print('Finished parallel smoothing')
    
    else:
        smooth_all_subs_in_series(
            full_subject_list=full_subject_list,
            input_dir=os.path.abspath(args.data),
            output_dir=os.path.abspath(args.output_dir),
            voxel_size=(int(args.voxel_size), int(args.voxel_size), int(args.voxel_size))
        )
    
    print('Finished')


