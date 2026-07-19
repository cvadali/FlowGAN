# FlowGAN

This repository contains our implementation of FlowGAN — a generative adversarial network (GAN) for T1 and ASL CBF to FDG-PET image translation. The architecture is based off of our previous work, [LowGAN](https://github.com/cvadali/LowGAN), which is a GAN model for low-to-high field (64mT to 3T) MR image translation, but with some minor tweaks. 

We actually **trained this network separately on two sets of data**: a cohort of **temporal lobe epilepsy (TLE)** subjects and a cohort with both **mild cognitive impairment (MCI)** and control subjects. Our motivation for training models for both of these cases was because FDG-PET can be an important tool for clinical diagnosis of both TLE (lateralization and/or SOZ localization) and Alzheimer's Disease (AD), which can initially manifest as MCI and then progress.

While the two models were trained on separate datasets, we've actually found that a model trained on one dataset still performs quite solidly when tested in the opposite dataset (e.g., a model trained in the TLE cohort and tested on the MCI cohort). Therefore, feel free to try out both sets of models on your data!

The code can run on either GPU or CPU, and provides the option of running certain steps of the pipeline in parallel or series, and also whether to use the standard model trained on the full dataset or to use the "ensemble" model and average the outputs at the end. The ensemble model was created because we performed 12-fold cross-validation on our data, meaning we split all of the subjects in the dataset into 12 different folds, where each subject was part of the training set in 11 of the folds and part of the test set in exactly one fold, and then trained and tested a separate version of our network for each of these 12 folds. As a result, we ended up with 12 trained FlowGAN networks, and we found that while every model produces an output with a little noise, different models don't necessarily produce the exact same noise for the same input (i.e., the noise is a bit random); therefore, if we use each of the 12 models to create a set of outputs and then average those outputs across the 12 models, we get **outputs with improved signal-to-noise ratios**. The tradeoff is that the ensemble model takes longer to run and is more computationally expensive, as it involves computing the outputs of 12 FlowGAN networks instead of just 1, so it may not be the best choice for every use case. The normal model already performs quite well, but for particularly challenging ASL CBF inputs, the ensemble model may be a better approach (albeit a more lengthy and computationally expensive one).


## Installation

1. Clone this GitHub repository.
2. Create a virtual conda environment to install the necessary packages using the chosen requirements file by running the following command:
    `conda create -n FlowGAN python=3.11`

    Then, activate the conda environment with the following command:
    `conda activate FlowGAN`

    Then, if you want to run the code on **CPU**, run the following command:
    `pip install -r requirements_cpu.txt`

    If you instead want to run the code on **GPU**, run the following command:
    `pip install -r requirements_gpu.txt`

3. To clone the necessary pix2pix submodule, run `git submodule init` and then `git submodule update`
4. Download the `checkpoints.tar.gz` file containing the models from this [Google Drive](https://drive.google.com/file/d/1d7LlHWllhU6Yy8lm3KfJ8b5iEk2dsCJY/view?usp=sharing) and then copy the file to the [code](https://github.com/cvadali/FlowGAN/tree/main/code) directory.
    
    Unarchive and unzip the file by running the following command: `tar -xvzf checkpoints.tar.gz`

    After it has been unarchived and unzipped, you can remove the `checkpoints.tar.gz` file using `rm checkpoints.tar.gz`

    You should end up with both cohorts' weights laid out like this:

    ```
    code/checkpoints
    ├── FlowGAN_TLE
    │   ├── FlowGAN_axial
    │   ├── FlowGAN_coronal
    │   ├── FlowGAN_sagittal
    │   └── FlowGAN_ensemble
    │       └── fold_0 ... fold_11
    └── FlowGAN_MCI
        └── (same layout)
    ```

## Data format

FlowGAN takes a **T1-weighted volume and an ASL CBF map** for each subject, and returns a synthetic FDG-PET volume.

Your data should be in the following format:

```
data
├── P001
│   ├── P001_T1.nii.gz
│   └── P001_cbf.nii.gz
└── P002
    ├── P002_T1.nii.gz
    └── P002_cbf.nii.gz
```

That is, one directory per subject, named with the subject ID, containing exactly two files: `<ID>_T1.nii.gz` and `<ID>_cbf.nii.gz`. **The filenames must follow this pattern**, since the pipeline looks them up by name.

The pipeline writes its own intermediates into `<subject>/derivatives/` as it runs, so you do not need to create that directory yourself.

You will also need to **create a .txt file with the ID of each subject**. For example, above, there are 2 subjects, P001 and P002, so the subject file, which I will call `list_of_subjects.txt`, will look like this:
```
P001
P002
```

If there were 20 subjects, the file would look like this:
```
P001
P002
P003
.
.
.

P020
```

Just make sure to **place each subject ID on a new line**


## Usage

Make sure that **you are in the `code` directory**. If you are not there, run `cd code` from the `FlowGAN` directory to get there.

Additionally, **make sure your data is in the format specified above and that you created a .txt file with each subject**.

Now, you can run the code with just one command:

`python run_FlowGAN.py --subs_file <subs_file> --data <data> --output_dir <output_dir> [--cohort TLE|MCI --parallel --ensemble --intermediates]`

**Necessary arguments**:

- `<subs_file>` is the path to a .txt file in the style above that contains the IDs of the subjects that you will be running
- `<data>` is the path to a directory containing the data in the format above
- `<output_dir>` is the path of the directory where you want to save the outputs

**Optional arguments**:

- `<cohort>` selects which trained model to use, `TLE` or `MCI` (default: `TLE`). As described above, each model was trained on a separate dataset, but both generalize reasonably well to the other cohort, so it is worth trying both on your data
- `<parallel>` runs the creation of the pix2pix datasets, reconstruction of volumes from pix2pix outputs, reshaping of reconstructed volumes, and filtering of volumes using wavelet transform steps in parallel instead of in series (but it is computationally more expensive)
- `<ensemble>` creates outputs using 12 FlowGAN models instead of just 1 and then averages the final outputs to improve signal-to-noise ratio. This approach takes longer and is more computationally expensive, but it can be useful if the ASL CBF inputs are particularly challenging
- `<intermediates>` keeps the intermediate files generated by the pipeline and puts them in `<output_dir>/intermediates`. Normally, the intermediate files are removed, as they can take up a decent amount of space, but this can be useful for debugging


Congrats! Your final outputs should be in `<output_dir>/FlowGAN_outputs`

## Example

We have provided two example subjects in the `sample_data` directory, one from each cohort: `sub-MCI0289` (MCI) and `sub-RID0576` (TLE). Both are in the format specified above, which is necessary for the pipeline to work. We have also provided a corresponding file, `sample_list_of_subjects.txt`, which lists both subjects, in the format described above.

`--cohort` selects which trained model is applied to the subjects you run. To run FlowGAN on both sample subjects using the MCI model, you can run the following command (pass `--cohort TLE` instead to use the TLE model — FlowGAN generalizes across cohorts, so either model runs on both subjects):

`python run_FlowGAN.py --subs_file ../sample_list_of_subjects.txt --data ../sample_data/ --output_dir ../sample_data_outputs/ --cohort MCI`

For the closest match to each subject's own data, use the MCI model for `sub-MCI0289` and the TLE model for `sub-RID0576`; to do that, point `--subs_file` at a file listing just the subject you want for each run.

This command will run FlowGAN on the sample subjects and save the outputs in `sample_data_outputs`. The final FlowGAN outputs (one synthetic PET volume per subject) will be in `sample_data_outputs/FlowGAN_outputs`

Now that you have seen how it works, feel free to try it out with your data!


## Data availability

The raw imaging data (T1-weighted MRI, ASL, and FDG-PET) are not shared, as they are protected health information. The regional values extracted from those images — from which every figure, statistic, and table in the manuscript can be regenerated — are shared alongside the analysis code, together with a notebook that reproduces each reported value.

<!-- TODO: link the manuscript reproduction repository here once it is public. -->

## Citation

**FlowGAN**:

**Multimodal MRI-to-PET Image Translation Recovers Disease-Specific Hypometabolism in Epilepsy and Mild Cognitive Impairment**

Lucas A\*, Vadali C\*, Mouchtaris S, Arnold TC, Gugger JJ, Kulick C, Josyula M, Petillo N, Dolui S, Wolk D, Das S, Dubroff J, Stein JM, Detre JA\*, Davis KA\*.

\*These authors contributed equally.

_Communications Medicine_ (accepted, 2026)

<!-- TODO: add the link/DOI to the paper once it is published. -->

**LowGAN**, the architecture FlowGAN is based on:

**Multisequence 3-T Image Synthesis from 64-mT Low-Field-Strength MRI Using Generative Adversarial Networks in Multiple Sclerosis**

Lucas A, Arnold TC, Okar SV, Vadali C, Kawatra KD, Ren Z, Cao Q, Shinohara RT, Schindler MK, Davis KA, Litt B, Reich DS, Stein JM.

_Radiology_ (2025). [Link to paper](https://pubs.rsna.org/doi/10.1148/radiol.233529)

## License

This project is released under the MIT License — see [LICENSE](LICENSE).