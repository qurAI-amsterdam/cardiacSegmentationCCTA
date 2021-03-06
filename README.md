# Automatic cardiac segmentation in CCTA
This repository contains code for training and inference of a 3D CNN for whole-heart segmentation in CCTA. Please cite the following paper if you use this code.

Steffen Bruns, Jelmer M Wolterink, Thomas PW van den Boogert, José P Henriques, Jan Baan, R Nils Planken, and Ivana Išgum. Automatic whole-heart segmentation in 4D TAVI treatment planning CT. In Medical Imaging 2021: Image Processing, volume 11596, page 115960B. International Society for Optics and Photonics, 2021. (https://doi.org/10.1117/12.2581020)

This project was financially supported by Philips Healthcare.

## Usage
All functionality is implemented in the python file cardiacSegmentationCCTA.py. It can perform either CNN training or inference.

### A) CNN training
For CNN training, CT images in .mhd format are required. Additionally, reference segmentations with the same size and format are required. In the reference segmentation, each voxel of class i should be assigned the value i (with 0 for background), e.g. 0: background, 1: left ventricular cavity, 2: right ventricle, 3: left atrium, 4: right atrium, 5: left ventricular myocardium (NCLASS=6). If CT images and reference segmentations are prepared, follow these steps to train a CNN:
1) Make sure the training and validation images and reference segmentations are located in the correct relative paths, e.g.
- code in: /home/user/Segmentation/src
- training images in: /home/user/Segmentation/mmwhs/fold_01/train/images
- training reference in: /home/user/Segmentation/mmwhs/fold_01/train/reference
- validation images in: /home/user/Segmentation/mmwhs/fold_01/validate/images
- validation reference in: /home/user/Segmentation/mmwhs/fold_01/validate/reference
2) Start the program from the command line and supply the arguments for the argument parser, e.g.

python cardiacSegmentationCCTA.py --mode train --train_dir mmwhs --lr 0.001 --lr_step_size 4000 --lr_gamma 0.3 --n_iterations 10000 --n_class 6 --vox_size 0.8 --batch_size 8 --fold 1 --rand_seed 123 --tag testCardiacSegmentationCCTA

(Some of the arguments are optional. Additional information on the specific arguments can be found in the code itself.)

### B) CNN inference
For CNN inference, CT images in .mhd format and a trained network are required. Follow these steps to perform CNN inference:
1) Make sure the CT images for inference and the trained network(s) are located in the correct relative paths, e.g.
- code in: /home/user/Segmentation/src
- CT images in: /home/user/Segmentation/mmwhs/fold_01/test/images
- trained network(s) in: /home/user/Segmentation/experiments/testCardiacSegmentationCCTA
2) Start the program from the command line and supply the arguments for the argument parser, e.g.

python cardiacSegmentationCCTA.py --mode test --trained_networks /home/user/Segmentation/experiments/testCardiacSegmentationCCTA/10000.pt --test_dir /home/user/Segmentation/mmwhs/fold_01/test/images --n_class 6 --vox_size 0.8

(Some of the arguments are optional. Additional information on the specific arguments can be found in the code itself.)
