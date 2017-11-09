# Machine-Learning-in-Medical-Imaging--U-Net

[U-Net](https://arxiv.org/pdf/1505.04597.pdf) is a Convolutional Neural Network (CNN) that was designed for biological image segmentation. In order to preserve finer feature maps, skipping connections are used that complement the data in deeper layers. In this work, the same architecture was adapted for MRI brain scans to predict one modality given another. This was done by sclicing raw volumetric MRI data scanned in two different modalities into 2D images that the network could be trained on. The network was implemented using [matConvNet](https://github.com/vlfeat/matconvnet), a MATLAB toolbox for CNNs.

[Final Presentation](Presentation/Unet_presentation.pdf)


