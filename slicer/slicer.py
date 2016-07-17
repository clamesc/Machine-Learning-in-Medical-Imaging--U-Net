#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as itk
from PIL import Image
from skimage import exposure
import pylab as P


""" 
Daten Check:    - Anzahl insg 79800k (798 datein)
                - Anzahl / Ordner
                - Auflösung
                - Kanäle

"""



def organize_read_in(batch_size):
    image_list = []

    for img_name in os.listdir():

        if img_name[-7:] == '.nii.gz':

            image_list.append(img_name)

    number_of_batches = len(image_list)/batch_size

    return image_list, number_of_batches



def read3DImages(image_list, batch_size):

    print('Reading files ...')

    image3d_list = []

    for img_name in image_list[:batch_size]:

        image3d = itk.ReadImage(img_name)    # get 3D Image from file

        image3d_list.append(image3d)

    return image_list, image3d_list



def saveArrayAsImage(array, file_name):

    img = Image.fromarray(array)    # create image from np.array

    img = img.convert('RGB')    # convert image to RGB

    img.save(file_name)



def slice3DObject(image_list, image3d_list, folder_name, extract_range, newSize):

    counter = 1

    for c in range(len(image3d_list)):           # for all images in image3d_list

        img_name = image_list[c]
        img3d = image3d_list[c]

        print('Processing image number %3.0i: %1s' % (counter, img_name))
        counter = counter + 1

        arr3d = itk.GetArrayFromImage(img3d)     # Get np.array from 3D Image

        if img_name[-8] == '1':
            arr3d = normalize(arr3d)             # Normalize image data
            subfolderDir = 'T1'
        elif img_name[-8] == '2':
            subfolderDir = 'T2'
        else:
            subfolderDir = "error"

        number = extract_range[0]
        for im2d in arr3d[extract_range[0]:-extract_range[1]]:

            #im2d = normalize(im2d)               # Normalize image data

            file_name = folder_name + '/' + subfolderDir + '/' + img_name + '_layer' + str(number) + '.jpg'

            if subfolderDir == "T1":
                im2d = adjust_size(im2d, newSize)     # Resize Image

            saveArrayAsImage(im2d[::-1], file_name)

            number = number + 1



def show_histogram(img):
    # the histogram of the data with histtype='step'
    n, bins, patches = P.hist(img, 50, normed=1, histtype='stepfilled')
    P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    P.title("Histogram")
    P.xlabel("x")
    P.ylabel("Relative Häufigkeit")



def adjust_size(arr, newSize):
    arrNew = np.zeros((newSize[0], newSize[1]))

    diffX = abs(arr.shape[0] - newSize[0]) / 2    # claculate position of upper left image point
    diffY = abs(arr.shape[1] - newSize[1]) / 2    # of arr in arrNew

    arrNew[diffX : diffX+arr.shape[0], diffY : diffY+arr.shape[1]] = arr

    return arrNew





def normalize(arr):
    mode = 1

    if mode == 0:
        arr /= arr.max()
        arr *= 265
        arr -= 5

    elif mode == 1:
    # Contrast stretching (1)
        p2, p98 = np.percentile(arr, (30, 99.5))
        arr = exposure.rescale_intensity(arr, in_range=(p2, p98))
        arr *= 255.0

    elif mode == 2:
    # Histogram Equalization (2)
        arr = exposure.equalize_hist(arr)
        arr *= 255

    elif mode == 3:
    # Adaptive Equalization (3)
        arr /= arr.max()
        arr = exposure.equalize_adapthist(arr)

    return arr



def slice_all(batch_size, folder_name, extract_range, newSize):

    image_list, n = organize_read_in(batch_size)                # n: number of batches
    print('Number of batches: ', int(n))

    if not os.path.exists(folder_name):                         # Create directory
        os.mkdir(folder_name)
        os.mkdir(folder_name + '/T1')
        os.mkdir(folder_name + '/T2')
        os.mkdir(folder_name + '/error')

    while 1:
        image_list, image3d_list = read3DImages(image_list, batch_size)    # Get 3D Objects from zipped files

        slice3DObject(image_list, image3d_list, folder_name, extract_range, newSize)    # extract 2D images from 3D object and save 

        image_list[:batch_size] = []                            # remove loaded batch from name list

        if image_list == []:
            print('*** Done ***')
            break



def slice_one(newSize):
    showHistogram=False
    batch_size=1

    save_dir = '2DImg_test_dir'

    if not os.path.exists(save_dir):                    # Create directory
        os.mkdir(save_dir)


    #im3d_name, im3d = read3DImages([image_list[0]], batch_size)    # Get 3D Objects from zipped files
    im3d_name, im3d = read3DImages(['R_IXI193-Guys-0810-T1.nii.gz'],1)


    arr3d = itk.GetArrayFromImage(im3d[0])             # get numpy array of shape (256,256,150)

    print('Befor Normaization, MIN Value of array: ', np.amin(arr3d))
    print('Befor Normaization, MAX Value of array: ', np.amax(arr3d))

    im2d_original = arr3d[100,:,:]        # get slice(x,y) of 3d object
    saveArrayAsImage(im2d_original[::-1], save_dir+'/test_file_orig.png')    # save original image

    arr3d = normalize(arr3d)              # Normalize Image

    print('After Normaization, MIN Value of array: ', np.amin(arr3d))
    print('After Normaization, MAX Value of array: ', np.amax(arr3d))

    im2d = arr3d[100,:,:]                 # get slice(x,y) of 3d object

    im2d = im2d[::-1]                     # rotate image

    im2d = adjust_size(im2d, newSize)


    if showHistogram is True:
        show_histogram(im2d)              # show the histogram of an image

    saveArrayAsImage(im2d, save_dir+'/test_file.png')



def main():
    mode = 0                      # Switch mode: (0) process ALL images in folder
                                  #              (1) process only ONE 3D-image from folder
    batch_size = 10                # Number of images processed at once (loaded into the RAM)
    folder_name = '2d_images_2'     # Define directory name for all slices
    extract_range = [30, 20]      # Don't extract first 20 and last 10 images
    newSize = [372, 372]          # resize imgaes to (newSize[0] x newSize[1])


    if mode == 0:
        slice_all(batch_size, folder_name, extract_range, newSize)
    elif mode == 1:
        slice_one(newSize)



if __name__ == "__main__":    # If run as a script, create a test object
    main()
    plt.show() 




