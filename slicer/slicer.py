#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as itk
from PIL import Image
from skimage import exposure



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



def slice3DObject(image_list, image3d_list, folder_name, extract_range):

    counter = 1

    for c in range(len(image3d_list)):

        img_name = image_list[c]
        img3d = image3d_list[c]

        print('Processing image number %3.0i: %1s' % (counter, img_name))
        counter = counter + 1

        arr3d = itk.GetArrayFromImage(img3d)     # Get np.array from 3D Image

        arr3d = normalize(arr3d)                # Normalize image data

        number = extract_range[0]
        for im2d in arr3d[extract_range[0]:-extract_range[1]]:

            #im2d = normalize(im2d)               # Normalize image data

            file_name = folder_name + '/' + img_name + '_layer' + str(number) + '.png'

            saveArrayAsImage(im2d[::-1], file_name)

            number = number + 1



def normalize(arr):
    arr *= 280.0/arr.max()

    # Contrast stretching
    #p2, p98 = np.percentile(arr, (0, 99.5))
    #arr = exposure.rescale_intensity(arr, in_range=(p2, p98))
    #arr *= 255.0

    return arr



def slice_all(batch_size, folder_name, extract_range):

    image_list, n = organize_read_in(batch_size)                # n: number of batches
    print('Number of batches: ', int(n))

    if not os.path.exists(folder_name):                         # Create directory
        os.mkdir(folder_name)

    while 1:
        image_list, image3d_list = read3DImages(image_list, batch_size)    # Get 3D Objects from zipped files

        slice3DObject(image_list, image3d_list, folder_name, extract_range)    # extract 2D images from 3D object and save 

        image_list[:batch_size] = []                            # remove loaded batch from name list

        if image_list == []:
            print('*** Done ***')
            break



def slice_one(batch_size=1):
    save_dir = '2DImg_test_dir'

    if not os.path.exists(save_dir):                 # Create directory
        os.mkdir(save_dir)    

    image_list, n = organize_read_in(batch_size)       # Get list of images in folder

    im3d_name, im3d = read3DImages([image_list[0]], batch_size)    # Get 3D Objects from zipped files

    arr3d = itk.GetArrayFromImage(im3d[0])             # get numpy array of shape (256,256,150)

    print('Befor Normaization, MIN Value of array: ', np.amin(arr3d))
    print('Befor Normaization, MAX Value of array: ', np.amax(arr3d))

    arr3d = normalize(arr3d)

    print('After Normaization, MIN Value of array: ', np.amin(arr3d))
    print('After Normaization, MAX Value of array: ', np.amax(arr3d))

    im2d = arr3d[100,:,:]                 # get slice(x,y) of 3d object

    saveArrayAsImage(im2d, save_dir+'/test_file.png')



def main():
    mode = 0                    # Switch mode: (0) process ALL images in folder
                                #              (1) process only ONE image from folder
    batch_size = 2             # Number of images processed at once (loaded into the RAM)
    folder_name = '2d_images'   # Define directory name for all slices
    extract_range = [50, 50]    # Don't extract first 20 and last 10 images

    if mode == 0:
        slice_all(batch_size, folder_name, extract_range)
    elif mode == 1:
        slice_one()



if __name__ == "__main__":    # If run as a script, create a test object
    main()
    #plt.show() 
