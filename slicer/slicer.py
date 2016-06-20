#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as itk
from PIL import Image


def organize_read_in(batch_size):
	image_list = []

	for	img_name in os.listdir():

		if img_name[-7:] == '.nii.gz':

			image_list.append(img_name)

	number_of_batches = len(image_list)/batch_size

	return image_list, number_of_batches




def read3DImages(image_list, batch_size):

	print('Reading files ...')

	image3d_list = []

	for	img_name in image_list[:batch_size]:

		image3d = itk.ReadImage(img_name)    # get 3D Image from file

		image3d_list.append(image3d)

	return image_list, image3d_list



def saveArrayAsImage(array, file_name):

	img = Image.fromarray(array)    # create image from np.array

	img = img.convert('RGB')    # convert image to RGB

	img.save(file_name)



def slice3DObject(image_list, image3d_list, folder_name):

	counter = 1

	for c in range(len(image3d_list)):

		img_name = image_list[c]
		img3d = image3d_list[c]

		print('Processing image number %3.0i: %1s' % (counter, img_name))
		counter = counter + 1

		array3d = itk.GetArrayFromImage(img3d)

		for z in range(array3d.shape[0]):   	# Get 150 slices of size 256x256

			file_name = folder_name + '/' + img_name + '_layer' + str(z) + '.png'

			im2d = array3d[z,:,:]    # get slice(x,y) of 3d object

			saveArrayAsImage(im2d, file_name)




def main():

	batch_size = 10    # Number of images processed at once (loaded into the RAM)
	folder_name = '2d_images'    # Define directory name for all slices


	image_list, n = organize_read_in(batch_size)    # n: number of batches
	print('Number of batches: ', int(n))

	if not os.path.exists(folder_name):    # Create directory
		os.mkdir(folder_name)

	while 1:
		image_list, image3d_list = read3DImages(image_list, batch_size)    # Get 3D Objects from zipped files

		slice3DObject(image_list, image3d_list, folder_name)    # extract 2D images from 3D object and save 

		image_list[:batch_size] = []    # remove loaded batch from name list

		if image_list == []:
			print('*** Done ***')
			break



if __name__ == "__main__":    # If run as a script, create a test object
	main()
	#plt.show() 
