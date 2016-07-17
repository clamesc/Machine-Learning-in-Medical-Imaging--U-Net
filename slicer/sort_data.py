#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
from shutil import copyfile
import PIL
from PIL import Image, ImageOps
from skimage import data, io, external
from scipy import misc


""" Read jpg files """
def read_data():
    image_list = []

    for img_name in os.listdir():

        if img_name[-4:] == '.jpg':

            image_list.append(img_name)

    return image_list



def copy_data(imageList, saveDir):

    if not os.path.exists(saveDir):             # Create directory
        os.mkdir(saveDir)

    for img in imageList:

        if img[-8] == "2":
            copyfile(img, saveDir + "/" + img)      # Copy data to new dir

    print("Copy Data Done!")



def sort_data(imageListT1, imageListT2, scrDirT1, scrDirT2, parts):
    imgSet = set()

    # get data names in set()
    for imgName in imageListT1:
        imgSet.add(imgName.split(".nii")[0][:-2])

    # devide data names into desired parts
    trainLen = int(len(imgSet)*parts[0])
    testLen = int(len(imgSet)*parts[1])
    valLen = len(imgSet) - trainLen - testLen

    # create new Sets for training, testing and validation
    trainSet = set()
    testSet = set()
    valSet = set()
    imgSet = list(imgSet)
    random.shuffle(imgSet)
    for i in range(trainLen):
        trainSet.add(imgSet.pop())
    for i in range(testLen):
        testSet.add(imgSet.pop())
    for i in range(valLen):
        valSet.add(imgSet.pop())


    # create new directories
    trainDir = "/Users/Leonard/Desktop/2d_images/test_set"
    testDir = "/Users/Leonard/Desktop/2d_images/train_set"
    valDir = "/Users/Leonard/Desktop/2d_images/val_set"
    if not os.path.exists(trainDir + "/T1"):                  
        os.mkdir(trainDir + "/T1")
    if not os.path.exists(testDir + "/T1"):                         
        os.mkdir(testDir + "/T1")    
    if not os.path.exists(valDir + "/T1"):                        
        os.mkdir(valDir + "/T1")
    if not os.path.exists(trainDir + "/T2"):                  
        os.mkdir(trainDir + "/T2")
    if not os.path.exists(testDir + "/T2"):                         
        os.mkdir(testDir + "/T2")    
    if not os.path.exists(valDir + "/T2"):                        
        os.mkdir(valDir + "/T2")


    # copy images to new directories
    for imgName in imageListT1:
        if imgName.split(".nii")[0][:-2] in trainSet:
            copyfile(scrDirT1 + "/" + imgName, trainDir + "/T1/" + imgName) 
        elif imgName.split(".nii")[0][:-2] in testSet:
            copyfile(scrDirT1 + "/" + imgName, testDir + "/T1/" + imgName) 
        else:
            copyfile(scrDirT1 + "/" + imgName, valDir + "/T1/" + imgName) 

    for imgName in imageListT2:
        if imgName.split(".nii")[0][:-2] in trainSet:
            copyfile(scrDirT2 + "/" + imgName, trainDir + "/T2/" + imgName) 
        elif imgName.split(".nii")[0][:-2] in testSet:
            copyfile(scrDirT2 + "/" + imgName, testDir + "/T2/" + imgName) 
        else:
            copyfile(scrDirT2 + "/" + imgName, valDir + "/T2/" + imgName) 
                    


def adjust_size(arr, newSize):
    """ Only for squared images """

    # Patch 0s around image to get new size
    if newSize > arr.shape[0]:
        arrNew = np.zeros((newSize, newSize))

        diff = int( (newSize - arr.shape[0]) / 2 )   # claculate position of upper left image point

        arrNew[diff : diff+arr.shape[0], diff : diff+arr.shape[1]] = arr

        return arrNew


    # Crop images to new size
    elif newSize < arr.shape[0]:

        diff = int((arr.shape[0] - newSize) / 2)

        return arr[diff:arr.shape[0]-diff, diff:arr.shape[1]-diff]

    else:
        return arr





def resize_images(imageList, newSize):

    for imgName in imageList:

        #image = Image.open(imgName)
        #ImageOps.expand(image, border=10, fill=0)
        #image.save("new_" + imgName)

        #img = data.imread(imgName)
        #img = adjust_size(img[:,:,0], newSize)
        #io.imsave("new_" + imgName, int(img))

        img = misc.imread(imgName)

        if len(img.shape) == 3:
            img = adjust_size(img[:,:,0], newSize)
        else:
            img = adjust_size(img, newSize)
        misc.imsave(imgName, img)



def main():
    """
    mode 0: Copy data
    mode 1: Sort data
    mode 2: Resize data

    """
    mode = int(sys.argv[1])



    if mode is 0:
        """ Copy certain files """
        print("*** Copy Mode ***")
        dataDir = "/Users/Leonard/Desktop/T2_data_3d"

        imageList = read_data(dataDir)
        copy_data(imageList)


    elif mode is 1:
        """ Sort data dependent on their name into different folders """
        print("*** Sort Mode ***")
        dataDirT1 = "T1"
        dataDirT2 = "T2"
        parts = [.6, .2, .2]    # procentual parts for [Train, Test, Val]

        imageListT1 = read_data(dataDirT1)
        imageListT2 = read_data(dataDirT2)
        sort_data(imageListT1, imageListT2, dataDirT1, dataDirT2, parts)


    elif mode is 2:
        """ Resize Images (replace images) """
        newSize = 252    # T1: 340,  T2: 252

        print("*** Resize Mode (size: %3.0i) ***" % (newSize))
        imageList = read_data()
        resize_images(imageList, newSize)
        print("*** Done Resizing ***")


    else:
        print("Wrong Mode !")






if __name__ == "__main__":    # If run as a script, create a test object
    main()
