from __future__ import print_function, division


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import pandas as pd
import torch.nn as nn
import torchvision
import os
import numpy as np
from tempfile import TemporaryFile
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from playsound import playsound
import cma

cap = cv2.VideoCapture(0)
 
if (cap.isOpened() == False):
	print("UNABLE TO OPEN CAMERA FEED.")
 			 
currentFrame = 0
while(currentFrame<=50):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    name = "frame%d.jpg"%currentFrame
    print ('Creating...' + name)
    cv2.imwrite(r'Frames/%s' %name, frame)
    currentFrame += 1

    cv2.imshow('frame',gray)
    if cv2.waitKey(10) != -1:
        break
 

cap.release()
cv2.destroyAllWindows()

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')



n_layers= 6
KERNEL = np.zeros([1, n_layers ], dtype=np.uint32)
PADDING = np.zeros([1, n_layers], dtype=np.uint32)
N_H = np.zeros([1, n_layers+1], dtype=np.uint32)
filter_arr = np.zeros([n_layers])

#parameters = np.array([2.96950262, 0.34795797, 3.64607391, 4.18699879, 4.17248074, 0.97604556])  #best with seed from report1
parameters =  np.array([2.09399337, -1.0827538 ,  1.58994055,  2.90343432,  3.5351136, 1.38136274]) 
#parameters =  np.array([ 9.1293918 , -3.70825475,  2.80793968,  6.35230729, -2.66111807, 01.00730026])
  
        
for idx in range(n_layers):

    if parameters[idx] < 0: filter_arr[idx] = 0

    elif parameters[idx] >= 0   and parameters[idx] <= 0.54: filter_arr[idx] = 0
    elif parameters[idx] > 0.54 and parameters[idx] <= 1.54: filter_arr[idx] = 1
    elif parameters[idx] > 1.54 and parameters[idx] <= 2.54: filter_arr[idx] = 2
    elif parameters[idx] > 2.54 and parameters[idx] <= 3.54: filter_arr[idx] = 3
    elif parameters[idx] > 3.54 and parameters[idx] <= 4.54: filter_arr[idx] = 4
    elif parameters[idx] > 4.54                            : filter_arr[idx] = 5

filter1 = filter_arr[0]
filter2 = filter_arr[1]
filter3 = filter_arr[2]
filter4 = filter_arr[3]
filter5 = filter_arr[4]
filter6 = filter_arr[5]
                    

kernel_array = np.array([filter1, filter2, filter3, filter4, filter5, filter6], dtype=np.uint32)
transposed_count = np.zeros([1,n_layers], dtype=np.uint32)
padding_array =np.zeros([1,n_layers], dtype=np.uint32)
channel_size = np.zeros([1,n_layers+1], dtype = np.uint32)
n_H_array = np.zeros([1,n_layers+1], dtype=np.uint32)
n_H_array[0,0] = 224
channel_size[0,0] = 1
idd = 0  

for i in range(n_layers):


    if i <= 4:  #First 5 Layers include MaxPool and n_H_array has to be divided by 2           
        if kernel_array[i] == 0:   #Identity Layer
            padding_array[0,i] = 0   #Padding = 0 for Identity Layer
            n_H_array[0,i+1] = np.floor((n_H_array[0,i]))
            channel_size[0,i+1] = channel_size[0,i]

        elif kernel_array[i] > 0 and kernel_array[i] <= 5:
            idd+=1
            kernel_array[i] = 2*kernel_array[i] - 1  #f=1=>1DConv, f=2=>3DConv, f=3=>5DConv, f=4=>7DConv, f=5=>9DConv   
            padding_array[0,i] = (kernel_array[i] - 1)/2  
            n_H_array[0,i+1] = np.floor((n_H_array[0,i] - kernel_array[i] + 2*padding_array[0,i] + 1)/2) 
            channel_size[0,i+1] = 8*(2**idd)

    elif i == 5:   #Last layer doesn't have MaxPool 
        if kernel_array[i] == 0:   #Identity Layer
            padding_array[0,i] = 0   #Padding = 0 for Identity Layer
            n_H_array[0,i+1] = np.floor((n_H_array[0,i]))
            channel_size[0,i+1] = channel_size[0,i]

        elif kernel_array[i] > 0 and kernel_array[i] <= 5:
            idd+=1
            kernel_array[i] = 2*kernel_array[i] - 1    #i=1=>1DConv, i=2=>3DConv, i=3=>5DConv, i=4=>7DConv, i=5=>9DConv   
            padding_array[0,i] = (kernel_array[i] - 1)/2  
            n_H_array[0,i+1] = np.floor(n_H_array[0,i] - kernel_array[i] + 2*padding_array[0,i] + 1) 
            channel_size[0,i+1] = 8*(2**idd)

            

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()


        # First Layer: (Check for Identity-Layer, or Convolution Layer)
        if kernel_array[0] == 0:
            self.layer1 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[0] > 0:      
            self.layer1 = nn.Sequential(
                nn.Conv2d(channel_size[0,0], channel_size[0,1], 
                        kernel_size=kernel_array[0], stride=1, padding=padding_array[0,0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) 


        # Second Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)    
        if kernel_array[1] == 0:
            self.layer2 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[1] > 0:      
            self.layer2 = nn.Sequential(
                nn.Conv2d(channel_size[0,1], channel_size[0,2], 
                        kernel_size=kernel_array[1], stride=1, padding=padding_array[0,1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) 
        
            
        # Third Layer: (Check for Identity-Layer, or Convolution Layer)
        if kernel_array[2] == 0:
            self.layer3 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[2] > 0:      
            self.layer3 = nn.Sequential(
                nn.Conv2d(channel_size[0,2], channel_size[0,3], 
                        kernel_size=kernel_array[2], stride=1, padding=padding_array[0,2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) 


        # Fourth Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)
        if kernel_array[3] == 0:
            self.layer4 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[3] > 0:     
            self.layer4 = nn.Sequential(
                nn.Conv2d(channel_size[0,3], channel_size[0,4], 
                        kernel_size=kernel_array[3], stride=1, padding=padding_array[0,3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) 


        # Fifth Layer: (Check for Identity-Layer, or Convolution Layer)
        if kernel_array[4] == 0:
            self.layer5 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[4] > 0:      
            self.layer5 = nn.Sequential(
                nn.Conv2d(channel_size[0,4], channel_size[0,5], 
                        kernel_size=kernel_array[4], stride=1, padding=padding_array[0,4]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) 


        # Sixth Layer: (Check for Identity-Layer, or Convolution Layer)
        if kernel_array[5] == 0:
            self.layer6 = nn.Sequential(
                nn.Identity(),
                nn.ReLU())

        elif kernel_array[5] > 0:      
            self.layer6 = nn.Sequential(
                nn.Conv2d(channel_size[0,5], channel_size[0,6], 
                        kernel_size=kernel_array[5], stride=1, padding=padding_array[0,5]),
                nn.ReLU()) 

        self.fc1 = nn.Linear(channel_size[0,6]*n_H_array[0,6]*n_H_array[0,6], 1024)
        self.fc2 = nn.Linear(1024, 68 * 2) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return  out

model = ConvNet()
model.load_state_dict(torch.load('./Best_Landmark_Setection.pt', map_location=torch.device('cpu')))
#model.eval()


def showpoints(image,keypoints):

    plt.figure()
    
    keypoints = keypoints.data.numpy()
    keypoints = keypoints * 54 + 80
    keypoints = np.reshape(keypoints, (68, -1))
    
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, marker='.', c='r')
    plt.show()

def plot_full_image_points(img, pts, x, y):

    plt.figure()
    pts = pts.data.numpy()
    pts = pts * 54 + 80
    pts = np.reshape(pts, (68, -1))
    pts[:, 0] = pts[:, 0]+x
    pts[:, 1] = pts[:, 1]+y 
    
    plt.imshow(img, cmap='gray')
    plt.scatter(pts[:, 0], pts[:, 1], s=50, marker='.', c='r')
    plt.show()

def get_points_from_model(pts, x, y):

    points = pts.data.numpy()
    points = points * 54 + 80
    points = np.reshape(points, (68, -1))
    pts[:, 0] = pts[:, 0]+x
    pts[:, 1] = pts[:, 1]+y 
    return points    
    

n_images = 5
# load in color image for face detection
for i in range(n_images):
        
    image = cv2.imread('Frames/frame' + str(i*10) + '.jpg')

    # switch red and blue color channels 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)

    # plot the image
    fig1 = plt.figure(figsize=(9,9))
    plt.imshow(image)
    plt.show()

    # run the Haar Cascade detector
    faces = face_cascade.detectMultiScale(image, 1.2, 2)
    print(faces.size)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

    fig2 = plt.figure(figsize=(9,9))

    plt.imshow(image_with_detections)
    plt.show()

    image_copy = image.copy()
    image_copy_helmet = image.copy()

    # loop over the detected faces from your haar cascade
    for (x,y,w,h) in faces:
        
        # Select the region of interest that is the face in the image 
        img_2_train = image_copy[y:y+h,x:x+w]

        ##Convert the face region from RGB to grayscale
        img_2_train = cv2.cvtColor(img_2_train, cv2.COLOR_RGB2GRAY)
        image_2_plot = img_2_train

        ##Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        img_2_train = img_2_train/255.0
        
        ##Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        img_2_train = cv2.resize(img_2_train, (224,224))
        
        ##Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        img_2_train = np.expand_dims(img_2_train, 0)
        img_2_train = np.expand_dims(img_2_train, 0)
        
        ##Make facial keypoint predictions using your loaded, trained network 
        img_2_train_torch = Variable(torch.from_numpy(img_2_train))
        
        img_2_train_torch = img_2_train_torch.type(torch.FloatTensor)
        keypoints = model(img_2_train_torch)
        
        showpoints(image_2_plot,keypoints)

        # load in sunglasses image with cv2 and IMREAD_UNCHANGED
        sunglasses = cv2.imread('images/sunglasses' + str(i) + '.png', cv2.IMREAD_UNCHANGED)
        # plot our image        keypoints_fixed[:,1] = keypoints[:,1]
        fig4 = plt.figure()
        plt.imshow(sunglasses)
        plt.show()        

        plot_full_image_points(image, keypoints, x, y)

        key_pts = get_points_from_model(keypoints, x, y)

        xp = int(key_pts[17, 0])+x-20
        yp = int(key_pts[17, 1])+y-20

        # height and width of sunglasses
        # h = length of nose
        hp = int(abs(key_pts[27,1] - key_pts[34,1]))+51
        # w = left to right eyebrow edges
        wp = int(abs(key_pts[17,0] - key_pts[26,0]))+51

        # read in sunglasses
        #sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
        # resize sunglasses

        new_sunglasses =  cv2.resize(sunglasses, (wp, hp), interpolation = cv2.INTER_CUBIC)

        # get region of interest on the face to change
        eyes_location_image = image_copy[yp:yp+hp,xp:xp+wp]

        # find all non-transparent pts
        ind = np.argwhere(new_sunglasses[:,:,3] > 0)

        # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
        for j in range(3):
            eyes_location_image[ind[:,0],ind[:,1],j] = new_sunglasses[ind[:,0],ind[:,1],j]    
        # set the area of the image to the changed region with sunglasses
        image_copy[yp:yp+hp,xp:xp+wp] = eyes_location_image

        fig6 = plt.figure()
        # display the result!
        plt.imshow(image_copy)
        plt.show()


        #####################################--Time to recognize the head--####################################
        xh = int(key_pts[0, 0])+x-68
        yh = int(key_pts[19, 1])+y-159

        # height and width of helmet
        # h = length head
        hh = int(abs(key_pts[19,1] - key_pts[8,1]))*2+80
        # w = left to right of head
        wh = int(abs(key_pts[0,0] - key_pts[16,0]))+160

        helmet = cv2.imread('images/helmet' + str(i) + '.png', cv2.IMREAD_UNCHANGED)

        new_helmet =  cv2.resize(helmet, (wh, hh), interpolation = cv2.INTER_CUBIC)

        # get region of interest on the face to change
        head_location_image = image_copy_helmet[yh:yh+hh,xh:xh+wh]

        # find all non-transparent pts
        ind_head = np.argwhere(new_helmet[:,:,3] > 0)

        # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
        for k in range(3):
            head_location_image[ind_head[:,0],ind_head[:,1],k] = new_helmet[ind_head[:,0],ind_head[:,1],k]    
        # set the area of the image to the changed region with sunglasses
        image_copy_helmet[yh:yh+hh,xh:xh+wh] = head_location_image

        fig77 = plt.figure()
        # display the result!
        plt.imshow(image_copy_helmet)
        plt.show()


    


