import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from face_data import load_faces

#load pre-trained classifier checkpoint
import neuralnet
neuralnet = reload(neuralnet)
from neuralnet import FeatureClassifier
model = FeatureClassifier()

#import classification helper functions
import classification
classification = reload(classification)
from classification import classify

#load and show an image
im = cv2.imread('faces/face_85.png', 0)
plt.imshow(im, cmap=plt.cm.gray)
plt.show()

# set up window parameters
window_size = 100
shift_size = 25

# scale the strokes
scale_factor = 8

# how many faces do you want to run?
num_faces = 10

for face in load_faces(n=num_faces):
    im = cv2.imread(face, 0)
    
    # documentation is in classification.py
    # thresh is minimum confidence
    # eyes, noses, mouths take the confident bounding boxes
    # verbose shows each window classification and accuracy
    classify(im,
             window_size=window_size,
             shift_size=shift_size,
             scale_factor=scale_factor,
             thresh=0.5,
             eyes=2, noses=1, mouths=1,
             verbose=True)
