import argparse
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from PIL import Image

import os
import random

import json

from utils import load_checkpoint, load_cat_names
from torch.autograd import Variable


def parse_args()
    parser = argparse.ArgumentParser(description='Prediction: Predict flower name from an image along with the probability of that name.')
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth', help='loads checkpoint')
    parser.add_argument('--top_k 3', dest="top_k", default="3", help='returns the most likely classes, 3 is default')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store', default='gpu', help='use GPU for inference as default')
    parser.add_argument('--file', dest='file', default='flowers/test/1/image_06754.jpg') 
    return parser.parse_args()

args = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    size = [0, 0]

    if image.size[0] > image.size[1]:
        size = [image.size[0], 256]
    else:
        size = [256, image.size[1]]
    
    image.thumbnail(size, Image.ANTIALIAS)
    
    w, h = image.size  

    l = (256 - 224)/2
    t = (256 - 224)/2
    r = (256 + 224)/2
    b = (256 + 224)/2

    image = image.crop((l, t, r, b))
    image = np.array(image)
    image = image/255.
                       
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
                       
    image = ((image - mean) / sd)
    
    image = np.transpose(image, (2, 0, 1))

    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    image = process_image(img)
    data = torch.from_numpy(image).type(torch.FloatTensor)
    data = data.unsqueeze(0)
    data = data.to('cuda')
    output = model.forward(data)
    
    ps = torch.exp(output).data
    probs, probs_labels = ps.topk(topk)
    prob = probs.cpu().numpy()[0]
    probs_labels = probs_labels.cpu().numpy()
    classes_list = list()
    classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
    for label in probs_labels[0]:
        classes_list.append(cat_to_name[classes_indexed[label]])
    return prob, classes_list


def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    image_path = args.file
    probs, classes = predict(img, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(probs)
    print(classes)
    
    i=0
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 







