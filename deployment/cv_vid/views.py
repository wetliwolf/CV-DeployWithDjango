
from django.shortcuts import render, redirect

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def base(request):
        
    return render(request, 'cv_vid/base.html')        
     

def load_model():
    # load the model for inference 
    model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    return model

def get_segmentation(input_image, model):
    #input_image = Image.open(img_file)