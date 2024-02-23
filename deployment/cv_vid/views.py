
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
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    rgb = seg2rgb(output_predictions)
    rgb = np.array(rgb)
    return rgb


label_colors = np.array([(0, 0, 0),  # 0=background
              # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
              (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

def seg2rgb(preds):
    colors = label_colors
    colors = label_colors.astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    rgb = Image.fromarray(preds.byte().cpu().numpy())#.resize(preds.shape)