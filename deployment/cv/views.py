
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

def base(request):
        
    return render(request, 'cv/base.html')        


def classification(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        # `img` is a PIL image of size 224x224
        img_file_ = settings.BASE_DIR + '/' + img_file
        img = image.load_img(img_file_, target_size=(224, 224))
        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        x = preprocess_input(x)
        model = VGG16(weights='imagenet')
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])
        pred = decode_predictions(preds, top=1)[0][0][1]