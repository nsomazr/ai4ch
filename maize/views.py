import re
import cv2
import time
import os
import string
import random
import torch
import warnings
import numpy as np
from torch import nn
from PIL import Image
import smtplib, ssl
from . models import MaizeData
from django.shortcuts import render
from django.http import HttpResponse
from . forms import  ImageHorizontal
from torchvision import transforms
from tensorflow.keras.models import load_model
from torch.autograd import Variable
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tensorflow.keras.preprocessing.image import img_to_array
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

# Create your views here.

def classifier(request):

    image_horizontal = ImageHorizontal()

    context = {'image_horizontal': image_horizontal}

    if request.method == 'POST' and request.FILES['image_file']:

        image_horizontal = ImageHorizontal(request.POST, request.FILES)

        if image_horizontal.is_valid():

                image_path = request.FILES['image_file']

                image_name = str(image_path.name).split('.')[0]

                # print('Image name: ', image_name)

                image_name = str(image_name).replace(' ', '_')

                if str(image_path.name).lower().endswith(".jpg") or str(image_path.name).endswith(".png"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    image_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = MaizeData(image_id=image_id, image_path=image_path, image_name=image_name)
                    # print("Saving Image")
                    new_file.save()

                    # import all import libraries

                    """load image, returns tensor"""
                    image_path=os.path.join(BASE_DIR,'media/images/'+image_path.name)
                    # print("Image path: ", image_path)
                    image = cv2.imread(image_path)

                    # pre-process the image for classification
                    image = cv2.resize(image, (244, 244))
                    image = image.astype("float") / 255.0
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    
                    since_time = time.time();
                    # load the saved model
                    loaded_model = load_model(os.path.join(BASE_DIR,'models/maize.h5'))

                    probabilities = loaded_model.predict(image)[0]
                    
                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    pred_proba = np.max(probabilities)
                    pred_index = np.argmax(probabilities)
                    
                    # labels dictionary
                    labels_dict = {'Cercospora Leaf Gray Leaf Spot': 0,
                                    'Common Rust': 1,
                                    'Northern Leaf Blight': 2,
                                    'Healthy': 3}
                    
                    pred_label = None
                    for class_name, class_index in labels_dict.items():
                        if class_index == pred_index:
                            if pred_proba >= 0.5:
                                pred_label = class_name    
                            else:
                                pred_label = 'Undetermined'  
    

                    time_elapse = time.time() - since_time
                    # print("Time elapse: ", time_elapse)
                    image_data = MaizeData.objects.get(image_id=image_id)
                    # print("Image Details: ", image_data.image_path)
                    context = {'image_horizontal': image_horizontal,'prediction':pred_label, 'proba': pred_proba,
                    'pred_index': pred_index, 'probabilities': prob, 'image':image_data}
                    return render(request, 'classifiers/maize/maize-classification.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'image_horizontal': image_horizontal,'format_massage': format_message}

                    return render(request, 'classifiers/maize/maize-classification.html', context=context)

        else:
            return render(request, template_name="classifiers/maize/maize-classification.html", context=context)

    return render(request, template_name="classifiers/maize/maize-classification.html", context=context)