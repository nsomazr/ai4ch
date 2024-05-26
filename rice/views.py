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
import requests
import tensorflow_hub as hub
from . models import RiceData
from . serializers import ImageSerializer
from django.shortcuts import render
from django.http import HttpResponse
from . forms import  ImageHorizontal
from torchvision import transforms
from keras.models import load_model
from torch.autograd import Variable
from email.message import EmailMessage
from email.mime.text import MIMEText
from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from io import BytesIO
from rest_framework.parsers import MultiPartParser, FormParser
from email.mime.multipart import MIMEMultipart
from keras.preprocessing.image import img_to_array
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

# Create your views here.

class PredictImageView(APIView):
    
    parser_classes = (MultiPartParser, FormParser)
    
    def get(self, request):
        return Response({'message':'This is rice prediction endpoint'})
    
    def post(self, request):
        
        serializer = ImageSerializer(data=request.data)
        # Validate the data

        if serializer.is_valid():
            # Access the image file
            image_file = serializer.validated_data['image']
            
            try:
                # Open the image file
                image = Image.open(image_file)
                
                # Resize and preprocess the image for classification
                image = image.resize((150, 150))
                image = np.array(image)  # Convert PIL image to numpy array
                image = image.astype("float") / 255.0  # Normalize pixel values
                image = np.expand_dims(image, axis=0)  # Add batch dimension
            
                # Load .h5 model
                model = load_model(os.path.join(BASE_DIR,'models/rice_inceptionV3.h5'))

                # Make prediction
                prediction = model.predict(image)[0]

                # Convert prediction to JSON format
                response_data = {
                    'Bacterial Blight': float(prediction[0]),
                    'Blast': float(prediction[1]),
                    'Browm Spot':float(prediction[2]),
                    'Tungro':float(prediction[3]),
                }

                return Response(response_data, status=200)
                 
            except Exception as e:
                # print(e)
                return Response({'error': 'Failed to download the image with error '}, status=400)
            
        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=400)

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

                if str(image_path.name).lower().endswith(".jpg") or str(image_path.name).endswith(".png") or str(image_path.name).endswith(".jpeg"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    image_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = RiceData(image_id=image_id, image_path=image_path, image_name=image_name)
                    # print("Saving Image")
                    new_file.save()

                    # import all import libraries

                    """load image, returns tensor"""
                    image_path=os.path.join(BASE_DIR,'media/images/'+str(image_path.name).replace(' ', '_'))
                    # print("Image path: ", image_path)
                    image = cv2.imread(image_path)

                    # pre-process the image for classification
                    image = cv2.resize(image, (150, 150))
                    image = image.astype("float") / 255.0
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    
                    since_time = time.time();
                    # load the saved model
                    loaded_model = load_model(os.path.join(BASE_DIR,'models/rice_inceptionV3.h5'))

                    # (angula_leaf_spot, bean_rust, healthy)
                    probabilities = loaded_model.predict(image)[0]
                    
                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    pred_proba = np.max(probabilities)
                    pred_index = np.argmax(probabilities)
                    
                    # labels dictionary
                    labels_dict =  {'Bacterial Blight': 0,
                                    'Blast': 1,
                                    'Browm Spot': 2,
                                    'Tungro': 3}
                    
                    pred_label = None
                    for class_name, class_index in labels_dict.items():
                        if class_index == pred_index:
                            if pred_proba >= 0.5:
                                pred_label = class_name    
                            else:
                                pred_label = 'Undetermined'  
    

                    time_elapse = time.time() - since_time
                    # print("Time elapse: ", time_elapse)
                    image_data = RiceData.objects.get(image_id=image_id)
                    # print("Image Details: ", image_data.image_path)
                    context = {'image_horizontal': image_horizontal,'prediction':pred_label, 'proba': pred_proba,
                    'pred_index': pred_index, 'probabilities': prob, 'image':image_data}
                    return render(request, 'classifiers/rice/rice-classification.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'image_horizontal': image_horizontal,'format_massage': format_message}

                    return render(request, 'classifiers/rice/rice-classification.html', context=context)

        else:
            return render(request, template_name="classifiers/rice/rice-classification.html", context=context)

    return render(request, template_name="classifiers/rice/rice-classification.html", context=context)