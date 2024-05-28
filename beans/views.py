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
from . models import BeansData
from django.shortcuts import render
from . serializers import ImageSerializer
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
from rest_framework import status
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from rest_framework.parsers import MultiPartParser, FormParser
from keras.preprocessing.image import img_to_array
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

class PredictImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        return Response({'message': 'This is beans prediction endpoint'})

    def post(self, request):
        serializer = ImageSerializer(data=request.data)

        # Validate the data
        if serializer.is_valid():
            # Access the uploaded image file
            image_file = serializer.validated_data['image']
            
            # Sanitize the filename
            original_filename = image_file.name
            sanitized_filename = re.sub(r'[ ()]', '_', original_filename)  # Remove spaces and replace braces with underscores
            
            # Save the file with the sanitized name temporarily
            temp_path = os.path.join('/tmp', sanitized_filename)
            with open(temp_path, 'wb+') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
            
            try:
                image = Image.open(temp_path)
                
                # Preprocessing the image
                transform = transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

                img_tensor = transform(image)
                img_tensor = img_tensor / 255  # normalization of pixels

                # Calculate mean and std for normalization
                train_mean = img_tensor.view(3, -1).mean(dim=1)
                train_std = img_tensor.view(3, -1).std(dim=1)
                transformer_input = transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(train_mean, train_std)
                ])

                # Transform and normalize the image
                image = transformer_input(image).float()
                image = Variable(image, requires_grad=True)

                # Load the saved model
                model_path = os.path.join(BASE_DIR, 'models/resnet_model.pth')
                loaded_model = torch.load(model_path, map_location='cpu')
                loaded_model.eval()

                # Predict the class probabilities
                with torch.no_grad():
                    output_single = loaded_model(image.unsqueeze(0))
                    output_single_probability = torch.softmax(output_single, dim=1)
                    probabilities = output_single_probability.numpy()[0]

                # Convert prediction to JSON format
                response_data = {
                    'Angular Leaf Spot': float(probabilities[0]),
                    'Anthracnose': float(probabilities[1]),
                    'Ascochyta Leaf Spot': float(probabilities[2]),
                    'BCMV and BCMNV': float(probabilities[3]),
                    'Bean Rust': float(probabilities[4]),
                    'Common Bacterial Blight': float(probabilities[5]),
                    'Mixed Infection': float(probabilities[6])
                }

                # Clean up the temporary file
                os.remove(temp_path)

                return Response(response_data, status=status.HTTP_200_OK)

            except Exception as e:
                print(e)
                # Clean up the temporary file in case of an error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return Response({'error': 'Failed to process the image with error: {}'.format(str(e))}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def classifier(request):

    image_horizontal = ImageHorizontal()
    context = {'image_horizontal': image_horizontal}

    if request.method == 'POST' and request.FILES.get('image_file'):

        image_horizontal = ImageHorizontal(request.POST, request.FILES)

        if image_horizontal.is_valid():
            image_path = request.FILES['image_file']
            image_name = str(image_path.name).split('.')[0]

            # Sanitize the filename by removing spaces and replacing braces with underscores
            image_name = re.sub(r'[ ()]', '_', image_name)

            if str(image_path.name).lower().endswith((".jpg", ".png", ".jpeg")):
                image_id = ''.join(random.choice(string.ascii_uppercase) for i in range(2)) + str(np.random.randint(1000000))
                new_file = BeansData(image_id=image_id, image_path=image_path, image_name=image_name)
                new_file.save()

                image = Image.open(image_path)
                transform = transforms.Compose([
                    transforms.PILToTensor()
                ])
                img_tensor = transform(image)
                img_tensor = img_tensor / 255  # normalization of pixels
                train_mean = img_tensor.reshape(3, -1).mean(axis=1)
                train_std = img_tensor.reshape(3, -1).std(axis=1)

                transformer_input = transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(train_mean, train_std)
                ])

                image = Image.open(image_path)
                image = transformer_input(image).float()
                image = Variable(image, requires_grad=True)

                since_time = time.time()
                loaded_model = torch.load(os.path.join(BASE_DIR, 'models/resnet_model.pth'), map_location='cpu')
                loaded_model.eval()

                output_single = loaded_model(image.view(1, 3, 512, 512))
                output_single_probability = torch.softmax(output_single, dim=1)
                prediction_proba, prediction = torch.max(output_single_probability, 1)

                labels_dict = {'angular_leaf_spot': 0,
                               'anthracnose': 1,
                               'ascochyta_leaf_spot': 2,
                               'bcmv_and_bcmnv': 3,
                               'bean_rust': 4,
                               'common_bacterial_blight': 5,
                               'mixed_infection': 6}
                probabilities = output_single_probability.detach().numpy()[0]

                prob = [float(i) for i in probabilities]
                initial_pred = ''
                pred = ''
                pred_index = 0
                for class_name, class_index in labels_dict.items():
                    if class_index == prediction:
                        initial_pred = class_name
                        if prediction_proba.item() >= 0.5:
                            pred = class_name
                            pred_index = class_index
                        else:
                            pred = 'Undetermined'

                time_elapse = time.time() - since_time
                image_data = BeansData.objects.get(image_id=image_id)
                context = {'image_horizontal': image_horizontal, 'prediction': pred, 'proba': prediction_proba.item(),
                           'pred_index': pred_index, 'probabilities': prob, 'image': image_data}
                return render(request, 'classifiers/beans/beans-classification.html', context=context)

            else:
                format_message = "Unsupported format, supported formats are .png and .jpg"
                context = {'image_horizontal': image_horizontal, 'format_message': format_message}
                return render(request, 'classifiers/beans/beans-classification.html', context=context)

        else:
            return render(request, template_name="classifiers/beans/beans-classification.html", context=context)

    return render(request, template_name="classifiers/beans/beans-classification.html", context=context)