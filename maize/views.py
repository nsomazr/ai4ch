import re
import cv2
import time
import os
import random
import torch
import string
import warnings
import numpy as np
from torch import nn
from PIL import Image
import smtplib, ssl
import requests
import io
from PIL import Image as im
import torch
from django.shortcuts import render
from django.views.generic.edit import CreateView
import tensorflow_hub as hub
from . models import MaizeData
from django.shortcuts import render
from django.http import HttpResponse
from . forms import  UploadForm
from torchvision import transforms
from keras.models import load_model
from torch.autograd import Variable
from email.message import EmailMessage
from email.mime.text import MIMEText
from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from . serializers import ImageSerializer
from io import BytesIO
from ultralytics import YOLO
from urllib.parse import urlparse
from email.mime.multipart import MIMEMultipart
from keras.preprocessing.image import img_to_array
warnings.filterwarnings("ignore")
from rest_framework.parsers import MultiPartParser, FormParser
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
# Create your views here.

class PredictImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def get(self, request):
        return Response({'message':'This is maize prediction endpoint'})

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
                image = image.resize((244, 244))
                image = np.array(image)  # Convert PIL image to numpy array
                image = image.astype("float") / 255.0  # Normalize pixel values
                image = np.expand_dims(image, axis=0)  # Add batch dimension
                
                # Load .h5 model
                model = load_model(os.path.join(BASE_DIR, 'models/maize.h5'))
                
                # Make prediction
                prediction = model.predict(image)[0]
                
                # Convert prediction to JSON format
                response_data = {
                    'Cercospora Leaf Gray Leaf Spot': float(prediction[0]),
                    'Common Rust': float(prediction[1]),
                    'Northern Leaf Blight': float(prediction[2]),
                    'Healthy': float(prediction[3]),
                }
                
                return Response(response_data, status=200)
                 
            except Exception as e:
                return Response({'error': f'Failed to process the image: {str(e)}'}, status=400)

        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=400)

def maize_classifier(request):

    form = UploadForm()

    context = {'form': form}

    if request.method == 'POST' and request.FILES['file']:

        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():

                file_path = request.FILES['file']

                file_name = str(file_path.name).split('.')[0]

                # print('file name: ', file_name)

                file_name = str(file_name).replace(' ', '_')

                if str(file_path.name).lower().endswith(".jpg") or str(file_path.name).endswith(".png") or str(file_path.name).endswith(".jpeg"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = MaizeData(file_id=file_id, file_path=file_path, file_name=file_name)
                    # print("Saving file")
                    new_file.save()

                    # import all import libraries

                    """load file, returns tensor"""
                    file_path=os.path.join(BASE_DIR,'media/files/'+str(file_path.name).replace(' ', '_'))
                    # print("file path: ", file_path)
                    file = cv2.imread(file_path)

                    # pre-process the file for classification
                    file = cv2.resize(file, (244, 244))
                    file = file.astype("float") / 255.0
                    file = img_to_array(file)
                    file = np.expand_dims(file, axis=0)
                    
                    since_time = time.time();
                    # load the saved model
                    loaded_model = load_model(os.path.join(BASE_DIR,'models/maize.h5'))

                    probabilities = loaded_model.predict(file)[0]
                    
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
                    file_data = MaizeData.objects.get(file_id=file_id)
                    # print("file Details: ", file_data.file_path)
                    context = {'form': form,'prediction':pred_label, 'proba': pred_proba,
                    'pred_index': pred_index, 'probabilities': prob, 'file':file_data}
                    return render(request, 'classifiers/maize/maize-classification.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'form': form,'format_massage': format_message}

                    return render(request, 'classifiers/maize/maize-classification.html', context=context)

        else:
            return render(request, template_name="classifiers/maize/maize-classification.html", context=context)

    return render(request, template_name="classifiers/maize/maize-classification.html", context=context)


def maize_detect(request):
    form = UploadForm(request.POST, request.FILES)
    if form.is_valid():
        files = request.FILES.getlist('file')  # Get multiple files
        results_list = []

        for file_path in files:
            file_name = str(file_path.name).split('.')[0]
            extension = str(file_path.name).split('.')[-1]
            file_name = str(file_name).replace(' ', '_')
            
            letters = string.ascii_uppercase
            file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
            file_instance = MaizeData(file_id=file_id, file_path=file_path, file_name=file_name)
            file_instance.save()

            uploaded_file_qs = MaizeData.objects.filter().last()
            file_bytes = uploaded_file_qs.file_path.read()

            model = YOLO(os.path.join(BASE_DIR, 'models/maize_yolo.pt'))

            if extension.lower() in ['jpg', 'jpeg', 'png']:
                img = im.open(io.BytesIO(file_bytes))
                results = model.predict([img])
                print("Results: ", results)
                for i, r in enumerate(results):
                    im_bgr = r.plot()
                    class_names = [r.names[i.item()] for i in r.boxes.cls]
                    unique_class_names = list(set(class_names))
                    class_count = {name: class_names.count(name) for name in unique_class_names}
                    # print("Class Names: ", class_names)
                    # print("Class Count: ", class_count)
                    output_path = os.path.join('media', 'yolo_out', f'results_{file_name}_{i}.jpg')
                    cv2.imwrite(output_path, im_bgr)
                    results_list.append({"type": "image", "path": output_path, "names": class_count})

            elif extension.lower() in ['mp4', 'avi', 'mov']:
                temp_video_path = os.path.join(BASE_DIR, 'media', 'temp_video.' + extension)
                with open(temp_video_path, 'wb') as f:
                    f.write(file_bytes)
                
                cap = cv2.VideoCapture(temp_video_path)
                out_path = os.path.join('media', 'yolo_out', f'result_video_{file_name}.' + extension)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

                video_results = []  # Store video results
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict([frame])
                    for r in results:
                        frame = r.plot()
                        class_names = [r.names[i.item()] for i in r.boxes.cls]
                        unique_class_names = list(set(class_names))
                        class_count = {name: class_names.count(name) for name in unique_class_names}
                        # print("Class Names: ", class_names)
                        # print("Class Count: ", class_count)
                        video_results.append(class_count)  
                results_list.append({"type": "video", "path": out_path, "names": video_results})

                cap.release()
                out.release()
                os.remove(temp_video_path)

        form = UploadForm()
        context = {
            "form": form,
            "results_list": results_list
        }
        return render(request, template_name="classifiers/maize/maize-detection.html", context=context)

    else:
        form = UploadForm()
        context = {
            "form": form
        }
        return render(request, template_name="classifiers/maize/maize-detection.html", context=context)

