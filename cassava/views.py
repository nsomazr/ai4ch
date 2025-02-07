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
import subprocess
from rest_framework import status
from PIL import Image as im
import torch
import warnings
import numpy as np
from torch import nn
from PIL import Image
import smtplib, ssl
import requests
import tensorflow_hub as hub
from . models import CassavaData, CassavaDetectionResult, CassavaPredictionResult
from django.shortcuts import render
from . serializers import ImageSerializer
from django.http import HttpResponse
from . forms import  UploadForm
from torchvision import transforms
from keras.models import load_model
from torch.autograd import Variable
from email.message import EmailMessage
from email.mime.text import MIMEText
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from PIL import Image
from users.models import User
from .img_loc import extract_image_location
from . serializers import ImageSerializer, FileSerializer
from rest_framework import status
from io import BytesIO
from django.conf import settings
from ultralytics import YOLO
from urllib.parse import urlparse
from django.shortcuts import  render, redirect
from email.mime.multipart import MIMEMultipart
from keras.preprocessing.image import img_to_array
warnings.filterwarnings("ignore")
from rest_framework.parsers import MultiPartParser, FormParser
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
from config import base
import base64
import json

# Beem Africa
def send_detection_sms(phone_number, type, names):
    api_key = base.Config.BEEM_SMS_API_KEY
    secret_key = base.Config.BEEM_SMS_SECRET_KEY
    
    # Create names string in both languages
    names_en = "\n".join([f"- {name}: {count} " for name, count in names.items()])
    names_sw = "\n".join([f"- {name}: {count} " for name, count in names.items()])

    # English SMS
    sms_en = f"Here is the result from the {type} you have uploaded:\n{names_en}\n\nThank you for using our platform."
    
    # Swahili SMS
    sms_sw = f"Haya ndiyo matokeo ya {type} uliyopakia:\n{names_sw}\n\nAsante kwa kutumia jukwaa letu."

    # Combine both languages
    sms = f"{sms_en}\n-----\n{sms_sw}"
    
    phone_number = str(phone_number)[1:]
    post_data = {
        'source_addr': 'CROP HEALTH',
        'encoding': 0,
        'schedule_time': '',
        'message': sms,
        'recipients': [{'recipient_id': '1', 'dest_addr': phone_number}]
    }
    url = 'https://apisms.beem.africa/v1/send'

    headers = {
        'Authorization': 'Basic ' + base64.b64encode(f"{api_key}:{secret_key}".encode()).decode(),
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=json.dumps(post_data), verify=False)

    data = response.json()
    
    # print(data)

    return data.get('successful', False)
# Create your views here.

class CassavaPredictImageView(APIView):
    
    parser_classes = (MultiPartParser, FormParser)
    
    def get(self, request):
        return Response({'message':'This is cassava prediction endpoint'})
        
    def post(self, request):
        
        serializer = ImageSerializer(data=request.data)

        # Validate the data
        if serializer.is_valid():
            # Access the image file
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
                # Open the image file
                image = Image.open(image_file)
                
                # Resize and preprocess the image for classification
                image = image.resize((300, 300))
                image = np.array(image)  # Convert PIL image to numpy array
                image = image.astype("float") / 255.0  # Normalize pixel values
                image = np.expand_dims(image, axis=0)  # Add batch dimension
            
                # Load .h5 model
                loaded_model = load_model(os.path.join(BASE_DIR,'models/classification/cassava_classification.h5'))
                # Make prediction
                prediction = loaded_model.predict(image)[0]

                # Convert prediction to JSON format
                response_data = {
                    'Bacterial Blight': float(prediction[0]),
                    'Brown Spot': float(prediction[1]),
                    'Browm Steak': float(prediction[2]),
                    'Green Mottle':float(prediction[3]),
                    'Healthy':float(prediction[4]),
                    'Mosaic Desease':float(prediction[5]),
                }

                return Response(response_data, status=200)
                 
            except Exception as e:
                # print(e)
                return Response({'error': 'Failed to download the image with error '}, status=400)
        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=400)

def image_cassava_classifier(request):
    if request.session.get('user_id'):
        upload_form = UploadForm()

        context = {'upload_form': upload_form}

        if request.method == 'POST' and request.FILES['file']:

            upload_form = UploadForm(request.POST, request.FILES)

            if upload_form.is_valid():

                file_path = request.FILES['file']

                file_name = str(file_path.name).split('.')[0]

                file_name = re.sub(r'[ ()]', '_', file_name)


                if str(file_path.name).lower().endswith(".jpg") or str(file_path.name).lower().endswith(".png") or str(file_path.name).lower().endswith(".jpeg"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    user = User.objects.get(id=request.session['user_id'])
                    new_file = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name, uploaded_by=user)
                    # print("Saving file")
                    new_file.save()

                    # import all import libraries

                    """load file, returns tensor"""
                    file_path=os.path.join(BASE_DIR,'media/files/'+str(file_path.name).replace(' ', '_'))
                    # print("file path: ", file_path)
                    file = cv2.imread(file_path)

                    # pre-process the file for classification
                    file = cv2.resize(file, (300, 300))
                    file = file.astype("float") / 255.0
                    file = img_to_array(file)
                    file = np.expand_dims(file, axis=0)
                    
                    # since_time = time.time();
                    # load the saved model
                    loaded_model = load_model(os.path.join(BASE_DIR,'models/classification/cassava_classification.h5'))

                    probabilities = loaded_model.predict(file)[0]
                        
                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    pred_proba = np.max(probabilities)
                    pred_index = np.argmax(probabilities)
                    
                    # labels dictionary
                    labels_dict = {'Bacterial Blight': 0,
                                    'Brown Spot': 1,
                                    'Browm Steak': 2,
                                    'Green Mottle': 3,
                                    'Healthy':4,
                                    'Mosaic Desease': 5,
                                    }
                    
                    pred_label = None
                    for class_name, class_index in labels_dict.items():
                        if class_index == pred_index:
                            if pred_proba >= 0.5:
                                pred_label = class_name    
                            else:
                                pred_label = 'Undetermined'  


                    # time_elapse = time.time() - since_time
                    # print("Time elapse: ", time_elapse)
                    image_data = CassavaData.objects.get(file_id=file_id)
                    
                    prediction_result = CassavaPredictionResult.objects.create(
                        user=user,
                        file_name=file_name,
                        file_path=file_path,
                        predicted_disease=pred_label,
                        confidence_score=pred_proba,
                        probabilities={
                            'Bacterial Blight': float(prob[0]),
                            'Brown Spot': float(prob[1]),
                            'Browm Steak': float(prob[2]),
                            'Green Mottle':float(prob[3]),
                            'Healthy':float(prob[4]),
                            'Mosaic Desease':float(prob[5]),
                        }
                    )
                    # print("Image Details: ", image_data.image_path)
                    context = {'upload_form': upload_form,'prediction':pred_label, 'proba': pred_proba,
                    'pred_index': pred_index, 'probabilities': prob, 'image':image_data}
                    return render(request, 'interfaces/cassava/cassava-classification.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'upload_form': upload_form,'format_massage': format_message}

                    return render(request, 'interfaces/cassava/cassava-classification.html', context=context)

            else:
                return render(request, template_name="interfaces/cassava/cassava-classification.html", context=context)

        return render(request, template_name="interfaces/cassava/cassava-classification.html", context=context)
    else:
        return redirect("ai4chapp:login")

def tensor_to_list(tensor):
    return tensor.numpy().tolist()



def image_cassava_detect(request):
    if request.session.get('user_id'):
        upload_form = UploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            files = request.FILES.getlist('file')  # Get multiple files
            results_list = []

            for file_path in files:
                file_name = str(file_path.name).split('.')[0]
                extension = str(file_path.name).split('.')[-1]
                file_name = str(file_name).replace(' ', '_')
                
                letters = string.ascii_uppercase
                file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                user = User.objects.get(id=request.session['user_id'])
                if file_path:  # Only on creation
                    location_info = extract_image_location(file_path)
                    # print("Location info: ",location_info)
                    if location_info:
                        latitude = location_info['latitude']
                        longitude = location_info['longitude']
                        region = location_info['region']
                        district = location_info['district']
                        country = location_info['country']
                        full_address = location_info['full_address']
                        file_instance = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name, 
                                            latitude=latitude,
                                            longitude=longitude,
                                            region=region,
                                            district=district,
                                            country=country,
                                            full_address=full_address,
                                            uploaded_by=user)
                        file_instance.save()
                    else:
                        file_instance = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name,uploaded_by=user)
                        file_instance.save()

                uploaded_file_qs = CassavaData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()

                model = YOLO(os.path.join(BASE_DIR, 'models/detection/cassava_detection.pt'))
            

                if extension.lower() in ['jpg', 'jpeg', 'png']:
                    img = im.open(io.BytesIO(file_bytes))
                    results = model.predict([img])
                    
                    for i, r in enumerate(results):
                        im_bgr = r.plot()
                        class_names = [r.names[i.item()] for i in r.boxes.cls]
                        unique_class_names = list(set(class_names))
                        class_count = {name: class_names.count(name) for name in unique_class_names}
                        # print("Class Names: ", class_names)
                        # print("Class Count: ", class_count)
                        output_path_ = os.path.join('yolo_out', f'results_{file_name}_{i}.jpg')
                        output_path = os.path.join('media', 'yolo_out', f'results_{file_name}_{i}.jpg')
                        cv2.imwrite(output_path, im_bgr)
                        detection_result = CassavaDetectionResult.objects.create(
                            result_id = file_id,
                            user=user,
                            file_name=file_name,
                            file_path=file_path,
                            output_path=output_path_,
                            file_type='image',
                            detection_results=class_count
                        )
                        results_list.append({"type": "image", "path": output_path, "names": class_count})

            upload_form = UploadForm()
            if results_list:
                send_detection_sms(request.user.phone_number, 'image', results_list[0]['names'])
            context = {
                "upload_form": upload_form,
                "results_list": results_list
            }
            return render(request, template_name="interfaces/cassava/cassava-detection.html", context=context)

        else:
            upload_form = UploadForm()
            context = {
                "upload_form": upload_form
            }
            return render(request, template_name="interfaces/cassava/cassava-detection.html", context=context)
    else:
        return redirect("ai4chapp:login")

def video_cassava_detect(request):
    if request.session.get('user_id'):
        upload_form = UploadForm(request.POST, request.FILES)
        results_list = []

        if upload_form.is_valid():
            files = request.FILES.getlist('file')  # Get multiple files

            for file_path in files:
                file_name = str(file_path.name).split('.')[0]
                extension = str(file_path.name).split('.')[-1]
                file_name = str(file_name).replace(' ', '_')
                
                letters = string.ascii_uppercase
                file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                user = User.objects.get(id=request.session['user_id'])
                if file_path:  # Only on creation
                    location_info = extract_image_location(file_path)
                    # print("Location info: ",location_info)
                    if location_info:
                        latitude = location_info['latitude']
                        longitude = location_info['longitude']
                        region = location_info['region']
                        district = location_info['district']
                        country = location_info['country']
                        full_address = location_info['full_address']
                        file_instance = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name, 
                                            latitude=latitude,
                                            longitude=longitude,
                                            region=region,
                                            district=district,
                                            country=country,
                                            full_address=full_address,
                                            uploaded_by=user)
                        file_instance.save()
                    else:
                        file_instance = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name,uploaded_by=user)
                        file_instance.save()

                uploaded_file_qs = CassavaData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()

                model = YOLO(os.path.join(BASE_DIR, 'models/detection/cassava_detection.pt'))

                if extension.lower() in ['mp4', 'avi', 'mov']:
                    temp_video_path = os.path.join(BASE_DIR, 'media', 'temp_video.' + extension)
                    with open(temp_video_path, 'wb') as f:
                        f.write(file_bytes)

                    converted_video_path = os.path.join(BASE_DIR, 'media', 'temp_video_converted.mp4')
                    # Convert video using ffmpeg
                    ffmpeg_command = [
                        'ffmpeg', '-i', temp_video_path, '-vcodec', 'libx264', '-acodec', 'aac', 
                        '-strict', 'experimental', converted_video_path
                    ]
                    subprocess.call(ffmpeg_command)

                    cap = cv2.VideoCapture(converted_video_path)
                    out_path_ = os.path.join('yolo_out', f'result_video_{file_name}.mp4')
                    out_path = os.path.join('media', 'yolo_out', f'result_video_{file_name}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

                    video_results = {}  # Store video results as a dictionary
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = model.predict([frame])
                        for r in results:
                            frame = r.plot()
                            out.write(frame)
                            class_names = [r.names[i.item()] for i in r.boxes.cls]
                            unique_class_names = list(set(class_names))
                            for name in unique_class_names:
                                video_results[name] = video_results.get(name, 0) + class_names.count(name)
                    detection_result = CassavaDetectionResult.objects.create(
                        user=user,
                        result_id=file_id,
                        file_name=file_name,
                        file_path=file_path,
                        output_path=out_path_,
                        file_type='video',
                        detection_results=video_results
                    )
                    results_list.append({"type": "video", "path": out_path, "names": video_results})
                    # print(f"Video saved at: {os.path.join(settings.MEDIA_ROOT, out_path)}")
                    cap.release()
                    out.release()
                    os.remove(temp_video_path)
                    os.remove(converted_video_path)

                    # Debugging: Print the video path
                    # print("Video saved at:", out_path)
        if results_list:
            send_detection_sms(request.user.phone_number, 'video', results_list[0]['names'])
        upload_form = UploadForm()
        context = {
            "upload_form": upload_form,
            "results_list": results_list
        }
        return render(request, template_name="interfaces/cassava/cassava-detection.html", context=context)
    else:
        return redirect("ai4chapp:login")


class CassavaDetectImageAPI(APIView):
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        # print("Hello post")
        serializer = FileSerializer(data=request.data)
        # Validate the data
        if serializer.is_valid():
            # Access the image file
            file_path = serializer.validated_data['file']
            user_id = serializer.validated_data['user_id']
            try:
                # Open the image file
                file_name = str(file_path.name).split('.')[0]
                extension = str(file_path.name).split('.')[-1]
                file_name = str(file_name).replace(' ', '_')
                
                letters = string.ascii_uppercase
                file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                user = User.objects.get(id=user_id)
                file_instance = CassavaData(file_id=file_id, file_path=file_path, file_name=file_name,uploaded_by=user)
                file_instance.save()

                uploaded_file_qs = CassavaData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()
                model = YOLO(os.path.join(BASE_DIR, 'models/detection/cassava_detection.pt'))
                if extension.lower() in ['jpg', 'jpeg', 'png']:
                    img = im.open(io.BytesIO(file_bytes))
                    results = model.predict([img])                    
                    result = results[0]
                    boxes = result.boxes
                    names = result.names if hasattr(result, 'names') else None
                    orig_shape = result.orig_shape if hasattr(result, 'orig_shape') else None

                    if boxes is not None:
                        # Extract data from the Boxes object
                        cls = tensor_to_list(boxes.cls)
                        conf = tensor_to_list(boxes.conf)
                        data = tensor_to_list(boxes.data)
                        xywh = tensor_to_list(boxes.xywh)
                        xywhn = tensor_to_list(boxes.xywhn)
                        xyxy = tensor_to_list(boxes.xyxy)
                        xyxyn = tensor_to_list(boxes.xyxyn)

                        boxes_data = {
                            'cls': cls,
                            'conf': conf,
                            'data': data,
                            'xywh': xywh,
                            'xywhn': xywhn,
                            'xyxy': xyxy,
                            'xyxyn': xyxyn
                        }
                        response_data = {
                            'boxes': boxes_data,
                            'names': names,
                            'orig_shape': orig_shape
                        }
                    else:
                        response_data = {
                            'error': 'No boxes found in results'
                        }
                    return Response({"results": response_data}, status=status.HTTP_200_OK)

                elif extension.lower() in ['mp4', 'avi', 'mov']:
                    temp_video_path = os.path.join(BASE_DIR, 'media', 'temp_video.' + extension)
                    with open(temp_video_path, 'wb') as f:
                        f.write(file_bytes)

                    cap = cv2.VideoCapture(temp_video_path)
                    out_path = os.path.join('media', 'yolo_out', f'result_video_{file_name}.' + extension)
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = model.predict([frame])                        
                        result = results[0]
                        boxes = result.boxes
                        names = result.names if hasattr(result, 'names') else None
                        orig_shape = result.orig_shape if hasattr(result, 'orig_shape') else None

                        if boxes is not None:
                            # Extract data from the Boxes object
                            cls = tensor_to_list(boxes.cls)
                            conf = tensor_to_list(boxes.conf)
                            data = tensor_to_list(boxes.data)
                            xywh = tensor_to_list(boxes.xywh)
                            xywhn = tensor_to_list(boxes.xywhn)
                            xyxy = tensor_to_list(boxes.xyxy)
                            xyxyn = tensor_to_list(boxes.xyxyn)

                            boxes_data = {
                                'cls': cls,
                                'conf': conf,
                                'data': data,
                                'xywh': xywh,
                                'xywhn': xywhn,
                                'xyxy': xyxy,
                                'xyxyn': xyxyn
                            }

                            response_data = {
                                'boxes': boxes_data,
                                'names': names,
                                'orig_shape': orig_shape
                            }
                        else:
                            response_data = {
                                'error': 'No boxes found in results'
                            }

                    cap.release()
                    out.release()
                    os.remove(temp_video_path)
                    return Response({"results": response_data}, status=status.HTTP_200_OK)
                
            except Exception as e:
                return Response({'error': f'Failed to process the file: {str(e)}'}, status=status.HTTP_200_OK)
            
        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=400)