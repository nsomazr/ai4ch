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
from rest_framework import status
from PIL import Image as im
import torch
import warnings
import numpy as np
from torch import nn
from PIL import Image
import smtplib, ssl
import requests
import subprocess
import tensorflow_hub as hub
from . models import BeansData, BeansDetectionResult, BeansPredictionResult
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
from PIL import Image
from rest_framework.permissions import AllowAny
from . serializers import ImageSerializer, FileSerializer
from rest_framework import status
from io import BytesIO
from ultralytics import YOLO
from .img_loc import extract_image_location
from urllib.parse import urlparse
from users.models import User
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


class PredictImageView(APIView):
    
    permission_classes = [AllowAny]
    
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
                # image = Image.open(temp_path)
                
                image = cv2.imread(temp_path)

                # pre-process the image for classification
                image = cv2.resize(image, (250, 250)) 
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                
                since_time = time.time();
                # load the saved model
                loaded_model = load_model(os.path.join(BASE_DIR,'models/classification/beans_classification.h5'))

                probabilities = loaded_model.predict(image)[0]
                
                probs=[]
                for i in probabilities:
                    probs.append(i)

                # Convert prediction to JSON format
                response_data = {
                    'Angular Leaf Spot': float(probs[0]),
                    'Anthracnose': float(probs[1]),
                    'Ascochyta Leaf Spot': float(probs[2]),
                    'Common Bacterial Blight': float(probs[3]),
                    'Common Mosaic Infection': float(probs[4]),
                    'Bean Rust': float(probs[5]),
                }

                # Clean up the temporary file
                os.remove(temp_path)

                return Response(response_data, status=status.HTTP_200_OK)

            except Exception as e:
                # print(e)
                # Clean up the temporary file in case of an error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return Response({'error': 'Failed to process the image with error: {}'.format(str(e))}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Return a response with validation errors if the data is invalid
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def image_beans_classifier(request):
    if request.session.get('user_id'):
        upload_form = UploadForm()
        context = {'upload_form': upload_form}

        if request.method == 'POST' and request.FILES.get('file'):

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
                    new_file = BeansData(file_id=file_id, file_path=file_path, file_name=file_name,uploaded_by=user)
                    # print("Saving file")
                    new_file.save()

                    # import all import libraries

                    """load file, returns tensor"""
                    file_path=os.path.join(BASE_DIR,'media/files/'+str(file_path.name).replace(' ', '_'))
                    # print("file path: ", file_path)
                    file = cv2.imread(file_path)

                    # pre-process the file for classification
                    file = cv2.resize(file, (250, 250))
                    file = file.astype("float") / 255.0
                    file = img_to_array(file)
                    file = np.expand_dims(file, axis=0)
                    
                    # since_time = time.time();
                    # load the saved model
                    loaded_model = load_model(os.path.join(BASE_DIR,'models/classification/beans_classification.h5'))

                    probabilities = loaded_model.predict(file)[0]
                    
                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    pred_proba = np.max(probabilities)
                    pred_index = np.argmax(probabilities)
                    
                    # labels dictionary
                    labels_dict = {'Angular Leaf Spot': 0,
                                'Anthracnose': 1,
                                'Ascochyta Leaf Spot': 2,
                                'Common Bacterial Blight': 3,
                                'Common Mosaic Rust': 4,
                                'Bean Rust': 5,
                                }
                    
                    pred_label = None
                    for class_name, class_index in labels_dict.items():
                        if class_index == pred_index:
                            if pred_proba >= 0.5:
                                pred_label = class_name    
                            else:
                                pred_label = 'Undetermined'  

                    # time_elapse = time.time() - since_time
                    image_data = BeansData.objects.get(file_id=file_id)
                    prediction_result = BeansPredictionResult.objects.create(
                                user=user,
                                file_name=file_name,
                                file_path=file_path,
                                predicted_disease=pred_label,
                                confidence_score=pred_proba,
                                probabilities={
                                    'Angular Leaf Spot': prob[0],
                                    'Anthracnose': prob[1],
                                    'Ascochyta Leaf Spot': prob[2],
                                    'Common Bacterial Blight': prob[3],
                                    'Common Mosaic Rust': prob[4],
                                    'Bean Rust': prob[5],
                                    # Add other probabilities as needed
                                }
                            )
                    context = {'upload_form': upload_form, 'prediction': pred_label, 'proba': pred_proba,
                            'pred_index': pred_index, 'probabilities': prob, 'image': image_data}
                    return render(request, 'interfaces/beans/beans-classification.html', context=context)

                else:
                    format_message = "Unsupported format, supported formats are .png and .jpg"
                    context = {'upload_form': upload_form, 'format_message': format_message}
                    return render(request, 'interfaces/beans/beans-classification.html', context=context)

            else:
                return render(request, template_name="interfaces/beans/beans-classification.html", context=context)

        return render(request, template_name="interfaces/beans/beans-classification.html", context=context)
    else:
        return redirect("ai4chapp:login")

def tensor_to_list(tensor):
    return tensor.numpy().tolist()



def find_nearby_agrovets(region=None, district=None, ward=None, street=None):
    """
    Find nearby agrovets based on location parameters.
    Returns agrovets in order of closest match (exact location match first, then region, etc.)
    """
    agrovet_queryset = User.objects.filter(role='agrovet', is_verified=False, status=True)
    
    # Start with most specific location match
    if street and ward and district and region:
        exact_matches = agrovet_queryset.filter(
            street=street,
            ward=ward,
            district=district,
            region=region
        )
        if exact_matches.exists():
            return exact_matches
    
    # Try ward level match
    if ward and district and region:
        ward_matches = agrovet_queryset.filter(
            ward=ward,
            district=district,
            region=region
        )
        if ward_matches.exists():
            return ward_matches
    
    # Try district level match
    if district and region:
        district_matches = agrovet_queryset.filter(
            district=district,
            region=region
        )
        if district_matches.exists():
            return district_matches
    
    # Fall back to region level match
    if region:
        region_matches = agrovet_queryset.filter(region=region)
        if region_matches.exists():
            return region_matches
    
    # If no matches found, return empty queryset
    return User.objects.none()

def image_beans_detect(request):
    if request.session.get('user_id'):
        upload_form = UploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            files = request.FILES.getlist('file')
            results_list = []
            nearby_agrovets = []

            for file_path in files:
                file_name = str(file_path.name).split('.')[0]
                extension = str(file_path.name).split('.')[-1]
                file_name = str(file_name).replace(' ', '_')
                
                letters = string.ascii_uppercase
                file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                user = User.objects.get(id=request.session['user_id'])
                
                # Get location info and find nearby agrovets
                location_info = extract_image_location(file_path)
                if location_info:
                    # Use image location
                    nearby_agrovets = find_nearby_agrovets(
                        region=location_info['region'],
                        district=location_info['district'],
                        ward=None,  # Assuming image location doesn't provide ward
                        street=None  # Assuming image location doesn't provide street
                    )
                    latitude = location_info['latitude']
                    longitude = location_info['longitude']
                    region = location_info['region']
                    district = location_info['district']
                    country = location_info['country']
                    full_address = location_info['full_address']
                    file_instance = BeansData(
                        file_id=file_id, file_path=file_path, file_name=file_name,
                        latitude=latitude, longitude=longitude,
                        region=region, district=district,
                        country=country, full_address=full_address,
                        uploaded_by=user
                    )
                else:
                    # Use user's location
                    nearby_agrovets = find_nearby_agrovets(
                        region=user.region,
                        district=user.district,
                        ward=user.ward,
                        street=user.street
                    )
                    file_instance = BeansData(
                        file_id=file_id,
                        file_path=file_path,
                        file_name=file_name,
                        uploaded_by=user
                    )
                file_instance.save()

                # print("agrovet details: ", nearby_agrovets)
                # Process image detection
                uploaded_file_qs = BeansData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()
                model = YOLO(os.path.join(BASE_DIR, 'models/detection/beans_detection.pt'))

                if extension.lower() in ['jpg', 'jpeg', 'png']:
                    img = im.open(io.BytesIO(file_bytes))
                    results = model.predict([img])
                    for i, r in enumerate(results):
                        im_bgr = r.plot()
                        class_names = [r.names[i.item()] for i in r.boxes.cls]
                        unique_class_names = list(set(class_names))
                        class_count = {name: class_names.count(name) for name in unique_class_names}
                        
                        output_path_ = os.path.join('yolo_out', f'results_{file_name}_{i}.jpg')
                        output_path = os.path.join('media', 'yolo_out', f'results_{file_name}_{i}.jpg')
                        cv2.imwrite(output_path, im_bgr)
                        
                        detection_result = BeansDetectionResult.objects.create(
                            result_id=file_id,
                            user=user,
                            file_name=file_name,
                            file_path=file_path,
                            output_path=output_path_,
                            file_type='image',
                            detection_results=class_count
                        )
                        results_list.append({
                            "type": "image",
                            "path": output_path,
                            "names": class_count
                        })

            # Format agrovet data for template
            agrovet_list = [{
                'name': agrovet.get_full_name(),
                'phone_number': agrovet.phone_number,
                'location': f"{agrovet.street}, {agrovet.ward}, {agrovet.district}, {agrovet.region}",
                'email': agrovet.email
            } for agrovet in nearby_agrovets]

            upload_form = UploadForm()
            
            if results_list:
                send_detection_sms(request.user.phone_number, 'image', results_list[0]['names'])
            
            context = {
                "upload_form": upload_form,
                "results_list": results_list,
                "nearby_agrovets": agrovet_list
            }
            return render(request, template_name="interfaces/beans/beans-detection.html", context=context)
        else:
            upload_form = UploadForm()
            context = {
                "upload_form": upload_form
            }
            return render(request, template_name="interfaces/beans/beans-detection.html", context=context)
    else:
        return redirect("ai4chapp:login")

def video_beans_detect(request):
    if request.session.get('user_id'):
        upload_form = UploadForm(request.POST, request.FILES)
        results_list = []
        nearby_agrovets = []

        if upload_form.is_valid():
            files = request.FILES.getlist('file')  # Get multiple files

            for file_path in files:
                file_name = str(file_path.name).split('.')[0]
                extension = str(file_path.name).split('.')[-1]
                file_name = str(file_name).replace(' ', '_')
                
                letters = string.ascii_uppercase
                file_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                user = User.objects.get(id=request.session['user_id'])
                location_info = extract_image_location(file_path)
                if location_info:
                    # Use image location
                    nearby_agrovets = find_nearby_agrovets(
                        region=location_info['region'],
                        district=location_info['district'],
                        ward=None,  # Assuming image location doesn't provide ward
                        street=None  # Assuming image location doesn't provide street
                    )
                    latitude = location_info['latitude']
                    longitude = location_info['longitude']
                    region = location_info['region']
                    district = location_info['district']
                    country = location_info['country']
                    full_address = location_info['full_address']
                    file_instance = BeansData(
                        file_id=file_id, file_path=file_path, file_name=file_name,
                        latitude=latitude, longitude=longitude,
                        region=region, district=district,
                        country=country, full_address=full_address,
                        uploaded_by=user
                    )
                else:
                    # Use user's location
                    nearby_agrovets = find_nearby_agrovets(
                        region=user.region,
                        district=user.district,
                        ward=user.ward,
                        street=user.street
                    )
                    file_instance = BeansData(
                        file_id=file_id,
                        file_path=file_path,
                        file_name=file_name,
                        uploaded_by=user
                    )
                file_instance.save()

                uploaded_file_qs = BeansData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()

                model = YOLO(os.path.join(BASE_DIR, 'models/detection/beans_detection.pt'))

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
                    out_path_ = os.path.join('media', 'yolo_out', f'result_video_{file_name}.mp4')
                    out_path = os.path.join('media', 'yolo_out', f'result_video_{file_name}.mp4')
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                    detection_result = BeansDetectionResult.objects.create(
                        user=user,
                        result_id=file_id,
                        file_name=file_name,
                        file_path=file_path,
                        output_path=out_path_,
                        file_type='video',
                        detection_results=video_results
                    )
                    results_list.append({"type": "video", "path": out_path, "names": video_results})

                    cap.release()
                    out.release()
                    os.remove(temp_video_path)
                    os.remove(converted_video_path)

                    # Debugging: Print the video path
                    # print("Video saved at:", out_path)
            # Format agrovet data for template
            agrovet_list = [{
                'name': agrovet.get_full_name(),
                'phone_number': agrovet.phone_number,
                'location': f"{agrovet.street}, {agrovet.ward}, {agrovet.district}, {agrovet.region}",
                'email': agrovet.email
            } for agrovet in nearby_agrovets]

            upload_form = UploadForm()
            
            if results_list:
                send_detection_sms(request.user.phone_number, 'image', results_list[0]['names'])
            
            context = {
                "upload_form": upload_form,
                "results_list": results_list,
                "nearby_agrovets": agrovet_list
            }
            return render(request, template_name="interfaces/beans/beans-detection.html", context=context)
    else:
        return redirect("ai4chapp:login")
        

class BeansDetectAPI(APIView):
    
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
                file_instance = BeansData(file_id=file_id, file_path=file_path, file_name=file_name, uploaded_by=user)
                file_instance.save()

                uploaded_file_qs = BeansData.objects.filter().last()
                file_bytes = uploaded_file_qs.file_path.read()
                model = YOLO(os.path.join(BASE_DIR, 'models/detection/beans_detection.pt'))
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
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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