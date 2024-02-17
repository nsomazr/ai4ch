import re
from django.shortcuts import render
from django.http import HttpResponse
from . forms import  ImageHorizontal
from . models import BeansData
import string
import random
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
import torch
from torchvision import transforms
import warnings
import cv2
import numpy as np
import time
from torch import nn
from torch.autograd import Variable
from PIL import Image
import smtplib, ssl
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings("ignore")

# Create your views here.

def classifier(request):

    image_horizontal = ImageHorizontal()

    context = {'image_horizontal': image_horizontal}

    if request.method == 'POST' and request.FILES['image_file']:

        image_horizontal = ImageHorizontal(request.POST, request.FILES)

        if image_horizontal.is_valid():

                image_path = request.FILES['image_file']

                image_name = image_path.name

                # print('Image name: ', image_name)

                image_name = str(image_name).replace(' ', '_')

                if str(image_name).lower().endswith(".jpg") or str(image_name).endswith(".png"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    image_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = BeansData(image_id=image_id, filepaths=image_path, filename=image_name)

                    new_file.save()

                    # import all import libraries
                    
                    image = Image.open(os.path.join(BASE_DIR,'media/images/std.jpg'))
                    transform = transforms.Compose([
                        transforms.PILToTensor()
                    ])
                    img_tensor = transform(image)
                    img_tensor = img_tensor / 255 #normalization of pixels
                    train_mean = img_tensor.reshape(3,-1).mean(axis=1)
                    train_std = img_tensor.reshape(3,-1).std(axis=1)
                    # create a dataset transformer
                    transformer_input = transforms.Compose([transforms.Resize(512),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(512),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(train_mean, train_std)])

                    """load image, returns tensor"""
                    image_path=os.path.join(BASE_DIR,'media/images/'+image_name)
                    # print("Image path: ", image_path)
                    image = Image.open(image_path)
                    # img = cv2.imread(image_path)
                    # cv2.imshow("", img)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows() 
                    image = transformer_input(image).float()
                    image = Variable(image, requires_grad=True)

                    since_time = time.time();
                    # load the saved model
                    loaded_model = torch.load(os.path.join(BASE_DIR,'models/resnet_model.pth'), map_location='cpu')
                    loaded_model.eval()

                    output_single = loaded_model(image.view(1, 3, 512, 512))
                    output_single_probability = torch.softmax(output_single, dim=1)
                    prediction_proba,prediction=torch.max(output_single_probability, 1)
                    
                    
                    # labels dictionary
                    labels_dict = {'angular_leaf_spot': 0,
                                    'anthracnose': 1,
                                    'ascochyta_leaf_spot': 2,
                                    'bcmv_and_bcmnv': 3,
                                    'bean_rust': 4,
                                    'common_bacterial_blight': 5,
                                    'mixed_infection': 6}
                    probabilities = output_single_probability.detach().numpy()[0]

                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    initial_pred = ''
                    pred = ''
                    pred_index = 0
                    for class_name, class_index in labels_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                        if class_index == prediction:
                            initial_pred = class_name
                            if prediction_proba.item() >= 0.5:
                                 pred = class_name    
                                 pred_index = class_index
                            else:
                                pred = 'Undetermined'      
                    print("Probabilities: ", probabilities)  
                    print("Initial Prediction: ", initial_pred)       
                    print("Class: ", pred, " | Probabilty: ", prediction_proba.item() )
                    time_elapse = time.time() - since_time
                    print("Time elapse: ", time_elapse)
                    # getting all the objects of hotel.
                    image = BeansData.objects.get(image_id=image_id)
                    context = {'image_horizontal': image_horizontal,'prediction':pred, 'proba': prediction_proba.item(),
                    'pred_index': pred_index, 'probabilities': prob, 'image_path':image_name,'image_name':image_name.split('.')[-2].upper(), 'image_id':image_id, 'image':image}
                    return render(request, 'beans/pages/beans-classification-demo.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'image_horizontal': image_horizontal,'format_massage': format_message}

                    return render(request, 'beans/pages/beans-classification-demo.html', context=context)

        else:
            return render(request, template_name="beans/pages/beans-classification-demo.html", context=context)

    return render(request, template_name="beans/pages/beans-classification-demo.html", context=context)