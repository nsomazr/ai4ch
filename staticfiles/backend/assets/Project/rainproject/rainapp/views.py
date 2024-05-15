import re
from django.shortcuts import render, redirect
from django.http import HttpResponse
from . forms import  ImageHorizontal
from . models import RainData
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
from .forms import LoginForm, RegisterForm
from .models import Users
warnings.filterwarnings("ignore")

def register(request):

    if request.method == 'POST':

        register_form = RegisterForm(request.POST)

        if register_form.is_valid():

            print(" Passed ")

            fname = request.POST['first_name']

            print(fname)

            lname = request.POST['last_name']
            email = request.POST['email']
            username = request.POST['username']
            status = 1
            password = request.POST['password']
            cpassword= request.POST['cpassword']

            if password != cpassword:

                register_form = RegisterForm()

                matcherror = "Passwords don't match"

                context = {'register_form': register_form,'merror':matcherror}

                return render(request,'pages/register.html',context=context)

            new_user = Users(first_name=fname,last_name=lname, username=username, email=email, status=status, password=password)
            new_user.save()
            login_form = LoginForm()
            register_sms = "Registered Successful"
            return render(request, template_name='pages/login.html', context={'login_form':login_form, 'register_sms': register_sms})

        else:
            register_form = RegisterForm()

            context = {'register_form': register_form}

            return render(request, 'pages/register.html', context)

    else:
        register_form = RegisterForm()

        context = {'register_form':register_form}

        return render(request,'pages/register.html',context)


def login(request):

    if request.method == 'POST':

        login_form = LoginForm(request.POST)

        if login_form.is_valid():

            email_username = request.POST['email_username']

            password = request.POST['password']

            try:

                if Users.objects.filter(email=email_username, password=password):

                    my_info = Users.objects.get(email=email_username, password=password)

                    request.session['user_id'] = my_info.user_id

                    request.session['fname'] = my_info.first_name

                    request.session['lname'] = my_info.last_name

                    request.session['email'] = my_info.email

                    if my_info.status == 0:

                        context = {}

                        return render(request, 'pages/admin_dashboard.html', context)

                    elif my_info.status == 1:


                        image_horizontal = ImageHorizontal()

                        context = {'image_horizontal': image_horizontal}

                        return render(request, 'pages/rain-detection.html', context)
                    else:

                        login_form = LoginForm()

                        errormessage = "Incorrect credentials"

                        context = {'login_form': login_form, 'error': errormessage}

                        return render(request, 'pages/login.html', context)

                else:

                    login_form = LoginForm()

                    errormessage = "Incorrect credentials"

                    context = {'login_form': login_form, 'error': errormessage}

                    return render(request, 'pages/login.html', context)

            except ValueError:

                login_form = LoginForm()

                errormessage = "Incorrect credentials"

                context = {'login_form': login_form, 'error': errormessage}

                return render(request, 'pages/login.html', context)


        else:
            login_form = LoginForm()

            errormessage = "Incorrect credentials"

            context = {'login_form': login_form, 'error': errormessage}

            return render(request, 'pages/login.html', context)

    else:
        login_form = LoginForm()

        context = {'login_form': login_form}

        return render(request, template_name='pages/login.html',context=context)
# Create your views here.

def classifier(request):

    image_horizontal = ImageHorizontal()

    context = {'image_horizontal': image_horizontal}

    if request.method == 'POST' and request.FILES['image_file']:

        image_horizontal = ImageHorizontal(request.POST, request.FILES)

        if image_horizontal.is_valid():

                image_path = request.FILES['image_file'] 
                 
                image_name = image_path.name
                
                image_path_save=os.path.join(BASE_DIR,'media/images/'+image_name)  

                # print('Image name: ', image_name)

                image_name = str(image_name).replace(' ', '_')

                if str(image_name).lower().endswith(".jpg") or str(image_name).endswith(".png") or str(image_name).endswith(".jpeg"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    image_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = RainData(image_id=image_id, filepaths= image_path, filename=image_name)

                    new_file.save()

                    # import all import libraries
                    
                    image = Image.open(os.path.join(BASE_DIR,'media/images/std.png'))
                    transform = transforms.Compose([
                        transforms.PILToTensor()
                    ])
                    img_tensor = transform(image)
                    img_tensor = img_tensor / 255 #normalization of pixels
                    train_mean = img_tensor.reshape(3,-1).mean(axis=1)
                    train_std = img_tensor.reshape(3,-1).std(axis=1)
                    # create a dataset transformer
                    transformer_input = transforms.Compose([transforms.Resize((256,256)),
                                  transforms.Grayscale(num_output_channels=3),
                                  transforms.ToTensor(),
                                  transforms.Normalize((.2,),(0.3,))])

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
                    loaded_model = torch.load(os.path.join(BASE_DIR,'models/rd_model.pth'), map_location='cpu')
                    loaded_model.eval()

                    output_single = loaded_model(image.view(1, 3, 256, 256))
                    output_single_probability = torch.softmax(output_single, dim=1)
                    prediction_proba,prediction=torch.max(output_single_probability, 1)
                    
                    
                    # labels dictionary
                    labels_dict = {'cloud': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}
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
                    image = RainData.objects.get(image_id=image_id)
                    print("Path: ", image.filepaths.url)
                    context = {'image_horizontal': image_horizontal,'prediction':pred, 'proba': prediction_proba.item(),
                    'pred_index': pred_index, 'probabilities': prob, 'image_path':image_name,'image_name':image_name.split('.')[-2].upper(), 'image_id':image_id, 'image':image,'base':BASE_DIR}
                    return render(request, 'pages/rain-detection.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'image_horizontal': image_horizontal,'format_massage': format_message}

                    return render(request, 'pages/rain-detection.html', context=context)

        else:
            return render(request, template_name="pages/rain-detection.html", context=context)

    return render(request, template_name="pages/rain-detection.html", context=context)

def logout(request):
    try:
        del request.session['user_id']
    except KeyError:
        pass
    return redirect("login")