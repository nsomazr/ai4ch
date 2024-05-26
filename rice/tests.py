import requests

url = 'https://ai4crophealth.or.tz/rice/predict/'
file_path = './rice.jpg'

with open(file_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())