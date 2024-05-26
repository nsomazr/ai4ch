import requests

url = 'https://ai4crophealth.or.tz/beans/predict/'
file_path = './test.jpeg'

with open(file_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())