import requests

url = 'https://ai4crophealth.or.tz/cassava/predict/'
file_path = './cassava.JPG'

with open(file_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())