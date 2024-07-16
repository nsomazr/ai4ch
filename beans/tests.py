import requests

# url = 'https://portalai4crophealth.or.tz/beans/predict/'
url  = 'http://127.0.0.1:8000/beans/predict/'
file_path = './bean.jpeg'

with open(file_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())