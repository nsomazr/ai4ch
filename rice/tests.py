import requests

url = 'https://ai4crophealth.or.tz/rice/detect/'
# url  = 'http://127.0.0.1:8000/rice/detect/'
file_path = './rice.jpg'

with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files, data={'user_id': 2}, verify=False)
    print(response.json())