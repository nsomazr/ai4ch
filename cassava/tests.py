import requests

# url = 'https://ai4crophealth.or.tz/cassava/predict/'
url  = 'http://127.0.0.1:8000/cassava/detect/'
file_path = './cassava.JPG'

with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files, data={'user_id': 2}, verify=False)
    print(response.json())