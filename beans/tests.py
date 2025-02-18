import requests

url = 'https://portal.ai4crophealth.or.tz/beans/detect/'
# url  = 'http://127.0.0.1:8000/beans/detect/'
file_path = './bean.jpeg'

with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files, data={'user_id': 2}, verify=False)
    print(response.json())