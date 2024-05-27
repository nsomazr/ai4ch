import requests

url = 'https://ai4crophealth.or.tz/maize/predict/'
file_path = './maize.JPG'

with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())