import requests

# url = 'https://portal.ai4crophealth.or.tz/maize/detect/'
# # url = 'http://127.0.0.1:8000/maize/detect/'
# file_path = './maize.JPG'

# with open(file_path, 'rb') as image_file:
#     files = {'file': image_file}
#     data = {'user_id': 2}
#     response = requests.post(url, files=files, data=data, verify=False)

#     print("Response: ", response)
#     try:
#         print(response.json())
#     except ValueError:
#         print("Response content is not in JSON format")



url  = 'http://127.0.0.1:8000/maize/predict/'
file_path = './maize.JPG'

with open(file_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files, verify=False)
    print(response.json())