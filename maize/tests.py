import requests

# url = 'https://portal.ai4crophealth.or.tz/maize/detect/'
url = 'http://127.0.0.1:8000/maize/detect/'
file_path = './maize.JPG'

with open(file_path, 'rb') as image_file:
    files = {'file': image_file}
    data = {'user_id': 2}
    response = requests.post(url, files=files, data=data, verify=False)

    print("Response: ", response)
    try:
        print(response.json())
    except ValueError:
        print("Response content is not in JSON format")



# url  = 'http://127.0.0.1:8000/maize/predict/'
# file_path = './maize.JPG'

# with open(file_path, 'rb') as image_file:
#     files = {'image': image_file}
#     response = requests.post(url, files=files, verify=False)
#     print(response.json())


# First, get token
# login_url = 'http://127.0.0.1:8000/api/token/'
# credentials = {
#     'username': 'your_username',
#     'password': 'your_password'
# }
# token_response = requests.post(login_url, data=credentials)
# token = token_response.json()['access']

# # Then use token in your request
# url = 'http://127.0.0.1:8000/maize/detect/'
# headers = {
#     'Authorization': f'Bearer {token}'
# }

# with open(file_path, 'rb') as image_file:
#     files = {'file': image_file}
#     data = {'user_id': 2}
#     response = requests.post(url, files=files, data=data, headers=headers, verify=False)