import requests

image_path = 'kura_2.png'

# Create a dictionary with the image file
files = {'image': open(image_path, 'rb')}

# Make a POST request to the /scan endpoint
response = requests.post('http://127.0.0.1:5000/scanner', files=files)
# Check the response
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print('Request failed with status code:', response.status_code)    
