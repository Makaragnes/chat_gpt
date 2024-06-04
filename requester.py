import requests

url = 'http://localhost:5000/generate'
data = {'prompt': 'Hello, how are you?', 'length': 50}

response = requests.post(url, json=data).json()
print(response['response'])