import requests

headers = {
    'Content-Type': 'application/json',
}

json_data = {[104, 2, 1015, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
response = requests.post('http://127.0.0.1:5000/invocations', headers=headers, json=json_data)
print(response.text)