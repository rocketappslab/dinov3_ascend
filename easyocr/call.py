
import requests

url = "http://127.0.0.1:4443/ocr"
payload = {"image_path": "/home/ei/workspace/rocket/Project/nanfang_capsule/data/unzipped/v1/barrett's_esophagus/patient8/lesion_context/231_091601.jpg"}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
response = response.json()
print(response.get('count'))

