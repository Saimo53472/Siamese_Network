import requests
import os

manifest_url = 'https://collections.library.yale.edu/manifests/11625507'
os.makedirs('manuscript_images2', exist_ok=True)

response = requests.get(manifest_url)
manifest = response.json()

for i, canvas in enumerate(manifest['items']):
    annotation_page = canvas['items'][0]
    annotation = annotation_page['items'][0]
    body = annotation['body']

    image_service_id = body['service'][0]['@id']

    full_res_url = f"{image_service_id}/full/full/0/default.jpg"

    image_response = requests.get(full_res_url)
    image_response.raise_for_status() 

    filename = f"manuscript_images2/page_{i+1}.jpg"
    with open(filename, 'wb') as f:
        f.write(image_response.content)
    print(f"Downloaded {filename}")
