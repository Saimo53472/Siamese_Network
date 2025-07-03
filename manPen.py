import requests
import os
import json

# Downloads all page images of Ricettario manuscript into 'manuscript_images' folder.
manifest_url = 'https://colenda.library.upenn.edu/items/ark:/81431/p3rs2h/manifest'

os.makedirs('manuscript_images', exist_ok=True)

response = requests.get(manifest_url)
manifest = response.json()

for i, canvas in enumerate(manifest['sequences'][0]['canvases']):
    image_service = canvas['images'][0]['resource']['service']['@id']
    full_res_url = f"{image_service}/full/full/0/default.jpg"

    image_response = requests.get(full_res_url)
    
    with open(f'manuscript_images/page_{i+1}.jpg', 'wb') as f:
        f.write(image_response.content)
    print(f'Downloaded page {i+1}')