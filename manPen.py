import requests
import os
import json

# Replace with the actual IIIF manifest URL
manifest_url = 'https://colenda.library.upenn.edu/items/ark:/81431/p3rs2h/manifest'

# Create a directory to save images
os.makedirs('manuscript_images', exist_ok=True)

# Fetch the manifest
response = requests.get(manifest_url)
manifest = response.json()

# Iterate through canvases and download images
for i, canvas in enumerate(manifest['sequences'][0]['canvases']):
    image_service = canvas['images'][0]['resource']['service']['@id']
    full_res_url = f"{image_service}/full/full/0/default.jpg"

    image_response = requests.get(full_res_url)
    
    # Save the image
    with open(f'manuscript_images/page_{i+1}.jpg', 'wb') as f:
        f.write(image_response.content)
    print(f'Downloaded page {i+1}')
