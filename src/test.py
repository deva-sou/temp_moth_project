import os
from PIL import Image
from io import BytesIO
import requests
import pandas as pd

'''
Todo:
- Automated check if image exist
- Maybe create the name of folder from the scientific name
'''
print('test')
d_path = '/home/deva/code/temp/fielguide_images/'
df = pd.read_csv('Fieldguide Lep Species Example.csv')
image_urls = df['image_url']
print(image_urls)
import sys
print(sys.path)
""" for image_url in image_urls:
    # example : http://production-chroma.s3.amazonaws.com/photos/5cf6d953fe9c0e088df4243e/c51d6ab9ce4846e99a00bffdef7abd8b.jpg
    category = image_url.split('/')[-2]
    image_name = image_url.split('/')[-1]
    result = f"{d_path}{category}"
    if not os.path.exists(result):
        os.makedirs(result)
        Image.open(BytesIO(requests.get(image_url).content)).save(f'{result}/{image_name}')
    if os.path.exists(result):
        Image.open(BytesIO(requests.get(image_url).content)).save(f'{result}/{image_name}')

 """