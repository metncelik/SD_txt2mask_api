from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
import requests
import cv2 
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from io import BytesIO
from torch import autocast
from transformers import CLIPSegTextConfig, CLIPSegTextModel
import base64

# Initializing a CLIPSegTextConfig with CIDAS/clipseg-rd64 style configuration
configuration = CLIPSegTextConfig()

# Initializing a CLIPSegTextModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
model = CLIPSegTextModel(configuration)

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('/content/clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False);

def get_mask(url):
    # or load from URL...
    image_url = url 
    input_image = Image.open(requests.get(image_url, stream=True).raw)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    img = transform(input_image).unsqueeze(0)
    prompts = ['cloth,body']

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
    filename = f"mask.png"
    plt.imsave(filename,torch.sigmoid(preds[0][0]))

    img2 = cv2.imread(filename)

    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # For debugging only:
    cv2.imwrite(filename,bw_image)

    masked_image = open(filename, "rb")

    b64_string = base64.b64encode(masked_image.read()).decode("utf_8")

    print(b64_string)


class Model(BaseModel):
    url:str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server on..."}

@app.post("/post/")
async def post(req:Model):
    b64_string = get_mask(req.url)
    return b64_string

uvicorn.run(app, host="0.0.0.0", port=8000)