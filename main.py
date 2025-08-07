from fastapi import FastAPI,UploadFile,File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from io import BytesIO

app=FastAPI()
ml=load_model("waste_management.h5")

def read_img(file)->Image.Image:
    return Image.open(BytesIO(file)).convert("RGB")

@app.get("/")
def root():
    return {"msg":"vanakam da mapla"}

@app.post("/predict")
def predict(file:UploadFile=File(...)):
    img=read_img(file.file.read())
    img=img.resize((224,224))
    img_dim=image.img_to_array(img)
    img_dim=np.expand_dims(img_dim,axis=0)/225.0
    prd=ml.predict(img_dim)
    class_name=["Non-Recyclable","Organic","Recyclable"]
    res=np.argmax(prd)
    return {"predicted_type":class_name[res]}
