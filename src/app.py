from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from src.inference import predict

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    label = predict("D:/MlOps/Cifar10/checkpoint/model1.pth", image)
    return {"class": label}