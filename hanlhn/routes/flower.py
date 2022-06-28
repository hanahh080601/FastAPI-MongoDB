from fastapi import APIRouter, Form, UploadFile
from PIL import Image
import io
import torch
from prediction.predict import Prediction
from models.flower import Flower
from schemas.flower import flowersEntity
from config.mongodb import conn

flower = APIRouter()

@flower.get('/')
async def list_all_flowers():
    print(conn.hanlhn.flowers.find())
    print(flowersEntity(conn.hanlhn.flowers.find()))
    return flowersEntity(conn.hanlhn.flowers.find())

@flower.post('/')
async def predict_flower(file: UploadFile=Form(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prediction = Prediction(device)
        pred = prediction.predict(img)
        pred_msg = {
            "Predict": str(pred)
        }
    except Exception as e:
        print(e)
    finally:
        await file.close()
    return pred_msg


