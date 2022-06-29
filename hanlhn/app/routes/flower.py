from fastapi import APIRouter, Form, UploadFile
from PIL import Image
import io
from schemas.flower import flowerEntity, flowersEntity
from config import conn
from prediction import prediction

flower = APIRouter()

@flower.get('/')
async def list_all_flowers():
    return flowersEntity(conn.hanlhn.flowers.find())

@flower.post('/')
async def predict_flower(file: UploadFile=Form(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        pred = prediction.predict(img)
    except Exception as e:
        print(e)
    finally:
        await file.close()
    return flowerEntity(conn.hanlhn.flowers.find_one({"English name":pred}))


