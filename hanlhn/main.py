from fastapi import FastAPI
from routes.flower import flower

app = FastAPI()
app.include_router(flower)