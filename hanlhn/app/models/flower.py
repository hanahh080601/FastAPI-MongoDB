from pydantic import BaseModel

class Flower(BaseModel):
    class_index: str 
    eng_name: str
    vie_name: str
    description: str 