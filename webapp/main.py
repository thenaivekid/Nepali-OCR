from fastapi import FastAPI,HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import pipeline
import cv2
import numpy as np
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    # file_path = os.path.join("/images", file.filename)
    file_path = file.filename
    if file_path:
        # print(file_path)
        # with open(file_path, "wb") as f:
        #     f.write(contents)
        print(file.filename, "received")
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        text = str(pipeline(image))
        # return {"text": text, "image_url": str(file_path)}
        return {"text": text}
    else:
        raise HTTPException(status_code=404, detail="File not found")