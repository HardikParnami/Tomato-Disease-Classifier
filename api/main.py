from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
from tensorflow import keras

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = keras.models.load_model("../models_tomato/1.keras")



CLASSNAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

#def read_file_as_image(data)->np.ndarray:
   # image= Image.open(BytesIO(data))
  #  return image
def read_file_as_image(data) -> np.ndarray:
    image =np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch=np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    np.argmax(predictions[0])
    predicted_class = CLASSNAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class':predicted_class ,
        ' confidence':float(confidence)
    }

if __name__ =="__main__":
    uvicorn.run(app, host='localhost', port =8000)    