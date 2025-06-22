from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image, ImageOps
import io
import os
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Tambahkan semua class yang umum dipakai oleh YOLOv8
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from torch.nn.modules.container import Sequential
from torch.nn import Conv2d, BatchNorm2d
from torch.nn.modules.activation import SiLU

# Register safe globals
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Conv,
    C2f,
    Conv2d,
    BatchNorm2d,
    SiLU,
])


load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
CSV_PATH = os.getenv("CSV_PATH")
print("MODEL_PATH:", os.getenv("MODEL_PATH"))
print("CSV_PATH:", os.getenv("CSV_PATH"))
print("Current Working Directory:", os.getcwd())

app = FastAPI()

# Konfigurasi path
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti jika ingin lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan data nutrisi
try:
    model = YOLO(MODEL_PATH)
    nutrition_df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"Error loading model or nutrition data: {e}")
    model = None
    nutrition_df = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize ke 640x640 sesuai input model
        image = ImageOps.pad(image, (640, 640), color=(114, 114, 114))

        # Inference
        results = model(image)
        detections = results[0].boxes.data.tolist()

        output = []

        for box in detections:
            x1, y1, x2, y2 = [round(v, 2) for v in box[:4]]
            confidence = round(float(box[4]), 2)
            class_id = int(box[5])

            if nutrition_df is not None and 0 <= class_id < len(nutrition_df):
                nutrition = nutrition_df.iloc[class_id]
                class_name = nutrition["class"]

                output.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "nutrition": {

                        "Calories (kcal)": int(nutrition["Calories (kcal)"]),
                        "Protein (g)": float(nutrition["Protein (g)"]),
                        "Carbohydrates (g)": float(nutrition["Carbohydrates (g)"]),
                        "Fat (g)": float(nutrition["Fat (g)"]),
                        "Fiber (g)": float(nutrition["Fiber (g)"])
}

                })
            else:
                output.append({
                    "class_id": class_id,
                    "class_name": f"class_{class_id}",
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "nutrition": None
                })

        return {"detections": output}

    except Exception as e:
        return {"error": str(e)}