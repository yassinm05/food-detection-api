from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Global Food Detection API", description="YOLOv12 Food Classification")

# Load your globally adaptive YOLO model
model = YOLO("best.pt")

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Run inference
        results = model.predict(image, conf=0.30, iou=0.45, verbose=False)
        result = results[0] 

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # 1-indexed UEC ID to perfectly match the .NET SQL database
            uec_id = class_id + 1 
            
            # We only return the ID and Name now
            detection_dict = {
                "class_id": uec_id,
                "class_name": class_name
            }
            
            # Prevent duplicate entries of the same food in a single image
            if detection_dict not in detections:
                detections.append(detection_dict)

        return {
            "food_detected": len(detections) > 0,
            "detected_items": detections
        }

    except Exception as e:
        return {"error": str(e)}