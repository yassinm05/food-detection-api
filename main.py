import os
import json
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv 


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI(title="DiaMate Global Food Detection API", description="YOLOv12 + Gemini 2.5 Flash Pipeline")


model = YOLO("best.pt")


class FoodItem(BaseModel):
    class_name: str

class DetectionResult(BaseModel):
    food_detected: bool
    detected_items: list[FoodItem]

@app.post("/detect-food")
async def detect_food(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        
        results = model.predict(image, conf=0.30, iou=0.45, verbose=False)
        
      
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes.xyxy:
            
            x1, y1, x2, y2 = box.tolist()
           
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
       
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        vlm_image_bytes = img_byte_arr.getvalue()

        
        image_blob = {
            "mime_type": "image/jpeg",
            "data": vlm_image_bytes
        }

       
        vlm_model = genai.GenerativeModel("gemini-2.5-flash")

        
        prompt = """
        You are the core intelligence of a global dietary engine. Analyze the image, paying special attention to the areas enclosed in red bounding boxes.
        
        CRITICAL RULES:
        1. VALIDATION: First, determine if there is actually food in the image. Set 'food_detected' to true if food is present, or false if it is not.
        2. CERTAINTY: If food_detected is true, list only the items you are absolutely certain exist inside or around the red bounding boxes. Do not guess. If false, leave the 'detected_items' list empty.
        3. ATOMIC DECONSTRUCTION: Break down complex meals into single, fundamental ingredients or base items. 
           (e.g., If you see spaghetti with meat sauce, return separate items for 'spaghetti', 'ground beef', and 'tomato sauce').
        4. NO DESCRIPTORS: Never use connecting words like 'with', 'and', 'or', 'fried', 'baked'. Return only the core noun.
           (e.g., Return 'chicken', NOT 'fried chicken'. Return 'cheese', NOT 'melted cheese').
        """
        
        response = vlm_model.generate_content(
            [prompt, image_blob],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DetectionResult,
                temperature=0.1, # Keep it extremely low to prevent hallucination
            )
        )

        # 7. Return the final JSON to the mobile app
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))