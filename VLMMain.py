import os
import json
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Configure your permanent API key (ideally load this from an .env file)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI()


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
        # Read the raw image bytes directly from the uploaded file
        image_bytes = await file.read()
        
        # Format it into the blob structure the Gemini SDK expects
        image_blob = {
            "mime_type": file.content_type,
            "data": image_bytes
        }

        # Initialize the model
        model = genai.GenerativeModel("gemini-2.5-flash")

        # 2. The Strict System Prompt
        prompt = """
        You are a highly precise food classification engine. Analyze the image.
        
        CRITICAL RULES:
        1. VALIDATION: First, determine if there is actually food in the image. Set 'food_detected' to true if food is present, or false if it is not.
        2. CERTAINTY: If food_detected is true, list only the items you are absolutely certain exist. Do not guess. If false, leave the 'detected_items' list empty.
        3. ATOMIC DECONSTRUCTION: Break down complex meals into single, fundamental ingredients or base items. 
           (e.g., If you see spaghetti with meat sauce, return separate items for 'spaghetti', 'ground beef', and 'tomato sauce').
        4. NO DESCRIPTORS: Never use connecting words like 'with', 'and', 'or', 'fried', 'baked'. Return only the core noun.
           (e.g., Return 'chicken', NOT 'fried chicken'. Return 'cheese', NOT 'melted cheese').
        """

        
        response = model.generate_content(
            [prompt, image_blob],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DetectionResult,
                temperature=0.1, # Extremely low temperature to prevent hallucination/guessing
            )
        )

        # The response.text is guaranteed to be a string matching your Pydantic schema
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#uvicorn VLMMain:app --port 8080 --reload