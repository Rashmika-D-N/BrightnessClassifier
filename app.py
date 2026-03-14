import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image, ImageStat
from model import get_model
import os

app = FastAPI(title="Brightness Classifier API")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=2)

MODEL_PATH = 'model.pth'
if os.path.exists(MODEL_PATH):
    # To handle loading a model trained on a machine with/without GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded model weights.")
else:
    print("Warning: model.pth not found. The model is untrained!")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ['dark', 'very_dark']

class ClassificationResponse(BaseModel):
    category: str
    image_base64: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return "<h1>Index.html not found!</h1>"

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    # Check overall brightness using relative luminance approximation
    grayscale_image = image.convert('L')
    stat = ImageStat.Stat(grayscale_image)
    avg_brightness = stat.mean[0]
    
    # Threshold for a bright image (0-255 scale)
    if avg_brightness > 80:
        # User requested: Bright images should just be labeled as 'dark'
        category = 'dark'
    else:
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)
            category = classes[predicted.item()]
        
    # Base64 encode original image
    encoded_string = base64.b64encode(contents).decode('utf-8')
    mime_type = file.content_type
    image_base64 = f"data:{mime_type};base64,{encoded_string}"
    
    return ClassificationResponse(
        category=category,
        image_base64=image_base64
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
