from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
import platform
import os
import json

app = FastAPI(title="Math Symbol Classifier API")

# Allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/resnet18_math_symbols_v3.pth"
CLASSES_PATH = "models/classes_v3.json"

# Fallback to v1 if v2 doesn't exist yet
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/resnet18_math_symbols.pth"
    CLASSES_PATH = "models/classes.json"

# Load the model and classes once on startup
# We initialize it to None and load it on the first request to avoid crashing 
# if the model hasn't been trained yet (e.g., during setup).
model = None
class_names = []

def load_system():
    global model, class_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("Warning: Model or classes.json not found! Please run train.py first.")
        return False
        
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return True

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def segment_image(image):
    """
    Very basic contour-based symbol segmentation.
    Finds bounding boxes from left to right.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize image to white background, black ink.
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
        image = cv2.bitwise_not(image)
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate a bit to connect disconnected parts of a single symbol like '=' or '÷'
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Filter out tiny specks of noise
    bounding_boxes = [b for b in bounding_boxes if b[2] * b[3] > 50]
    
    # Sort boxes from left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    symbols = []
    
    for (x, y, w, h) in bounding_boxes:
        pad = max(5, int(0.05 * max(w, h)))
        y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
        
        symbol_crop = image[y1:y2, x1:x2]
        
        # We need to make it square and add tight padding to match dataset distribution
        side = max(symbol_crop.shape[0], symbol_crop.shape[1])
        final_side = int(side * 1.1)
        squared = np.full((final_side, final_side, 3), 255, dtype=np.uint8) # white background
        
        y_off = (final_side - symbol_crop.shape[0]) // 2
        x_off = (final_side - symbol_crop.shape[1]) // 2
        squared[y_off:y_off+symbol_crop.shape[0], x_off:x_off+symbol_crop.shape[1]] = symbol_crop
        
        symbols.append((squared, (x, y, w, h)))
        
    return symbols

def to_latex(predictions):
    mapping = {
        'times': '\\times',
        'div': '\\div',
        'add': '+',
        'sub': '-',
        'eq': '=',
        'dec': '.',
        'Delta': '\\Delta',
        'alpha': '\\alpha',
        'beta': '\\beta',
        'gamma': '\\gamma',
        'sigma': '\\sigma',
        'phi': '\\phi',
        'theta': '\\theta',
        'lambda': '\\lambda',
        'mu': '\\mu',
        'pi': '\\pi',
        'sqrt': '\\sqrt',
        'sum': '\\sum',
        'int': '\\int',
        'infty': '\\infty',
        'in': '\\in',
        'geq': '\\geq',
        'leq': '\\leq',
        'gt': '>',
        'lt': '<',
        'neq': '\\neq',
        'log': '\\log',
        'lim': '\\lim',
        'cos': '\\cos',
        'sin': '\\sin',
        'tan': '\\tan',
        'ldots': '\\dots',
        'forall': '\\forall',
        'exists': '\\exists',
        'rightarrow': '\\rightarrow',
        'pm': '\\pm',
        'prime': "'",
        'forward_slash': '/',
        'ascii_124': '|',
        'COMMA': ',',
        '{': '\\{',
        '}': '\\}'
    }
    latex = []
    for p in predictions:
        if p in mapping:
            latex.append(mapping[p])
        elif len(p) == 1:
            latex.append(p)
        else:
            latex.append(p)
    return " ".join(latex)

@app.get("/api/health")
async def health():
    status = "ready" if (model is not None or load_system()) else "waiting_for_model"
    return {"status": status, "device": str(device)}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        success = load_system()
        if not success:
            return {"error": "Model not trained yet."}
            
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image file."}
        
    symbols = segment_image(img)
    if not symbols:
        return {"error": "No symbols detected in the image."}
        
    predictions = []
    symbol_results = []
    
    with torch.no_grad():
        for sym_img, (x, y, w, h) in symbols:
            tensor = val_transforms(sym_img).unsqueeze(0).to(device)
            outputs = model(tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds.item()]
            predictions.append(predicted_class)
            
            # encode sym_img as base64 so we can show it on frontend
            _, buffer = cv2.imencode('.png', sym_img)
            import base64
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            symbol_results.append({
                "class": predicted_class,
                "box": {"x": x, "y": y, "w": w, "h": h},
                "image_base64": img_b64
            })
            
    latex_string = to_latex(predictions)
    
    return {
        "latex": latex_string,
        "predictions": symbol_results,
        "raw_classes": predictions
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
