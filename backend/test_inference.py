import torch
from torchvision import models, transforms
from PIL import Image
import json
import cv2
import numpy as np

MODEL_PATH = "models/resnet18_math_symbols_v3.pth"
CLASSES_PATH = "models/classes_v3.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_cv2(img_cv2):
    img_tensor = val_transforms(img_cv2).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        return class_names[preds[0]]

# Test 1: Real dataset image pi
pi_real = cv2.imread(r"data_split_full\val\pi\pi_101113.jpg")
print("Prediction on real pi dataset image:", predict_cv2(pi_real))

# Test 2: Let's create a synthetic blank image with a "pi" drawn like the user:
canvas = np.full((300, 800, 3), 255, dtype=np.uint8)
cv2.line(canvas, (300, 100), (500, 100), (0,0,0), 6) # top bar
cv2.line(canvas, (350, 100), (350, 200), (0,0,0), 6) # left leg
cv2.line(canvas, (450, 100), (450, 200), (0,0,0), 6) # right leg

def segment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [b for b in bounding_boxes if b[2] * b[3] > 50]
    
    symbols = []
    for (x, y, w, h) in bounding_boxes:
        pad = max(5, int(0.05 * max(w, h)))
        y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
        symbol_crop = image[y1:y2, x1:x2]
        side = max(symbol_crop.shape[0], symbol_crop.shape[1])
        final_side = int(side * 1.5)
        squared = np.full((final_side, final_side, 3), 255, dtype=np.uint8)
        y_off = (final_side - symbol_crop.shape[0]) // 2
        x_off = (final_side - symbol_crop.shape[1]) // 2
        squared[y_off:y_off+symbol_crop.shape[0], x_off:x_off+symbol_crop.shape[1]] = symbol_crop
        symbols.append(squared)
    return symbols

synthetic_crops = segment(canvas)
for i, sc in enumerate(synthetic_crops):
    cv2.imwrite(f"test_synthetic_crop_{i}.jpg", sc)
    print("Prediction on synthetic crop:", predict_cv2(sc))

