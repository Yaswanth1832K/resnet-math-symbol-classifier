import torch
from torchvision import models, transforms
from PIL import Image
import json
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("models/classes_v3.json", "r") as f:
    class_names = json.load(f)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/resnet18_math_symbols_v3.pth", map_location=device))
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

canvas = np.full((300, 800, 3), 255, dtype=np.uint8)
cv2.line(canvas, (300, 100), (500, 100), (0,0,0), 6) # top bar
cv2.line(canvas, (350, 100), (350, 200), (0,0,0), 6) # left leg
cv2.line(canvas, (450, 100), (450, 200), (0,0,0), 6) # right leg

gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = [b for b in bounding_boxes if b[2] * b[3] > 50]

for (x, y, w, h) in bounding_boxes:
    # Tight padding
    pad = 5
    y1, y2 = max(0, y - pad), min(canvas.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(canvas.shape[1], x + w + pad)
    symbol_crop = canvas[y1:y2, x1:x2]
    
    # Square with tight margin
    side = max(symbol_crop.shape[0], symbol_crop.shape[1])
    final_side = int(side * 1.1)
    squared = np.full((final_side, final_side, 3), 255, dtype=np.uint8)
    y_off = (final_side - symbol_crop.shape[0]) // 2
    x_off = (final_side - symbol_crop.shape[1]) // 2
    squared[y_off:y_off+symbol_crop.shape[0], x_off:x_off+symbol_crop.shape[1]] = symbol_crop
    
    cv2.imwrite("test_synthetic_crop_tight.jpg", squared)
    print("Prediction with tight padding:", predict_cv2(squared))

    # And with 1.5 padding to contrast
    final_side_15 = int(side * 1.5)
    squared_15 = np.full((final_side_15, final_side_15, 3), 255, dtype=np.uint8)
    y_off = (final_side_15 - symbol_crop.shape[0]) // 2
    x_off = (final_side_15 - symbol_crop.shape[1]) // 2
    squared_15[y_off:y_off+symbol_crop.shape[0], x_off:x_off+symbol_crop.shape[1]] = symbol_crop
    print("Prediction with 1.5 padding:", predict_cv2(squared_15))
