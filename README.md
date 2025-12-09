# 🧮 Math OCR Symbol Recognizer  
### **Hybrid Classical + Deep Learning (ResNet) Approach**

This project implements a **complete handwritten math symbol recognition system** using two machine-learning pipelines:

1. **Classical ML Model:** HOG + LBP Feature Extraction + SVM  
2. **Deep Learning Model:** ResNet-18/34 with Transfer Learning  

The system supports symbol segmentation, data augmentation, collage test-image generation, LaTeX output, and full evaluation tools—making it an **end-to-end Math OCR engine**.

---

# 📂 Dataset

This project uses the **Handwritten Math Symbols Dataset** from Kaggle:

🔗 Dataset: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols  
**Author:** Sagyam Thapa  

### Dataset Includes:
- 10,000+ grayscale handwritten symbol images  
- **Digits:** `0–9`  
- **Operators:** `+  −  ×  ÷  =  .`  
- **Variables:** `x  y  z`  
- Preprocessed: resized to 64×64, normalized, and label-encoded  

---

# 🚀 Project Features

## ✔ Classical OCR Model (HOG + LBP + SVM)
- Custom image augmentation (rotation, shift, scaling)  
- HOG-based shape features  
- LBP-based texture features  
- Lightweight SVM classifier  
- Interpretable, traditional CV approach  

---

## ✔ Deep Learning Model (ResNet-18 / ResNet-34)
- Transfer learning from ImageNet  
- Modified final FC layer for math symbol classification  
- Data augmentation for robust training  
- Supports training, evaluation, and inference on custom images  

### **ResNet-18 Performance**
| Metric | Score |
|--------|--------|
| Validation Accuracy | **96.40%** |
| Test Accuracy | **96.60%** |
| Precision / Recall / F1 | **0.97** |

ResNet provides **excellent generalization**, robust feature extraction, and stable deep learning performance due to residual connections.

---

## ✔ Symbol Segmentation Module
- Segments handwritten equations into individual symbols  
- Works for fixed-width synthetic collages  
- Outputs symbol images for classification  

---

## ✔ Synthetic Test Collage Generator
Automatically creates realistic test samples:

- Randomly selects **3–5 symbols**
- Concatenates them horizontally  
- Saves the image + ground truth labels  
- Useful for benchmarking both models  

---

## ✔ Evaluation Tools
- Predict symbols using Classical or ResNet model  
- Visualize segmented symbols with predicted labels  
- Convert predicted symbols into **LaTeX expression format**  
- Compare accuracy between the two models  

---

# 🧠 Classical OCR Workflow

### **1. Load + Augment Dataset**
```python
ocr = AugmentedFeatureOCR()
features, labels = ocr.load_and_augment_dataset(data_dir, max_samples=4000)
```

### **2. Train SVM Model**
```python
ocr.train_classifier(features, labels)
```

### **3. Segment Image into Symbols**
```python
symbols = ocr.segment_symbols_fixed("test.jpg", n_symbols=4)
```

### **4. Predict + Generate LaTeX**
```python
predicted = ocr.classify_symbols(symbols)
latex = ocr.to_latex(predicted)
```

---

# 🤖 ResNet Model Workflow

### **Train ResNet**
```bash
python train_resnet.py --data dataset/ --epochs 20
```

### **Evaluate ResNet**
```bash
python evaluate.py --weights model.pth
```

---

# 🖼 Generate Test Collage Images
```python
test_images_info = generate_test_collages_variable_length(
    data_dir,
    class_folders,
    n_test=50,
    min_symbols=3,
    max_symbols=5
)
```

---

# 🔍 Test All Images
```python
test_all_images(ocr, test_images_info)
```

Outputs include:
- Ground truth  
- Predicted sequence  
- Generated LaTeX  
- Visualized segmentation + classification  

---

# 🧪 Supported Symbols

| Category | Symbols |
|----------|---------|
| **Digits** | `0 1 2 3 4 5 6 7 8 9` |
| **Operators** | `+  -  ×  ÷  =  .` |
| **Variables** | `x  y  z` |

---

# 🛠 Tech Stack

### **Classical Model**
- Python  
- OpenCV  
- NumPy  
- scikit-learn  
- scikit-image  
- Matplotlib  

### **Deep Learning Model**
- PyTorch / TensorFlow  
- ResNet-18 / ResNet-34  
- GPU-supported training  

---

# 📈 Future Enhancements
- Sequence-to-sequence (Seq2Seq) for entire expression recognition  
- Attention-based segmentation  
- End-to-end neural OCR pipeline  
- Support for hand-drawn real-world equations  
- Transformer-based symbol recognition  

---

⭐ *If this project helped you, consider starring the repository!* ⭐
