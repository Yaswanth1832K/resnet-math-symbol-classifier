# 🧮 Handwritten Math Recognition  
### **Full-Stack Application (Vite + React & FastAPI + ResNet-18)**

This is a complete, modernized handwritten math symbol recognition system. This application allows users to draw or upload images of extremely complex mathematical structures (including Greek letters, integral signs, sums, and trigonometry equations) to automatically translate them into functional **LaTeX Code** for easy documentation.

The project originally started as a localized Jupyter Notebook using HOG/SVM classical models, but has been entirely rewritten from the ground up to utilize a state-of-the-art **82-Class ResNet-18 Convolutional Neural Network**, wrapped in a sleek **Vite/React** interface and backed by an asynchronous **FastAPI** server.

---

# 🚀 Features

## ✔ Frontend Web Application (Vite + Tailwind + React)
- **High-DPI Drawing Canvas** with Undo/Eraser states.
- **Drag-and-Drop Image Uploader** for instant evaluation of pictures.
- **Beautiful Dark-Mode UI** featuring glassmorphism, glowing gradients, and responsive layouts.
- **Dynamic LaTeX Rendering** engine that immediately evaluates physics/math formulas on-screen.

## ✔ Backend Neural API (FastAPI + PyTorch)
- Asynchronous python web server evaluating image POST requests in under **100ms**.
- **Adaptive Image Preprocessing Pipeline** that cleans images using Gaussian Blur, OTSU Thresholding, Dilations, and Contour mapping to perfectly frame symbols before predicting. 
- Custom algorithms that intelligently join 82 distinct PyTorch categories into perfectly concatenated string formats (e.g. bridging `\int` and `\pi`).

## ✔ Deep Learning Model (ResNet-18)
- Upgraded the legacy architecture to exclusively use Deep Learning with transfer learning from ImageNet.
- Trained against a massive **370,000-image dataset** across **82 unique math/physics classes**.
- Attains extremely robust intelligence with a staggering **96.65% Validation Accuracy**.

---

# 🤖 Supported Symbols (82 Classes)

The neural network has been fully trained to recognize:
- **Digits**: `0` to `9`
- **Arithmetic**: `+  -  ×  ÷  =  .`
- **Greek Letters**: `\alpha`, `\beta`, `\gamma`, `\Delta`, `\theta`, `\lambda`, `\mu`, `\pi`, `\sigma`, `\phi`
- **Advanced Calculus**: Integrals (`\int`), limits (`\lim`), infinity (`\infty`), sums (`\sum`), square roots (`\sqrt`), logs (`\log`)
- **Trigonometry**: `\cos`, `\sin`, `\tan`
- **Comparisons & Syntax**: Less than (`<`), Greater than (`>`), Belongs to (`\in`), Prime marks (`'`), all brackets `{}[]()`, and alphabetical variables `x, y, z`, etc.

---

# ⚙️ Local Installation & Execution

### 1. Backend Server Setup
To execute the neural network API:
```bash
# Navigate to the backend
cd backend

# Install dependencies
pip install fastapi uvicorn python-multipart torch torchvision opencv-python numpy

# Run the API
python app.py
```
*(The server will boot up and host the inference API on `http://localhost:8000`)*

### 2. Frontend React Setup
To execute the frontend:
```bash
# Navigate to the frontend
cd frontend

# Install Node dependencies
npm install

# Start the dev environment
npm run dev
```
*(The website will boot up on `http://localhost:5173`. Simply go to the link to play with the application.)*

---


⭐ *If this project helped you, consider starring the repository!* ⭐
