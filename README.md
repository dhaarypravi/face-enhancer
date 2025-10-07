# Face Enhancer (Image Restoration using OpenCV & Super-Resolution)

## 🧠 Overview
This project is a **Face Enhancer tool** built with **Python and OpenCV**.  
It takes a **blurred image** as input and produces a **sharper, high-definition (HD) version** as output.  

The enhancement pipeline uses:
- **Denoising**
- **Sharpening**
- **Contrast enhancement (CLAHE)**
- **Super-resolution model (ESPCN)** for 4× HD upscaling

---

## 🧰 Tech Stack
- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy  

---

## 📦 Folder Structure
FaceEnhancer/
│
├── face_enhancer_hd.py # Main Python script
├── blurred_face.jpg # Input image
├── enhanced_face_hd.jpg # Output image (generated)
├── README.md # Project description

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Install Dependencies
```bash
pip install opencv-python opencv-contrib-python numpy
from OpenCV’s official model zoo:
https://github.com/fannymonori/TF-ESPCN

Place it inside your project folder.

🚀 Run the Project
bash
Copy code
python face_enhancer_hd.py
Input:
A blurred image (e.g. blurred_face.jpg)

Output:
An HD enhanced image saved as enhanced_face_hd.jpg

🧩 Working Steps
Load image

Denoise using fastNlMeansDenoisingColored

Sharpen edges using weighted Gaussian subtraction

Enhance contrast using CLAHE

Upscale image using ESPCN Super-Resolution model

Save & display the enhanced HD result

📊 Results Example
Original Image	Enhanced HD Image

🧾 Instructions for Submission
Include the following in your ZIP file:

Source code (.py file)

Input & output images


This README.md file

Short video demo showing before & after

👩‍💻 Author
Khushi Pandey
B.Tech (Computer Science Engineering)
Email: pandeykhushhi@gmail.com
LinkedIn: linkedin.com/in/khushi-pandey-9a441625b


Recommended Python version: 3.8+

For best results, use high-resolution or close-up face images.

