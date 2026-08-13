
# 👥 CrowdGuard AI-Intelligent Crowd Management System

## AI-Powered Crowd Detection, Counting and Risk Analysis

The **CrowdGuard AI-Intelligence Crowd Management System** is an intelligent computer-vision application developed to monitor crowds, detect people, count individuals, and identify different levels of crowd risk.

The application is designed as a **Streamlit-based capstone project** under the **Artificial Intelligence Careers for Women (AICW)** program. It supports crowd analysis from **images, videos, and live camera input** and provides crowd-risk information to support safer public spaces.

---

## 🎯 Project Goal

The goal of this project is to provide an AI-powered solution for:

- Monitoring crowds
- Detecting people automatically
- Counting detected people
- Identifying potentially high-risk crowd situations
- Providing crowd-risk levels
- Monitoring crowd locations
- Supporting safer public spaces

---

## 🚨 Problem Statement

Monitoring large crowds manually can be difficult, especially in public places and situations where the number of people changes continuously.

Traditional monitoring depends heavily on human observation, which can make continuous crowd counting and early identification of crowded situations challenging.

This project addresses the problem by using **Artificial Intelligence and Computer Vision** to automatically analyze crowd images, videos, and live camera input.

---

## 💡 Motivation

Crowded situations can create safety concerns when the number of people becomes high or when crowd conditions change quickly.

The motivation of this project is to develop a simple and accessible AI-based system that can help monitor crowds and provide early indications of potentially risky crowd conditions.

---

## ✨ Key Features

### 📷 Image Analysis

- Upload crowd images
- Detect people using AI
- Count detected people
- Display crowd-risk level

### 🎥 Video Analysis

- Analyze crowd videos
- Detect people in video frames
- Count detected people
- Analyze crowd conditions

### 📹 Live Camera Monitoring

- Support live camera monitoring
- Real-time people detection
- Real-time people counting
- Crowd-risk analysis

### 🚨 Risk Detection

The system classifies crowd conditions into:

- 🟢 **LOW**
- 🟡 **MEDIUM**
- 🔴 **HIGH**

A high-crowd condition can generate a crowd alert.

### 📍 Location Monitoring

The application supports location-related information, including:

- GPS location support
- Crowd monitoring location
- Location information
- Alert location information

---

## 🏗️ System Workflow

```text
             IMAGE / VIDEO / LIVE CAMERA
                         │
                         ▼
                Video/Image Processing
                         │
                         ▼
                  YOLO Object Detection
                         │
                         ▼
                  Person Detection
                         │
                         ▼
                   People Counting
                         │
                         ▼
                  Crowd Risk Analysis
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             LOW       MEDIUM      HIGH
                                    │
                                    ▼
                              Crowd Alert
                                    │
                                    ▼
                           Location Information
```

---

## 🧠 How the System Works

### 1. Input

The user provides crowd data through:

- An uploaded image
- An uploaded video
- A live camera

### 2. AI-Based Detection

The system uses the **YOLO object-detection model** to identify people in the input.

### 3. People Counting

Detected people are counted automatically.

### 4. Crowd Risk Classification

Based on the crowd condition, the system displays one of three levels:

```text
LOW
MEDIUM
HIGH
```

### 5. Alert

When a high crowd condition is identified, the application can display a high-crowd alert.

### 6. Location Information

Location/GPS support can be used to associate monitoring and alert information with a location.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **YOLO** | AI-based object/person detection |
| **Computer Vision** | Image and video analysis |
| **Streamlit** | Web application interface |
| **OpenCV** | Image/video processing |
| **Streamlit-WebRTC** | Live camera/WebRTC support |
| **Streamlit-Folium** | Map and location visualization |
| **Streamlit-JS-Eval** | Browser-side JavaScript interaction |
| **Folium** | Interactive maps |
| **Pandas** | Data processing |
| **NumPy** | Numerical operations |

The project's dependency list includes Streamlit, Streamlit-WebRTC, Streamlit-Folium, Streamlit-JS-Eval, OpenCV, Ultralytics, AV, Folium, Pandas, and NumPy. 

---

## 📦 Project Requirements

### Hardware

Recommended:

- Laptop/Desktop computer
- Minimum 4 GB RAM
- Webcam for live-camera functionality
- Internet connection for installing Python packages

### Software

- Python 3.9+
- Git
- GitHub account
- Modern web browser

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub details.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project dependencies include:

```text
streamlit
streamlit-webrtc
streamlit-folium
streamlit-js-eval
opencv-python-headless
ultralytics
av
folium
pandas
numpy
```

### Step 4: Run the Application

If your main Streamlit file is named `app.py`:

```bash
streamlit run app.py
```

If your main file has a different name, replace `app.py` with the actual filename.

The application will open in the browser at the local Streamlit address shown in the terminal.

---

## 📁 Suggested Repository Structure

```text
AI-Crowd-Management-System/
│
├── README.md
├── requirements.txt
├── app.py
│
├── pages/
│   └── 1_Crowd_Management_Application.py
│
├── models/
│   └── YOLO model files
│
├── images/
│   └── project screenshots
│
└── assets/
    └── project resources
```

> Keep the structure consistent with the actual files in your repository.

---

## 🖥️ Application Interface

The application provides a Streamlit interface with:

- Project title
- Project description
- Team information
- Project application navigation
- Image analysis
- Video analysis
- Live camera monitoring
- Crowd-risk analysis
- Location monitoring

The home page is configured with the title **AI Crowd Management System** and a wide Streamlit layout.

---

## 👩‍💻 Project Team

### Artificial Intelligence Careers for Women (AICW)

**Capstone Project**

1. **D.L. Priyanka**  
   doddipatlapriyanka1@gmail.com

2. **R.S.V. Madhulika**  
   rayudusrividyamadhulika@gmail.com

3. **Y. Leela Devi**  
   leelayejarla@gmail.com

4. **K. Vyjayanthi**  
   vyshu366@gmail.com

---

## 🏫 Institution

**VSM College of Engineering**  
Ramachandrapuram

---

## 👨‍🏫 Project Guide

**Abdul Aziz Md**  
Lead - AICW (South)

---

## 📸 Project Screenshots

Add your project screenshots to the repository and place them in this section.

Example:

```markdown
![System Architecture](images/system_architecture.png)

![YOLO Person Detection](images/yolov8_detection.png)

![Crowd Counting](images/crowd_counting.png)

![Project Demonstration](images/demonstration.png)
```

---

## 🌍 Potential Applications

The system can be used as a foundation for crowd monitoring in:

- Railway stations
- Airports
- Shopping malls
- Stadiums
- Temples
- Public events
- Educational institutions
- Bus terminals
- Exhibition centers
- Other public gathering areas

---

## 🔮 Future Scope

Possible future improvements include:

- Multi-camera monitoring
- Improved crowd-density estimation
- Advanced crowd-flow analysis
- Heatmap generation
- SMS/email notifications
- Mobile application support
- Cloud-based monitoring
- Historical crowd analytics
- Emergency response integration
- Improved performance for large-scale deployments

---

## ⚠️ Limitations

- Detection performance can vary with image/video quality.
- Poor lighting may affect detection.
- Heavy crowd overlap or occlusion can reduce detection accuracy.
- Live-camera performance depends on the available hardware and network/browser environment.
- Crowd-risk thresholds should be configured according to the specific monitoring environment.

---

## 🔐 Privacy and Responsible Use

This project is intended for **crowd safety and monitoring purposes**.

When deployed with cameras in real environments, users should follow applicable privacy, surveillance, and data-protection requirements.

The system should preferably be used for:

- Crowd statistics
- People counting
- Crowd-risk monitoring
- Safety alerts
- Location-based monitoring

rather than unnecessary identification of individuals.

---

## 📌 Project Status

**Status:** Capstone Project

**Project:** AI Crowd Management System

**Program:** Artificial Intelligence Careers for Women (AICW)

---

## 📜 License

This project is developed for **educational and academic purposes**.

If you plan to distribute or reuse the source code publicly, add an appropriate open-source license such as the MIT License according to your project's requirements.

---

## ⭐ Conclusion

The **AI Crowd Management System** demonstrates how Artificial Intelligence and Computer Vision can be used to automatically monitor crowds, detect people, count individuals, classify crowd-risk levels, and provide location-related monitoring information.

By combining **Python, YOLO, Computer Vision, Streamlit, OpenCV, and supporting libraries**, the project provides a practical foundation for AI-assisted crowd monitoring and safer public spaces.

---

### Artificial Intelligence Careers for Women (AICW)

**Capstone Project • VSM College of Engineering**

**Guided by Abdul Aziz Md, Lead - AICW (South)**
