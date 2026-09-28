<h1 align="center">🛣️ PathFinder</h1>
<p align="center">
  <strong>Real-Time Lane Detection & Drift Alert System</strong><br/>
  Bird's-Eye View · Polynomial Curve Fitting · Day/Night Adaptive · Live Web App
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black"/>
</p>

---

## 📌 Overview

**PathFinder** is an end-to-end computer vision pipeline that processes dashcam footage and returns the same video with lane markings overlaid and a real-time drift alert burned in. No black-box neural networks — just principled, high-performance classical computer vision and geometry.

| Feature | Details |
|---|---|
| **Lane Tracking** | Detects both straight and curved lanes via 2nd-degree polynomial fitting ($x = ay^2 + by + c$) |
| **Drift Alert** | Fires `DRIFT LEFT / RIGHT` when vehicle deviates > 8% of frame width from lane center |
| **Day / Night Mode** | Adaptive CLAHE, gamma correction, and HLS thresholds (auto-detected or manual) |
| **Web Interface** | Clean drag-and-drop video upload with side-by-side before/after comparison playback |
| **Live Deployment** | Packaged with Docker and deployed on Hugging Face Spaces (`ghyoco-lane-detection.hf.space`) |

---

## 🎬 Demo

Upload any dashcam clip (MP4/MOV, up to 100 MB) and get back the first ~30 seconds with the overlay applied.

```
Input:   Raw dashcam footage (Day, Night, or Curved roads)
Output:  Green lane polygon fill + Lane boundary traces + Real-time Drift HUD
```

> 💡 **Live Demo**: [https://lane-detection-cv.vercel.app/](https://lane-detection-cv.vercel.app/)

---

## 🏗️ Architecture & Project Structure

```
PathFinder/
├── model/
│   └── lane_detection.ipynb   # R&D notebook with step-by-step diagnostic grid
├── backend/
│   ├── lane_detection.py      # Core CV pipeline (BEV, Polyfit, Day/Night adaptation)
│   ├── main.py                # FastAPI server + FFmpeg H.264 video transcoding
│   ├── requirements.txt       # Production dependencies
│   └── Dockerfile             # Container configuration for HF Spaces / Cloud deployment
├── frontend/
│   ├── index.html             # Drag-and-drop web UI
│   ├── app.js                 # API fetch handler & video blob player
│   └── style.css              # Responsive UI styling
└── Dataset/
    ├── Day.MP4                # Daylight highway dataset
    ├── Night.MP4              # Low-light headlamp dataset
    └── Curve.MP4              # Winding road / curvature dataset
```

**Request Pipeline Flow:**

```
[ Browser Client ]
       │
       ▼ (multipart/form-data video upload)
[ FastAPI Endpoint: POST /api/process ]
       │
       ├─► 1. Preprocessing (Gamma, CLAHE, Bilateral Filter, HLS Color Mask)
       ├─► 2. Edge Extraction (Adaptive Canny & Trapezoidal ROI Masking)
       ├─► 3. Bird's-Eye View (BEV) Homography Warp
       ├─► 4. Sliding-Window Pixel Clustering (20 vertical windows)
       ├─► 5. 2nd-Degree Polynomial Fit (x = ay² + by + c)
       ├─► 6. Exponential Moving Average (EMA) Coefficient Smoothing
       ├─► 7. Drift Measurement & HUD Warning Generation
       ├─► 8. Inverse Perspective Unwarp & Lane Overlay Blending
       │
       ▼ (Raw OpenCV mp4v stream)
[ FFmpeg H.264 Transcoding ] (libx264, yuv420p for browser playback)
       │
       ▼
[ Client Video Player ] (Side-by-side Before & After playback)
```

---

## 🔬 Pipeline Deep-Dive

### 1. Adaptive Preprocessing
- **Gamma Correction**: Brightens dark/night frames before contrast equalization ($\gamma = 0.5$ at night, $1.0$ for day).
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Lifts faint lane markings under glare or pitch-black night conditions (`clipLimit=2.0` day, `4.5` night).
- **Bilateral Filtering**: Smooths high-frequency asphalt noise while preserving sharp line edges.
- **HLS Color Thresholding**: Isolates white lines ($L \ge 190$ day, $110$ night) and yellow lines ($H \in [10, 45]$, $S \ge 100$).
- **Morphological Dilation**: Connects fragmented dashed lane markings.

### 2. Edge Detection & ROI
- **Adaptive Canny**: Upper and lower gradient thresholds dynamically calculated from frame median intensity $\tilde{v}$:
  $$\text{threshold}_{\text{low}} = (1 - \sigma)\tilde{v}, \quad \text{threshold}_{\text{high}} = (1 + \sigma)\tilde{v}$$
- **Trapezoidal ROI Mask**: Clamps processing to the drivable horizon to avoid trees, signs, and sky.

### 3. Bird’s-Eye View (BEV) Perspective Warping
A perspective matrix $M$ projects the camera view into a top-down orthogonal road plane:
- Converts converging perspective lines into parallel lines.
- Widened top corners (`ROI_TOP_LEFT = (0.40, 0.46)`, `ROI_TOP_RIGHT = (0.62, 0.46)`) capture curved paths exiting the horizon.

### 4. Sliding-Window Clustering (20 Windows)
- Bins edge pixels into vertical slices ($N = 20$).
- If window pixel density $\ge 20$, the subsequent window shifts its center to $\text{mean}(x)$ of the cluster.
- **Pinch Guard**: Protects against false histogram merges when curves bring lanes visually closer in BEV.

### 5. 2nd-Degree Polynomial Fit & Smoothing
Lanes are modeled mathematically as quadratic curves:
$$x(y) = a y^2 + b y + c$$
- **Curvature Coefficient ($a$)**: Quantifies turn intensity ($a > 0$ curves right, $a < 0$ curves left). Physical limit filter ($|a| \le 0.005$) drops implausible fits.
- **Exponential Moving Average (EMA)**:
  $$\text{coeffs}_t = \alpha \cdot \text{coeffs}_{\text{cur}} + (1 - \alpha) \cdot \text{coeffs}_{t-1}$$
  Filters out instantaneous frame dropouts and camera vibration.

### 6. Vehicle Drift Telemetry
- Assumes centered camera mount: $\text{car\_center} = \frac{w}{2}$.
- Computes lane center at vehicle bumper level ($y_{\text{eval}} = h - 1$):
  $$\text{lane\_center} = \frac{x_{\text{left}}(y_{\text{eval}}) + x_{\text{right}}(y_{\text{eval}})}{2}$$
- **Offset ($\text{off}$)** $= \text{car\_center} - \text{lane\_center}$:
  - If $|\text{off}| > 0.08 \times w$: Triggers **`DRIFT RIGHT!`** ($\text{off} > 0$) or **`DRIFT LEFT!`** ($\text{off} < 0$) in red.
  - Otherwise: Displays **`On Lane`** in green.

---

## ⚙️ Configuration Reference

Tunable parameters configured in `backend/lane_detection.py`:

| Parameter | Day Setting | Night Setting | Purpose |
|---|---|---|---|
| `white_l_min` | `190` | `110` | Minimum lightness for white paint |
| `yellow_h` | `(10, 45)` | `(8, 50)` | Hue band for yellow lane markings |
| `yellow_s_min` | `100` | `55` | Saturation floor for reflective yellow paint |
| `canny_sigma` | `0.33` | `0.18` | Canny margin around median intensity |
| `window_margin` | `80 px` | `90 px` | Half-width of sliding search window |
| `ema_alpha` | `0.30` | `0.40` | Responsiveness of temporal coefficient filter |
| `clahe_clip` | `2.0` | `4.5` | Contrast amplification limit |
| `gamma` | `1.0` | `0.50` | Power-law pre-brightening exponent |
| `dilate_iter` | `1` | `3` | Dilation kernel repetitions for dashed lines |
| `DRIFT_THRESHOLD_FRAC` | `0.08` | `0.08` | Frame width fraction for lane departure trigger |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- FFmpeg installed on system PATH (or use bundled `imageio-ffmpeg`)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Backend API will be running at `http://localhost:8000`.

### 2. Frontend Setup

Open `frontend/index.html` directly in any web browser, or serve it using Python:
```bash
cd frontend
python -m http.server 3000
```
*Note: If running backend locally, ensure `API_URL` in `frontend/app.js` is set to `http://localhost:8000`.*

### 3. Running with Docker

```bash
cd backend
docker build -t pathfinder:latest .
docker run -p 7860:7860 pathfinder:latest
```

---

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV (`cv2`), NumPy
- **Backend Framework**: FastAPI, Uvicorn, Pydantic
- **Video Encoding**: FFmpeg, imageio-ffmpeg (H.264 / AAC)
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Video API, CSS3 Flexbox/Grid
- **DevOps & Cloud**: Docker, Hugging Face Spaces

---

