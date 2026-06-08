# Smartphone Placement Recognition Model Server Documentation

This document describes how to use, configure, and communicate with the placement recognition model server (`server.py`).

The server exposes a **FastAPI** application that serves the pre-trained AdaBoost ensemble model directly in Python. It supports **dual-input modes**:
1. **Raw iOS CoreMotion sensor values** (automatically preprocessed, resampled, and feature-extracted on the fly).
2. **Pre-computed 50 features** (directly passed as a dictionary or a vector).

---

## 🚀 Getting Started

### 1. Installation

Activate the local virtual environment and install the required dependencies:

```bash
# Activate the virtual environment
source venv/bin/activate

# Install the server dependencies
pip install -r requirements.txt
```

### 2. Running the Server

Start the API server by running:

```bash
python server.py
```

The server will load the model, parse the 449 decision trees into memory, and start listening on:
`http://0.0.0.0:8000`

---

## 📡 Endpoints

### 1. API Status and Required Features
* **URL**: `GET /`
* **Description**: Returns the online status of the server and the exact list of 50 features expected by the prediction model (in correct order).
* **Response Payload (JSON)**:
  ```json
  {
    "status": "online",
    "message": "Smartphone Placement Recognition Model API is active.",
    "features_required": [
      "CWT_Mean_Acc_B5",
      "SpecEntropyGyr",
      "KurtGyr",
      "... (50 total features) ..."
    ]
  }
  ```

---

### 2. Placement Prediction
* **URL**: `POST /predict`
* **Description**: Accepts either raw iOS CoreMotion data or pre-computed features and responds with the predicted smartphone placements under 3 different granularity schemes (6-class, 5-class, and 4-class).
* **Payload Type**: `application/json`

---

## 📱 iOS CoreMotion Ingestion (New)

The `/predict` endpoint can ingest raw iOS data directly. It is designed to handle standard serialization formats from Apple's CoreMotion framework:

### Format 1: Combined Device Motion List (Recommended)
This format sends a single array of combined accelerometer and gyroscope readings.

* **Request Payload (JSON)**:
  ```json
  {
    "motion": [
      {
        "timestamp": 1717849200.00,
        "acceleration": {"x": 0.05, "y": -0.98, "z": 0.01},
        "rotationRate": {"x": 0.12, "y": -0.05, "z": 0.03}
      },
      {
        "timestamp": 1717849200.01,
        "acceleration": {"x": 0.04, "y": -0.99, "z": 0.01},
        "rotationRate": {"x": 0.10, "y": -0.04, "z": 0.02}
      }
      // ... Repeat for a minimum of 9.0 seconds of walking data (ideally 10.0s) ...
    ]
  }
  ```
> **Note**: Both `"acceleration"` and `"userAcceleration"` are supported keys.

---

### Format 2: Separate Accelerometer & Gyroscope Arrays
This format handles separate sensor threads, which often log separate time series due to different collection rates or callbacks in Swift/Obj-C.

* **Request Payload (JSON)**:
  ```json
  {
    "accelerometer": [
      {"timestamp": 1717849200.00, "x": 0.05, "y": -0.98, "z": 0.01},
      {"timestamp": 1717849200.01, "x": 0.04, "y": -0.99, "z": 0.01}
      // ... minimum 9.0 seconds of data ...
    ],
    "gyroscope": [
      {"timestamp": 1717849200.00, "x": 0.12, "y": -0.05, "z": 0.03},
      {"timestamp": 1717849200.01, "x": 0.10, "y": -0.04, "z": 0.02}
      // ... minimum 9.0 seconds of data ...
    ]
  }
  ```

---

## ⚙️ Data Preprocessing & Feature Extraction Pipeline

When the server receives iOS data, it runs it through a custom-built digital signal processing (DSP) pipeline inside the `ios_dataformat` function:

1. **Unit Conversion**: iOS CoreMotion expresses raw acceleration in **G's**. The model expects values in **m/s²**. The server automatically multiplies all incoming `x`, `y`, `z` accelerometer values by $9.80665 \text{ m/s}^2$ to correct this. Gyroscope values remain in **rad/s**.
2. **Orientation Invariance**: To ensure predictions work regardless of how the user places or rotates the phone, the server collapses the 3-axis signals into 1D **L2 norms** (magnitude vector):
   $$\text{acc\_norm} = \sqrt{x^2 + y^2 + z^2}$$
   $$\text{gyr\_norm} = \sqrt{x^2 + y^2 + z^2}$$
3. **Resampling**: To normalize sample rates (e.g. if the iOS app collected data at 50 Hz, 60 Hz, or with slight jitters), the server performs a **linear interpolation** to resample both norms onto a uniform grid of exactly **100 Hz** spanning a **10-second window** (exactly 1000 points).
4. **On-the-fly Feature Extraction**: The server computes:
   * **Statistics**: Mean, Variance, Skewness, Pearson Kurtosis, IQR, RMS, SMA, Max, Range, and Jerk RMS.
   * **Spectral Features (FFT)**: Mean Frequency, Median Frequency, Spectral Entropy, and Dominant Power.
   * **Autocorrelation (ACF)**: Raw unnormalized autocorrelation coefficients for lags 1–10.
   * **Teager Energy Operator (TEO)**: Mean, Variance, and Peak Counts of the TEO signal.
   * **Multiscale Entropy (MSE)**: Coarse-grained sample entropy for scales 1 through 10.
   * **Continuous Wavelet Transform (CWT)**: Morse wavelet simulation to compute the Energy, Mean, Std, and Spectral Entropy across 8 distinct frequency bands.
   * **Harmonic Ratio (HR)**: Fundamental frequency and parity harmonic distribution.
5. **Feature Mapping**: Filters and orders the computed features to assemble the final 50-length feature vector required by the ensemble classifier.

---

## 🏷️ Classification Schemes (The 3 Classes)

The output is returned as a JSON object containing the predictions under three classification resolutions:

```json
{
  "class_6": "LB",
  "class_5": "LB",
  "class_4": "LB"
}
```

* **class_6** (6-classes): `LB` (Lower-Back), `H` (Handheld), `BP` (Back Pocket), `FP` (Front Pocket), `CP` (Coat Pocket), `SB` (Shoulder Bag).
* **class_5** (5-classes): `LB`, `H`, `TP` (Trousers Pocket - BP & FP merged), `CP`, `SB`.
* **class_4** (4-classes): `LB`, `H`, `P` (Pocket - BP, FP, & CP merged), `SB`.
