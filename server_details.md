# Smartphone Placement Recognition Model Server Documentation

This document describes how to use, configure, and communicate with the placement recognition model server (`server.py`).

The server exposes a **FastAPI** application that serves the pre-trained AdaBoost ensemble model directly in Python using `mat-io` for reading the `.mat` file structures. Inference runs locally in pure Python with no MATLAB runtime requirement.

---

## Getting Started

### 1. Installation

A local virtual environment has been created. To activate it and run the server, use:

```bash
# Activate the virtual environment
source venv/bin/activate

# Install the server dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Running the Server

Start the API server by running:

```bash
python server.py
```

The server will initialize, load the model (`ensModel.mat`) and selected feature names (`FeatSel_Results.mat`), parse all 449 decision trees into memory, and start listening on:
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
* **Description**: Accepts a single data sample containing 10-second walking window features and responds with the predicted smartphone placements under 3 different granularity schemes (6-class, 5-class, and 4-class).
* **Payload Type**: `application/json`

#### Input Format A: Dictionary (Recommended)
You can submit a JSON object containing the feature names as keys and float values as values.
* **Request Payload (JSON)**:
  ```json
  {
    "data": {
      "CWT_Mean_Acc_B5": 0.871815,
      "SpecEntropyGyr": 0.165810,
      "KurtGyr": 1.594311,
      "... (provide all 50 features) ...": 0.0
    }
  }
  ```

#### Input Format B: Vector List
You can submit a JSON array of exactly 50 float values in the same order as returned by the `GET /` endpoint.
* **Request Payload (JSON)**:
  ```json
  {
    "data": [
      0.871815,
      0.165810,
      1.594311,
      "... (provide exactly 50 floats) ..."
    ]
  }
  ```

#### Response Payload (JSON)
* **Response**:
  ```json
  {
    "class_6": "LB",
    "class_5": "LB",
    "class_4": "LB"
  }
  ```

---

## 🏷️ Classification Schemes (The 3 Classes)

The server output reports the predicted placement using three granularity levels mapping to common body locations:

| Scheme | Classes | Description & Mapping |
| :--- | :--- | :--- |
| **class_6** | `LB`, `H`, `BP`, `FP`, `CP`, `SB` | Original 6 placements:<br>- `LB`: Lower-Back (L5 level)<br>- `H`: Hand-held<br>- `BP`: Back pocket (trousers)<br>- `FP`: Front pocket (trousers)<br>- `CP`: Coat pocket<br>- `SB`: Shoulder bag |
| **class_5** | `LB`, `H`, `TP`, `CP`, `SB` | Merged pockets:<br>- `TP`: Trousers Pocket (merges `FP` & `BP`) |
| **class_4** | `LB`, `H`, `P`, `SB` | Fully merged pockets:<br>- `P`: Pocket (merges `FP`, `BP`, & `CP`) |

---

## 🛠️ Implementation Details

* **Inference Algorithm**: An ensemble of 449 decision trees trained with AdaBoostM2. The class probability vectors of all trees are averaged weighted by the ensemble's `LearnerWeights`. The class with the highest score is predicted.
* **Feature Ordering**: The model is sensitive to the order of features. In the dictionary format, the server automatically maps features to the correct columns before sending them to the model, eliminating potential user-side order mismatch bugs.
