import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

# Ensure matio is available. If not, try installing/using scratch venv path.
try:
    import matio
except ImportError:
    # Append the scratch virtual environment site-packages path as fallback
    venv_site_packages = "/Users/sulabhkatila/.gemini/antigravity-cli/brain/4eea5ebe-0e39-4f9d-acf5-bdd9cd1f8ac9/scratch/venv/lib/python3.13/site-packages"
    if os.path.exists(venv_site_packages):
        sys.path.append(venv_site_packages)
        import matio
    else:
        raise ImportError("mat-io library is not installed. Run 'pip install mat-io' first.")

app = FastAPI(
    title="Smartphone Placement Recognition API",
    description="API for recognizing smartphone placement (body location) using walking features.",
    version="1.0.0"
)

# Configuration and Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "best50_features", "ensModel.mat")
FEAT_SEL_PATH = os.path.join(BASE_DIR, "results", "models", "best50_features", "FeatSel_Results.mat")

# Global variables to store the loaded model and features
selected_features = []
class_names = ['LB', 'H', 'BP', 'FP', 'CP', 'SB']
parsed_trees = []
learner_weights = []

# Label Mapping Functions
def map_to_5_classes(label: str) -> str:
    # FP (Front Pocket) and BP (Back Pocket) are merged into TP (Trousers Pocket)
    if label in ("FP", "BP"):
        return "TP"
    return label

def map_to_4_classes(label: str) -> str:
    # BP, FP, and CP (Coat Pocket) are merged into P (Pocket)
    if label in ("BP", "FP", "CP"):
        return "P"
    return label

@app.on_event("startup")
def load_model_on_startup():
    global selected_features, parsed_trees, learner_weights
    
    print("Loading model and feature list at startup...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEAT_SEL_PATH):
        raise FileNotFoundError(f"Model or Feature Selection files missing at: {MODEL_PATH} or {FEAT_SEL_PATH}")
        
    # Load feature selection results
    feat_data = matio.load_from_mat(FEAT_SEL_PATH)
    selected_features = [s[0] if hasattr(s, 'flat') else s for s in feat_data['selected_features'].flat]
    print(f"Loaded {len(selected_features)} selected features.")
    
    # Load ensemble model
    model_data = matio.load_from_mat(MODEL_PATH)
    ens_model = model_data['ensModel']
    
    # Extract learners and weights
    impl = ens_model.properties['Impl']
    learners = impl.properties['Trained'].flat
    weights = impl.properties['Combiner'].properties['LearnerWeights'].flat
    learner_weights = np.array([w for w in weights])
    
    # Pre-parse learners for fast traversal in Python
    parsed_trees = []
    for learner in learners:
        tree_impl = learner.properties['Impl']
        children = tree_impl.properties['Children'].astype(int)
        cut_var = tree_impl.properties['CutVar'].flatten().astype(int)
        cut_point = tree_impl.properties['CutPoint'].flatten()
        class_prob = tree_impl.properties['ClassProb']
        
        parsed_trees.append({
            'children': children,
            'cut_var': cut_var,
            'cut_point': cut_point,
            'class_prob': class_prob
        })
    print(f"Loaded ensemble of {len(parsed_trees)} decision trees successfully.")

# Helper function for tree prediction
def predict_tree(x: np.ndarray, tree: dict) -> np.ndarray:
    children = tree['children']
    cut_var = tree['cut_var']
    cut_point = tree['cut_point']
    class_prob = tree['class_prob']
    
    node = 0
    while True:
        left_child = children[node, 0]
        right_child = children[node, 1]
        
        # Leaf node check
        if left_child == 0 and right_child == 0:
            return class_prob[node]
            
        v = cut_var[node] - 1
        val = x[v]
        
        if val < cut_point[node]:
            node = left_child - 1
        else:
            node = right_child - 1

def run_predictions(x: np.ndarray) -> dict:
    scores = np.zeros(len(class_names))
    for t_idx, tree in enumerate(parsed_trees):
        prob = predict_tree(x, tree)
        scores += learner_weights[t_idx] * prob
        
    pred_idx = np.argmax(scores)
    pred_class_6 = class_names[pred_idx]
    pred_class_5 = map_to_5_classes(pred_class_6)
    pred_class_4 = map_to_4_classes(pred_class_6)
    
    return {
        "class_6": pred_class_6,
        "class_5": pred_class_5,
        "class_4": pred_class_4
    }

# Dynamic Request Schemas
class FeatureVectorRequest(BaseModel):
    # Allows sending either:
    # 1. A dictionary mapping feature names to float values
    # 2. A list of 50 floats (in the order of selected_features)
    data: Union[Dict[str, float], List[float]]

class PredictionResponse(BaseModel):
    class_6: str
    class_5: str
    class_4: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Smartphone Placement Recognition Model API is active.",
        "features_required": selected_features
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: FeatureVectorRequest):
    input_data = request.data
    x = []
    
    if isinstance(input_data, list):
        if len(input_data) != len(selected_features):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid feature vector length. Expected {len(selected_features)} features, got {len(input_data)}."
            )
        x = np.array(input_data, dtype=float)
    else:
        # Dictionary format - check for missing features
        missing = [f for f in selected_features if f not in input_data]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature variables in input request: {missing}"
            )
        # Construct array in the exact feature order used in training
        x = np.array([input_data[f] for f in selected_features], dtype=float)
        
    # Perform prediction
    results = run_predictions(x)
    return results

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
