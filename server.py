import os
from contextlib import asynccontextmanager
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import scipy.stats as stats
import scipy.signal as signal
from scipy.spatial.distance import cdist

# Ensure matio is available. If not, try using scratch venv path.
try:
    import matio
except ImportError:
    venv_site_packages = "/Users/sulabhkatila/.gemini/antigravity-cli/brain/4eea5ebe-0e39-4f9d-acf5-bdd9cd1f8ac9/scratch/venv/lib/python3.13/site-packages"
    if os.path.exists(venv_site_packages):
        sys.path.append(venv_site_packages)
        import matio
    else:
        raise ImportError("mat-io library is not installed. Run 'pip install mat-io' first.")

@asynccontextmanager
async def lifespan(app):
    load_model_on_startup()
    yield

app = FastAPI(
    title="Smartphone Placement Recognition API",
    description="API for recognizing smartphone placement (body location) using raw iOS sensor data or pre-computed features.",
    version="2.0.0",
    lifespan=lifespan
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

# --- DSP / Feature Extraction helper functions ---

def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    N = len(x)
    if N <= m:
        return 0.0
    
    # Form m-dimensional templates
    Xm = np.array([x[i:i+m] for i in range(N - m + 1)])
    # Form (m+1)-dimensional templates
    Xmp = np.array([x[i:i+m+1] for i in range(N - m)])
    
    # Compute Chebyshev distances
    d_m = cdist(Xm, Xm, metric='chebyshev')
    B = np.sum(d_m <= r) - len(Xm) # subtract diagonal (self-matches)
    
    d_mp = cdist(Xmp, Xmp, metric='chebyshev')
    A = np.sum(d_mp <= r) - len(Xmp)
    
    if A == 0 or B == 0:
        return 0.0
    return -np.log(A / B)

def coarse_grain(x: np.ndarray, s: int) -> np.ndarray:
    N = len(x)
    num_points = N // s
    return np.mean(x[:num_points*s].reshape(num_points, s), axis=1)

def multiscale_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2, scales: range = range(1, 11)) -> dict:
    std_x = np.std(x)
    r = r_factor * std_x
    entropy_values = {}
    for s in scales:
        cg_x = coarse_grain(x, s)
        entropy_values[s] = sample_entropy(cg_x, m, r)
    return entropy_values

def harmonic_ratio(sig: np.ndarray, fs: float = 100.0) -> float:
    sig = sig - np.mean(sig)
    N = len(sig)
    if N == 0:
        return 0.0
    faxis = np.fft.fftfreq(N, 1/fs)
    Y = np.fft.fft(sig)
    P2 = np.abs(Y / N)
    
    # Keep only positive frequencies
    half_n = N // 2
    P1 = P2[:half_n]
    f = faxis[:half_n]
    
    # Find fundamental frequency
    if len(P1) <= 1:
        return 0.0
    idx_max = np.argmax(P1[1:]) + 1 # ignore DC
    f0 = f[idx_max]
    
    if f0 == 0:
        return 0.0
        
    # Number of harmonics
    num_harmonics = int(np.floor((fs / 2) / f0))
    
    odd_sum = 0.0
    even_sum = 0.0
    for k in range(1, num_harmonics + 1):
        idx_h = int(np.round(f0 * k / (fs / N)))
        if idx_h < len(P1):
            if k % 2 == 1:
                odd_sum += P1[idx_h]
            else:
                even_sum += P1[idx_h]
                
    if odd_sum == 0:
        return 0.0
    return even_sum / odd_sum

def compute_fft_features(x: np.ndarray, prefix: str, fs: float = 100.0) -> dict:
    N = len(x)
    x_detrend = x - np.mean(x)
    fft_val = np.abs(np.fft.fft(x_detrend))
    
    half_n = N // 2 + 1
    fft_side = fft_val[:half_n]
    
    sum_fft = np.sum(fft_side)
    if sum_fft > 0:
        fft_norm = fft_side / sum_fft
    else:
        fft_norm = np.zeros_like(fft_side)
        
    faxis = np.fft.fftfreq(N, 1/fs)[:half_n]
    
    if len(fft_norm) > 1:
        idx1 = np.argmax(fft_norm[1:]) + 1
        fft_temp = fft_norm.copy()
        fft_temp[idx1] = 0.0
        idx2 = np.argmax(fft_temp[1:]) + 1
        dom_freqs = faxis[[idx1, idx2]]
    else:
        dom_freqs = [0.0, 0.0]
        
    mean_freq = np.sum(faxis * fft_norm)
    cumsum_fft = np.cumsum(fft_norm)
    idx_med = np.searchsorted(cumsum_fft, 0.5)
    median_freq = faxis[idx_med] if idx_med < len(faxis) else 0.0
    spec_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-12))
    dom_power = np.max(fft_norm)
    
    return {
        f'SpecEntropy{prefix}': spec_entropy,
        f'DomPower{prefix}': dom_power,
        f'MeanFreq{prefix}': mean_freq,
        f'MedianFreq{prefix}': median_freq,
        f'F0F1Dist{prefix}': abs(dom_freqs[0] - dom_freqs[1])
    }

def compute_cwt_band_features(acc: np.ndarray, gyr: np.ndarray, fs: float = 100.0) -> dict:
    N = len(acc)
    acc_fft = np.fft.fft(acc - np.mean(acc))
    gyr_fft = np.fft.fft(gyr - np.mean(gyr))
    omega = 2 * np.pi * np.fft.fftfreq(N, 1/fs)
    
    frequencies = np.linspace(0.5, 25, 40)
    freq_bands = [
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
        (5.0, 8.0),
        (8.0, 12.0),
        (12.0, 18.0),
        (18.0, 25.0)
    ]
    
    band_cfs_acc = {i: [] for i in range(len(freq_bands))}
    band_cfs_gyr = {i: [] for i in range(len(freq_bands))}
    
    for f in frequencies:
        a = 6 / (2 * np.pi * f)
        filt = (omega > 0) * (2 * np.sqrt(np.pi))**0.5 * np.exp(-0.5 * (a * omega - 6)**2)
        
        # ACC CWT
        C_acc = np.fft.ifft(acc_fft * filt)
        power_acc = np.abs(C_acc)**2
        
        # GYR CWT
        C_gyr = np.fft.ifft(gyr_fft * filt)
        power_gyr = np.abs(C_gyr)**2
        
        for b_idx, (low, high) in enumerate(freq_bands):
            if low <= f < high:
                band_cfs_acc[b_idx].append(power_acc)
                band_cfs_gyr[b_idx].append(power_gyr)
                break
                
    features = {}
    spec_ent_fn = lambda P: -np.sum(P * np.log(P + 1e-12))
    
    for b_idx in range(len(freq_bands)):
        band_num = b_idx + 1
        
        # ACC
        powers_acc = band_cfs_acc[b_idx]
        if len(powers_acc) > 0:
            bp_acc = np.concatenate(powers_acc)
            features[f'CWT_Eng_Acc_B{band_num}'] = np.sum(bp_acc)
            features[f'CWT_Mean_Acc_B{band_num}'] = np.mean(bp_acc)
            features[f'CWT_Std_Acc_B{band_num}'] = np.std(bp_acc)
            p_norm_acc = bp_acc / (np.sum(bp_acc) + 1e-12)
            features[f'CWT_Entropy_Acc_B{band_num}'] = spec_ent_fn(p_norm_acc)
        else:
            features[f'CWT_Eng_Acc_B{band_num}'] = 0.0
            features[f'CWT_Mean_Acc_B{band_num}'] = 0.0
            features[f'CWT_Std_Acc_B{band_num}'] = 0.0
            features[f'CWT_Entropy_Acc_B{band_num}'] = 0.0
            
        # GYR
        powers_gyr = band_cfs_gyr[b_idx]
        if len(powers_gyr) > 0:
            bp_gyr = np.concatenate(powers_gyr)
            features[f'CWT_Eng_Gyr_B{band_num}'] = np.sum(bp_gyr)
            features[f'CWT_Mean_Gyr_B{band_num}'] = np.mean(bp_gyr)
            features[f'CWT_Std_Gyr_B{band_num}'] = np.std(bp_gyr)
            p_norm_gyr = bp_gyr / (np.sum(bp_gyr) + 1e-12)
            features[f'CWT_Entropy_Gyr_B{band_num}'] = spec_ent_fn(p_norm_gyr)
        else:
            features[f'CWT_Eng_Gyr_B{band_num}'] = 0.0
            features[f'CWT_Mean_Gyr_B{band_num}'] = 0.0
            features[f'CWT_Std_Gyr_B{band_num}'] = 0.0
            features[f'CWT_Entropy_Gyr_B{band_num}'] = 0.0
            
    return features

def extract_all_features(acc: np.ndarray, gyr: np.ndarray) -> dict:
    fs = 100
    features = {}
    
    # 1. Statistics
    for signal_arr, prefix in [(acc, 'Acc'), (gyr, 'Gyr')]:
        features[f'Mean{prefix}'] = np.mean(signal_arr)
        features[f'Var{prefix}'] = np.var(signal_arr, ddof=1)
        features[f'Skew{prefix}'] = stats.skew(signal_arr, bias=True)
        features[f'Kurt{prefix}'] = stats.kurtosis(signal_arr, bias=True, fisher=False)
        features[f'IQR{prefix}'] = stats.iqr(signal_arr)
        features[f'RMS{prefix}'] = np.sqrt(np.mean(signal_arr**2))
        features[f'SMA{prefix}'] = np.mean(np.abs(signal_arr))
        features[f'Max{prefix}'] = np.max(signal_arr)
        features[f'Range{prefix}'] = np.ptp(signal_arr)
        
    # 2. Jerk RMS
    features['JerkRMS'] = np.sqrt(np.mean(np.diff(acc)**2))
    
    # 3. Median Cross Rate
    features['MedCrossRateAcc'] = np.sum(np.diff(acc > np.median(acc)) != 0) / len(acc)
    features['MedCrossRateGyr'] = np.sum(np.diff(gyr > np.median(gyr)) != 0) / len(gyr)
    
    # 4. Teager Energy Operator
    for signal_arr, prefix in [(acc, 'Acc'), (gyr, 'Gyr')]:
        teo_val = signal_arr[1:-1]**2 - signal_arr[:-2] * signal_arr[2:]
        features[f'TEO_Mean_{prefix}'] = np.mean(teo_val)
        features[f'TEO_Var_{prefix}'] = np.var(teo_val, ddof=1)
        peaks, _ = signal.find_peaks(teo_val, prominence=2.5)
        features[f'TEO_Peaks_{prefix}'] = len(peaks)
        
    # 5. Multiscale Entropy (scales 1 to 10)
    entropy_acc = multiscale_entropy(acc, m=2, r_factor=0.2, scales=range(1, 11))
    for s, val in entropy_acc.items():
        features[f'En{s}_Acc'] = val
    features['EnSum_Acc'] = sum(entropy_acc.values())
    
    entropy_gyr = multiscale_entropy(gyr, m=2, r_factor=0.2, scales=range(1, 11))
    for s, val in entropy_gyr.items():
        features[f'En{s}_Gyr'] = val
    features['EnSum_Gyr'] = sum(entropy_gyr.values())
    
    # 6. Autocorrelation (ACF lags 1-10)
    for signal_arr, prefix in [(acc, 'Acc'), (gyr, 'Gyr')]:
        for k in range(1, 11):
            features[f'ACF{k}_{prefix}'] = np.sum(signal_arr[:-k] * signal_arr[k:])
            
    # 7. NumPeaks
    for signal_arr, prefix in [(acc, 'Acc'), (gyr, 'Gyr')]:
        peaks, _ = signal.find_peaks(signal_arr, prominence=2.5)
        features[f'NumPeaks{prefix}'] = len(peaks)
        
    # 8. FFT Frequency Domain features
    features.update(compute_fft_features(acc, 'Acc', fs))
    features.update(compute_fft_features(gyr, 'Gyr', fs))
    
    # 9. Harmonic Ratio
    features['HarmonicRatioAcc'] = harmonic_ratio(acc, fs)
    features['HarmonicRatioGyr'] = harmonic_ratio(gyr, fs)
    
    # 10. Acc-Gyr cross features
    features['GyroAccCorr'] = np.corrcoef(acc, gyr)[0, 1]
    features['GyroAccMax'] = np.max(acc) * np.max(gyr)
    features['GyroAccStdProd'] = np.std(acc) * np.std(gyr)
    
    # 11. CWT Band Features
    features.update(compute_cwt_band_features(acc, gyr, fs))
    
    return features

# --- iOS Data Parsing and Formatting function ---

def ios_dataformat(raw_data: dict) -> np.ndarray:
    """
    Parses accelerometer and gyroscope measurements provided in default iOS CoreMotion formats,
    processes them (converting G's to m/s^2, computing L2 norm magnitudes, resampling to 100 Hz),
    extracts the required signal processing features, and formats them for the model.
    """
    # 1. Parse different iOS CoreMotion structures
    if "motion" in raw_data:
        motion_list = raw_data["motion"]
        t_acc = np.array([m.get("timestamp") or m.get("time") for m in motion_list])
        t_gyr = t_acc
        
        acc_x = np.array([(m.get("acceleration") or m.get("userAcceleration"))["x"] for m in motion_list])
        acc_y = np.array([(m.get("acceleration") or m.get("userAcceleration"))["y"] for m in motion_list])
        acc_z = np.array([(m.get("acceleration") or m.get("userAcceleration"))["z"] for m in motion_list])
        
        gyr_x = np.array([m.get("rotationRate")["x"] for m in motion_list])
        gyr_y = np.array([m.get("rotationRate")["y"] for m in motion_list])
        gyr_z = np.array([m.get("rotationRate")["z"] for m in motion_list])
        
    elif "accelerometer" in raw_data and "gyroscope" in raw_data:
        acc_list = raw_data["accelerometer"]
        gyr_list = raw_data["gyroscope"]
        
        t_acc = np.array([a.get("timestamp") or a.get("time") for a in acc_list])
        t_gyr = np.array([g.get("timestamp") or g.get("time") for g in gyr_list])
        
        acc_x = np.array([a["x"] for a in acc_list])
        acc_y = np.array([a["y"] for a in acc_list])
        acc_z = np.array([a["z"] for a in acc_list])
        
        gyr_x = np.array([g["x"] for g in gyr_list])
        gyr_y = np.array([g["y"] for g in gyr_list])
        gyr_z = np.array([g["z"] for g in gyr_list])
    else:
        raise ValueError("Invalid iOS data format. Payload must contain 'motion' array or separate 'accelerometer' and 'gyroscope' arrays.")
        
    if len(t_acc) < 5 or len(t_gyr) < 5:
        raise ValueError("Insufficient data points in iOS CoreMotion arrays to perform analysis.")
        
    # Convert G's to m/s^2 for accelerometer measurements (iOS CoreMotion uses G's by default, model expects m/s^2)
    acc_x = acc_x * 9.80665
    acc_y = acc_y * 9.80665
    acc_z = acc_z * 9.80665
    
    # Compute L2 Norm (orientation-invariant magnitude)
    acc_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyr_norm = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    
    # Normalize timestamps relative to the start of the window
    t0 = min(t_acc[0], t_gyr[0])
    t_acc_rel = t_acc - t0
    t_gyr_rel = t_gyr - t0
    
    # Verify the window duration (should span roughly 10 seconds)
    duration = max(t_acc_rel[-1], t_gyr_rel[-1])
    if duration < 9.0:
        raise ValueError(f"Input data duration is too short. Got {duration:.2f} seconds, expected at least 9.0s to construct a 10s walking window.")
        
    # Resample signals to exactly 100 Hz over a 10-second window (1000 samples)
    t_target = np.linspace(0, 10, 1000)
    acc_resampled = np.interp(t_target, t_acc_rel, acc_norm)
    gyr_resampled = np.interp(t_target, t_gyr_rel, gyr_norm)
    
    # Extract the full suite of signal processing features
    features = extract_all_features(acc_resampled, gyr_resampled)
    
    # Map and order the features to match the exact 50 expected features
    x = []
    for f in selected_features:
        if f in features:
            x.append(features[f])
        else:
            # Fallback default value (should not happen since we compute everything)
            x.append(0.0)
            
    return np.array(x, dtype=float)

# --- Server Lifecycle & Endpoints ---

def load_model_on_startup():
    global selected_features, parsed_trees, learner_weights
    
    print("Loading model and feature list at startup...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEAT_SEL_PATH):
        raise FileNotFoundError(f"Model or Feature Selection files missing at: {MODEL_PATH} or {FEAT_SEL_PATH}")
        
    # Load feature selection results
    feat_data = matio.load_from_mat(FEAT_SEL_PATH)
    selected_features = [str(s[0]) if hasattr(s, 'flat') else str(s) for s in feat_data['selected_features'].flat]
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
def predict(payload: dict):
    # This endpoint is dual-input capable:
    # 1. Standard raw iOS data payload (JSON object containing "motion" or separate "accelerometer" and "gyroscope" lists)
    # 2. Legacy pre-computed 50 features payload (JSON object containing "data" key as dict or list)
    
    print("Connection created")
    try:
        if "data" in payload:
            # Handle direct pre-computed features input
            input_data = payload["data"]
            if isinstance(input_data, list):
                if len(input_data) != len(selected_features):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid feature vector length. Expected {len(selected_features)} features, got {len(input_data)}."
                    )
                x = np.array(input_data, dtype=float)
            else:
                missing = [f for f in selected_features if f not in input_data]
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing feature variables in input request: {missing}"
                    )
                x = np.array([input_data[f] for f in selected_features], dtype=float)
        else:
            # Handle raw iOS sensor measurements
            if not ("motion" in payload or ("accelerometer" in payload and "gyroscope" in payload)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unrecognized payload format. Expected one of: "
                        "(1) {\"data\": [...]} with 50 pre-computed features, "
                        "(2) {\"motion\": [...]} with combined CoreMotion samples, or "
                        "(3) {\"accelerometer\": [...], \"gyroscope\": [...]} with separate sensor arrays."
                    )
                )
            x = ios_dataformat(payload)
            
        results = run_predictions(x)
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
