# app.py
"""
LifeSaver AI - Single-file Streamlit app (Emergency Monitor Only)
Features:
- **MODIFIED**: Unsupervised section now uses ONLY the K-Means Clustering model.
- **FIXED**: Robust data cleaning and protocol timer logic.
- **FIXED**: Black background and bright cyan font.

Required files (in same folder):
- unclean_smartwatch_health_data.csv
- ai_protocol_alert.mp3  (optional)

Install required packages:
pip install streamlit scikit-learn pandas numpy joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# CONFIG / CONSTANTS
# ----------------------------
CSV_FILENAME = "unclean_smartwatch_health_data.csv"
UNSUPERVISED_MODEL_FILE = "unsupervised_models_kmeans.joblib" # Changed filename to avoid conflict
# **CLEANED DICTIONARY DEFINITION TO REMOVE U+00A0**
MODEL_FILENAMES = {
    "Logistic Regression": "lr_model.joblib",
    "Random Forest": "rf_model.joblib",
    "SVM": "svm_model.joblib"
}
AI_VOICE_FILE = "ai_protocol_alert.mp3" 

FEATURE_COLUMNS = [
    "Heart Rate (BPM)",
    "Blood Oxygen Level (%)",
    "Sleep Duration (hours)",
    "Stress Level",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
]

# ----------------------------
# CLASSY BLACK UI THEME
# ----------------------------
CLASSY_BLACK_UI_CSS = """
<style>

/* CRITICAL FIXES: Force Black Background on ALL Main Streamlit Containers */
html, body, [class*="css"] {
    background: #000000 !important;
    color: #00FFFF !important; /* BRIGHT CYAN/BLUE FONT FOR VISIBILITY */
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Aggressively target Streamlit main blocks, sidebar, and toolbar */
[data-testid="stAppViewBlock"], 
[data-testid="stToolbar"],
[data-testid="stVerticalBlock"],
.main {
    background: #000000 !important;
}

/* Sidebar Fix: Ensure sidebar content area and background are dark */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #040404 !important;
    border-right: 1px solid rgba(100, 100, 100, 0.1);
}

/* Global Text Color Fix for High Visibility (targeting components again) */
div, p, span, label, [data-testid="stText"], [data-testid="stMarkdownContainer"] {
    color: #00FFFF !important;
}

/* Headings - Brighter Neon Accent */
h1, h2, h3, h4 {
    color: #00e5ff !important;
    text-shadow: 0 0 6px rgba(0, 229, 255, 0.4);
    letter-spacing: 1px;
    font-weight: 800;
}

/* Glass Cards - Enhanced Aesthetic */
.metric-card {
    background: rgba(255,255,255,0.05); 
    border: 1px solid rgba(0, 229, 255, 0.12); 
    backdrop-filter: blur(10px); 
    padding: 20px;
    border-radius: 16px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 0 0 8px rgba(0, 229, 255, 0.1);
    margin-bottom: 15px;
}

/* Buttons - Dynamic Neon */
.stButton>button {
    background: linear-gradient(90deg, #00bfff, #00e5ff) !important;
    color: #021221 !important;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    box-shadow: 0 0 10px rgba(0, 191, 255, 0.4);
    transition: all 0.2s ease;
}

/* Inputs, Textarea, Sliders - Interactive Glow and Color Fix */
input, .stTextInput, .stNumberInput, textarea, [data-testid="textInputRoot"] label {
    background: #111111 !important; /* Darker background for inputs */
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    color: #00FFFF !important; /* Ensure input text is bright blue */
    border-radius: 6px;
    transition: border-color 0.2s, box-shadow 0.2s;
}

/* Alerts */
.alert-red {
    background: rgba(255, 40, 80, 0.15); 
    border-left: 6px solid #ff4d4d;
    padding: 15px;
    color: #FFDDDD !important; /* Keep alert text slightly different but visible */
    border-radius: 10px;
}
.alert-green {
    background: rgba(0, 255, 150, 0.1);
    border-left: 6px solid #00ff99;
    padding: 15px;
    color: #CCFFCC !important;
    border-radius: 10px;
}
.alert-yellow {
    background: rgba(255, 255, 0, 0.1); 
    border-left: 6px solid #ffff00;
    padding: 15px;
    color: #FFFFDD !important; 
    border-radius: 10px;
}

/* DataFrames */
.stDataFrame table {
    background: rgba(255,255,255,0.05);
    color: #00FFFF !important;
}

/* Footer & Muted Text */
.footer, .small-muted {
    color: rgba(0, 255, 255, 0.5) !important;
}

</style>
"""

st.set_page_config(page_title="LifeSaver AI", layout="wide", initial_sidebar_state="expanded")
st.markdown(CLASSY_BLACK_UI_CSS, unsafe_allow_html=True)

# ----------------------------
# DATA CLEANING & PREP 
# ----------------------------
@st.cache_data 
def clean_and_prepare_data():
    """Reads, cleans, mocks, and prepares data, fixing the TypeError issue."""
    try:
        df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        st.warning(f"Required file '{CSV_FILENAME}' not found. Using mock data.")
        data = {
            "Heart Rate (BPM)": np.random.randint(60, 110, 100),
            "Blood Oxygen Level (%)": np.random.randint(94, 100, 100),
            "Sleep Duration (hours)": np.random.uniform(5, 9, 100),
            "Stress Level": np.random.randint(1, 10, 100),
            "Systolic BP (mmHg)": np.random.randint(110, 140, 100),
            "Diastolic BP (mmHg)": np.random.randint(70, 90, 100),
        }
        df = pd.DataFrame(data)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    for c in missing_cols:
        if "BP" in c:
            df[c] = np.random.randint(70, 130, size=len(df))
        elif "Stress" in c:
            df[c] = np.random.randint(1, 10, size=len(df))
        else:
            df[c] = 0

    df = df[FEATURE_COLUMNS].copy()

    # CRITICAL FIX: Ensure all data is coerced to numeric BEFORE mean calculation
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df_mean = df.select_dtypes(include=[np.number]).mean() 
    df.fillna(df_mean, inplace=True)

    # Clipping and Synthetic Labeling
    df["Heart Rate (BPM)"] = np.clip(df["Heart Rate (BPM)"], 40, 200)
    df["Blood Oxygen Level (%)"] = np.clip(df["Blood Oxygen Level (%)"], 70, 100)
    df["Systolic BP (mmHg)"] = np.clip(df["Systolic BP (mmHg)"], 60, 220)
    high_hr = df["Heart Rate (BPM)"] > 105
    low_spo2 = df["Blood Oxygen Level (%)"] < 95
    high_sbp = df["Systolic BP (mmHg)"] > 140
    risk_proxies = high_hr.astype(int) + low_spo2.astype(int) + high_sbp.astype(int) 
    df["Emergency_Event"] = np.where(risk_proxies >= 2, 1, 0)

    X = df[FEATURE_COLUMNS]
    Y = df["Emergency_Event"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, Y, scaler

# ----------------------------
# MODEL FUNCTIONS 
# ----------------------------
@st.cache_resource
def train_supervised_models(X_scaled, Y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_FILENAMES[name])
            trained[name] = model
        except Exception as e:
            st.error(f"Error training {name}: {e}")
    return trained

@st.cache_resource
def load_supervised_models():
    df, X_scaled, Y, scaler = clean_and_prepare_data()
    files_exist = all(os.path.exists(fname) for fname in MODEL_FILENAMES.values())
    
    if not files_exist:
        models = train_supervised_models(X_scaled, Y)
    else:
        models = {name: joblib.load(fname) for name, fname in MODEL_FILENAMES.items()}

    return df, models, scaler

@st.cache_resource
def load_unsupervised():
    """Loads or trains ONLY the KMeans model."""
    df, _, _, scaler = clean_and_prepare_data()
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    models = {}

    if os.path.exists(UNSUPERVISED_MODEL_FILE):
        try:
            data = joblib.load(UNSUPERVISED_MODEL_FILE)
            if "KMeans" in data.get("models", {}):
                return {"KMeans": data["models"]["KMeans"]}, data["scaler"]
        except Exception:
            # File corrupted or invalid, proceed to retraining
            pass
    
    # Train only KMeans
    models = {
        "KMeans": KMeans(n_clusters=1, random_state=42, n_init='auto').fit(X_scaled),
    }
    joblib.dump({"models": models, "scaler": scaler}, UNSUPERVISED_MODEL_FILE)
    return models, scaler


def ensemble_predict(models, scaler, input_list, threshold):
    df_in = pd.DataFrame([input_list], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(df_in)
    votes = []
    confidences = []
    for name, model in models.items():
        try:
            pred = model.predict(X_scaled)[0]
            # Use decision_function as a fallback if predict_proba is not available/reliable
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[0][1]
            elif hasattr(model, "decision_function"):
                # Normalize decision function output for SVC
                decision_score = model.decision_function(X_scaled)[0]
                # Crude normalization for confidence
                proba = 1.0 / (1.0 + np.exp(-decision_score)) # Using sigmoid approx. for probability
            else:
                proba = 0.5
                
            votes.append(int(pred))
            confidences.append(float(proba))
            
        except Exception as e:
            st.warning(f"Prediction failed for {name}: {e}. Skipping model.")
            votes.append(0) 
            confidences.append(0.5) 
            
    combined_score = int(np.mean(confidences) * 100)
    status = "EMERGENCY ALERT" if combined_score >= threshold else "NORMAL"
    return status, combined_score, votes, confidences

def predict_unsupervised(models, scaler, data_row):
    """Predicts anomaly status using ONLY the KMeans model."""
    X = np.array(data_row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # K-Means distance from the cluster center (Sole anomaly score)
    d_kmeans = np.linalg.norm(X_scaled - models["KMeans"].cluster_centers_[0])
    anomaly_score = d_kmeans 
    
    # Adjusted thresholds for single distance metric
    if anomaly_score > 1.5:  # High distance = Emergency
        status = "EMERGENCY RISK"
    elif anomaly_score > 1.0: # Moderate distance = Moderate Risk
        status = "MODERATE RISK"
    else:
        status = "NORMAL"

    breakdown = {
        "KMeans Distance (Anomaly Score)": float(d_kmeans),
    }
    return status, anomaly_score, breakdown

# ----------------------------
# EMERGENCY PROTOCOL (SIMULATION)
# ----------------------------
def trigger_ai_protocol(risk_score, caller_number, ambulance_number, threshold):
    st.markdown("### 🚨 AI Emergency Protocol")
    
    # State management for protocol simulation
    if "protocol_state" not in st.session_state:
        st.session_state.protocol_state = "CLEARED"
    if "protocol_start_time" not in st.session_state or st.session_state.protocol_state == "CLEARED":
        st.session_state.protocol_start_time = time.time() # Reset timer when cleared
    
    # Logic when protocol is ACTIVE and AWAITING RESPONSE
    if st.session_state.protocol_state == "AWAITING_RESPONSE":
        
        st.markdown(f"<div class='alert-red'>🚨 EMERGENCY RISK DETECTED (Score: {st.session_state.last_score}%)</div>", unsafe_allow_html=True)
        
        # CRITICAL FIX: Trigger audio automatically if file exists
        if os.path.exists(AI_VOICE_FILE):
            st.audio(AI_VOICE_FILE, format="audio/mp3", autoplay=True)
        else:
            st.info("AI voice audio not found — proceed with visual simulation.")

        # CRITICAL FIX: Use unique keys for buttons to prevent StreamlitAPIException
        col1, col2 = st.columns(2)
        
        # Timer calculation
        elapsed = time.time() - st.session_state.get("protocol_start_time", time.time())
        time_placeholder = st.empty()
        time_placeholder.info(f"Protocol timer: {int(elapsed)}s (auto-call at 30s)")
        
        # Button actions
        if col1.button("Yes — I need help / No response", key="protocol_call_active", use_container_width=True):
            st.session_state.protocol_state = "CALL_INITIATED"
            st.rerun()
        if col2.button("No — I'm fine", key="protocol_safe_active", use_container_width=True):
            st.session_state.protocol_state = "SAFE"
            st.rerun()

        # Timeout check
        if elapsed >= 30 and st.session_state.protocol_state == "AWAITING_RESPONSE":
            st.session_state.protocol_state = "TIMEOUT_CALL"
            st.rerun()

        # Force rerun to update timer every 0.5s while active
        if st.session_state.protocol_state == "AWAITING_RESPONSE":
            time.sleep(0.5)
            st.rerun()

    # Logic when protocol is TERMINATED (Call initiated or safe)
    elif st.session_state.protocol_state in ["CALL_INITIATED", "TIMEOUT_CALL"]:
        ambulance_number = st.session_state.get('ambulance_num', 'N/A')
        caller_number = st.session_state.get('caller_num', 'N/A')
        st.error(f"CALL INITIATED: calling {ambulance_number} from {caller_number}")
        st.markdown(f"<a href='tel:{ambulance_number}' style='color:#5bbcff'>[Click to attempt actual call]</a>", unsafe_allow_html=True)
        st.session_state.protocol_state = "CLEARED" 
        
    elif st.session_state.protocol_state == "SAFE":
        st.success("User confirmed SAFE — protocol terminated.")
        st.session_state.protocol_state = "CLEARED"
        
    # Logic when monitoring is NORMAL/MODERATE (CLEARED state)
    else:
        current_score = st.session_state.get('last_score', risk_score)
        if current_score >= (threshold * 0.5):
            st.markdown(f"<div class='alert-yellow'>⚠ Elevated risk detected (Score: {current_score}%) — monitor closely.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-green'>🟢 Normal (Score: {current_score}%)</div>", unsafe_allow_html=True)
        
        if st.session_state.protocol_state != "CLEARED":
            st.session_state.protocol_state = "CLEARED"

# ----------------------------
# MAIN APPLICATION LOGIC 
# ----------------------------

def main_monitor():
    """Renders the main LifeSaver AI dashboard."""
    st.title("⚡ LifeSaver AI — Health Emergency Monitor")
    st.markdown("<p class='small-muted'>Supervised Ensemble + Unsupervised Anomaly Detection • Classy Black theme</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    df_clean, sup_models, sup_scaler = load_supervised_models()
    # Only load KMeans
    unsup_models, unsup_scaler = load_unsupervised() 

    # Configuration Section (in sidebar)
    with st.sidebar:
        st.header("⚙ Configuration")
        caller_num = st.text_input("Person (Caller) Number", value="8637529171", key="caller_input")
        ambulance_num = st.text_input("Emergency / Ambulance Number", value="9883641449", key="ambulance_input")
        threshold = st.slider("Emergency Alert Threshold (%)", min_value=50, max_value=95, value=70, step=5, key="threshold_input")
        st.markdown("---")
        st.markdown("<div class='small-muted'>Tip: To simulate high risk, try HR=115, SpO2=92, Systolic=145.</div>", unsafe_allow_html=True)
        st.markdown("<div class='footer'>Built for hackathon • LifeSaver AI</div>", unsafe_allow_html=True)
        
    st.session_state.caller_num = caller_num
    st.session_state.ambulance_num = ambulance_num

    # Input Card
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🧪 Enter Live Vitals")
    col1, col2 = st.columns(2)
    hr = col1.number_input("❤️ Heart Rate (BPM)", min_value=30, max_value=220, value=75, key="hr_input")
    spo2 = col2.number_input("🩸 Blood Oxygen (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.1, key="spo2_input")

    col3, col4 = st.columns(2)
    sbp = col3.number_input("💓 Systolic BP (mmHg)", min_value=60, max_value=220, value=120, key="sbp_input")
    dbp = col4.number_input("💥 Diastolic BP (mmHg)", min_value=30, max_value=140, value=80, key="dbp_input")

    col5, col6 = st.columns(2)
    sleep = col5.number_input("🛌 Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1, key="sleep_input")
    stress = col6.slider("😟 Stress Level (1-10)", min_value=1, max_value=10, value=3, key="stress_input")

    st.markdown("</div>", unsafe_allow_html=True)

    input_list = [hr, spo2, sleep, stress, sbp, dbp]
    st.markdown("---")

    # Supervised block
    st.subheader("🔮 Supervised Ensemble Prediction")
    
    supervised_result_placeholder = st.empty()
    
    # Initialize last score for protocol status display if it hasn't been run yet
    if 'last_score' not in st.session_state:
        st.session_state.last_score = 0
    
    with st.form("supervised_form"):
        submitted_supervised = st.form_submit_button("⚡ Run Supervised Prediction", use_container_width=True)
    
        if submitted_supervised:
            status, score, votes, confidences = ensemble_predict(sup_models, sup_scaler, input_list, threshold)
            
            st.session_state.last_score = score
            
            with supervised_result_placeholder.container():
                if status == "EMERGENCY ALERT":
                    st.markdown(f"<div class='alert-red'>🚨 EMERGENCY ALERT — Ensemble Score: {score}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-green'>🟢 NORMAL — Ensemble Score: {score}%</div>", unsafe_allow_html=True)

                st.subheader("📊 Model Breakdown")
                st.write({"Models": list(MODEL_FILENAMES.keys()), "Votes": votes, "Confidences": [round(c, 3) for c in confidences]})
                
                # Initiate protocol state change directly upon submission if criteria met
                if score >= threshold and st.session_state.protocol_state == "CLEARED":
                    st.session_state.protocol_state = "AWAITING_RESPONSE"
                    st.rerun()
    
    # Protocol status check outside the form submission to maintain timer/buttons
    # Show protocol status if active, or show monitoring status based on last score
    trigger_ai_protocol(st.session_state.get('last_score', 0), caller_num, ambulance_num, threshold)

    st.markdown("---")

    # Unsupervised block
    st.subheader("🧠 Unsupervised Anomaly Detection (KMeans Only)")
    if st.button("🔍 Run Unsupervised Detection", key="run_unsupervised", use_container_width=True):
        status_u, sc_u, detail_u = predict_unsupervised(unsup_models, unsup_scaler, input_list)
        if status_u == "EMERGENCY RISK":
            st.markdown(f"<div class='alert-red'>🚨 EMERGENCY ANOMALY DETECTED — Score: {sc_u:.3f}</div>", unsafe_allow_html=True)
        elif status_u == "MODERATE RISK":
            st.markdown(f"<div class='alert-yellow'>⚠ MODERATE ANOMALY — Score: {sc_u:.3f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-green'>🟢 NORMAL PATTERN — Score: {sc_u:.3f}</div>", unsafe_allow_html=True)

        st.subheader("📊 Model Breakdown (KMeans)")
        st.json(detail_u)

    st.markdown("---")


def main():
    # Initialize necessary protocol state variables
    if "protocol_state" not in st.session_state:
        st.session_state.protocol_state = "CLEARED"
    if "last_score" not in st.session_state:
        st.session_state.last_score = 0
    if "protocol_start_time" not in st.session_state:
        # Initialize timer to current time so elapsed time calculation doesn't fail
        st.session_state.protocol_start_time = time.time()
        
    main_monitor()

if __name__ == "__main__":
    main()