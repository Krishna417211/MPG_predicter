import streamlit as st
import joblib
import numpy as np
from typing import Tuple, Dict, Any, List

# --- 1. CONSTANTS ---
MODEL_PATH = 'lasso_model.pkl'
SCALER_PATH = 'std_scaler.pkl'
POLY_PATH = 'poly_feats.pkl'

FEATURE_ORDER = [
    'cylinders', 'displacement', 'horsepower', 'weight', 
    'acceleration', 'model year', 'origin'
]

ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}

# --- 2. STYLING ---
def load_css():
    """Loads and injects custom CSS for styling the app."""
    st.markdown("""
    <style>
    /* General App Styling */
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    
    /* Input Widgets */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Main Action Button */
    .stButton>button {
        background: linear-gradient(90deg, #00F260 0%, #0575E6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0, 242, 96, 0.4);
    }

    /* Prediction Result Metric */
    div[data-testid="stMetricValue"] {
        font-size: 42px;
        color: #00F260;
        text-shadow: 0 0 10px rgba(0, 242, 96, 0.3);
    }
    </style>
""", unsafe_allow_html=True)
    
# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models() -> Tuple[Any, Any, Any]:
    """
    Loads the trained model, scaler, and polynomial features transformer.
    Caches the loaded objects for performance.
    
    Returns:
        A tuple containing the loaded model, scaler, and poly transformer.
        Returns (None, None, None) if any file is not found.
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        poly = joblib.load(POLY_PATH)
        return model, scaler, poly
    except FileNotFoundError:
        st.error(f"⚠️ Error: One or more model files not found. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{POLY_PATH}' are in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ An error occurred loading the models: {e}")
        return None, None, None

# --- 4. UI COMPONENTS ---
def get_user_input() -> Dict[str, float]:
    """
    Displays input widgets for vehicle specifications and returns the data.
    
    Returns:
        A dictionary containing the user-provided feature values.
    """
    input_data = {}
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Engine Details")
        input_data['cylinders'] = st.number_input("Cylinders", min_value=0, max_value=12, value=0, step=1, help="Number of cylinders in the engine.")
        input_data['displacement'] = st.number_input("Displacement (cu. in.)", value=150.0, help="Engine displacement in cubic inches.")
        input_data['horsepower'] = st.number_input("Horsepower", value=100.0, help="Gross horsepower of the engine.")
        input_data['weight'] = st.number_input("Weight (lbs)", value=3000.0, help="Vehicle weight in pounds.")

    with col2:
        st.caption("Performance & Origin")
        input_data['acceleration'] = st.number_input("Acceleration (0-60 mph sec)", value=15.0, help="Time to accelerate from 0 to 60 mph in seconds.")
        input_data['model year'] = st.number_input("Model Year (e.g., 70, 80)", min_value=70, max_value=82, value=80, step=1, help="Vehicle model year (from 1970 to 1982).")
        origin_option = st.selectbox(
            "Origin", 
            options=list(ORIGIN_MAP.keys()), 
            format_func=lambda x: f"{x} - {ORIGIN_MAP[x]}",
            help="Region where the car was manufactured."
        )
        input_data['origin'] = origin_option
        
    return input_data

# --- 5. PREDICTION ---
def run_prediction(input_data: Dict[str, float], model: Any, scaler: Any, poly: Any):
    """
    Processes user input, runs the prediction, and displays the result.
    
    Args:
        input_data: Dictionary of feature values from the user.
        model: The trained prediction model.
        scaler: The fitted standard scaler.
        poly: The fitted polynomial features transformer.
    """
    try:
        # 1. Create a feature array in the correct order
        feature_values = [input_data[feature] for feature in FEATURE_ORDER]
        raw_features = np.array([feature_values])
        
        # 2. Scale the raw features
        scaled_features = scaler.transform(raw_features)
        
        # 3. Generate polynomial features
        poly_features = poly.transform(scaled_features)
        
        # 4. Predict using the model
        prediction = model.predict(poly_features)[0]
        
        # 5. Display the result
        st.success("✅ Calculation Complete!")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Predicted MPG", value=f"{prediction:.1f}")
        with col_res2:
            st.info(f"A car with these specifications is expected to achieve approximately **{prediction:.1f} miles per gallon**.")
            
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        
# --- 6. MAIN APP ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="MPG Predictor", layout="centered", initial_sidebar_state="collapsed")
    load_css()
    
    st.title("🚗 Car MPG Predictor")
    st.markdown("Enter vehicle specifications below to predict its fuel efficiency (MPG).")
    st.divider()
    
    model, scaler, poly = load_models()
    
    if all((model, scaler, poly)):
        input_data = get_user_input()
        st.divider()
        if st.button("🚀 Calculate MPG"):
            run_prediction(input_data, model, scaler, poly)
    else:
        st.warning("Models could not be loaded. The prediction service is unavailable.")
        
if __name__ == "__main__":
    main()