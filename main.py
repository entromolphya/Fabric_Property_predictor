
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import re
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Fabric Property Predictor",
    page_icon="🧵",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Fabric Property Prediction 🧵")
st.markdown("This application predicts Stitch Length and Yarn Count based on fabric properties.")

# ================================ LOAD TRAINED ARTIFACTS ================================

TARGET_COLS = [
    "SL_ground",
    "SL_elastane",
    "YC_ground_Ne",
    "YC_ground_Denier",
    "YC_ground_Filament",
    "YC_elastane_Denier"
]

ENCODERS_DIR = 'trained_encoders'
MODELS_DIR = 'trained_models'

FABRICATION_LE_PATH = os.path.join(ENCODERS_DIR, 'fabrication_label_encoder.pkl')
X_COLUMNS_PATH = 'X_columns.pkl'
VALID_COTTON_NE_PATH = 'valid_cotton_ne.pkl'
VALID_DENIER_PATH = 'valid_denier.pkl'

# Load artifacts - ALL using joblib now!
le = joblib.load(FABRICATION_LE_PATH)
X_columns = joblib.load(X_COLUMNS_PATH)              # ✅ Changed from pickle
VALID_COTTON_NE = joblib.load(VALID_COTTON_NE_PATH)  # ✅ Changed from pickle
VALID_DENIER = joblib.load(VALID_DENIER_PATH)        # ✅ Changed from pickle

# Load models
rf_models = {}
for target in TARGET_COLS:
    model_filename = os.path.join(MODELS_DIR, f'random_forest_model_{target}.pkl')
    rf_models[target] = joblib.load(model_filename)

st.success("Models and artifacts loaded successfully!")

# ================================ CONSTANTS AND HELPER FUNCTIONS ================================

FIBERS = [
    "Cotton", "Recycled_Cotton", "Polyester", "Elastane",
    "Viscose", "Modal", "Acrylic", "Wool", "Sea_shell", "Other"
]

FIBER_MAP = {
    "cotton": "Cotton",
    "organic cotton": "Cotton",
    "bci cotton": "Cotton",
    "recycle cotton": "Recycled_Cotton",
    "recycled cotton": "Recycled_Cotton",
    "recyle cotton": "Recycled_Cotton",
    "polyester": "Polyester",
    "polyster": "Polyester",
    "elastane": "Elastane",
    "elastan": "Elastane",
    "lycra": "Elastane",
    "viscose": "Viscose",
    "modal": "Modal",
    "acrylic": "Acrylic",
    "wool": "Wool",
    "sea shell": "Sea_shell"
}

mapping_dict = {
    'S/J': 'S/J',
    'LS/J': 'L-S/J',
    'L-S/J': 'L-S/J',
    'INTERLOCK': 'Interlock',
    'LINTERLOCK': 'L-Interlock',
    'L-INTERLOCK': 'L-Interlock',
    'PIQUE': 'Pique',
    'L-PIQUE': 'L-Pique',
    'LPIQUE': 'L-Pique',
    'MESH': 'Mesh',
    'L-MESH': 'L-Mesh',
    'LMESH': 'L-Mesh',
    'SINGLELACOSTE': 'Single Lacoste',
    '1X1RIB': '1X1 Rib',
    '1X1L-RIB': '1X1 L-Rib',
    '1X1LRIB': '1X1 L-Rib',
    '2X2RIB': '2X2 Rib',
    '2X2L-RIB': '2X2 L-Rib',
    '2X2LRIB': '2X2 L-Rib',
    '2X1RIB': '2X1 Rib',
    '2X1L-RIB': '2X1 L-Rib',
    '2X1LRIB': '2X1 L-Rib',
    '4X2RIB': '4X2 Rib',
    '4X2L-RIB': '4X2 L-Rib',
    '4X2LRIB': '4X2 L-Rib',
    '5X2RIB': '5X2 Rib',
    '5X2L-RIB': '5X2 L-Rib',
    '5X2LRIB': '5X2 L-Rib',
    'OTTOMANRIB': 'Ottoman Rib',
    'L-OTTOMANRIB': 'Ottoman Rib',  # Map to existing class
    'LOTTOMANRIB': 'Ottoman Rib',   # Map to existing class
}

all_colors = ['Average', 'Black', 'Dark', 'Light', 'Medium', 'Melange', 'Wash', 'White']


def split_composition(comp):
    """Converts a raw composition string into normalized fiber percentage columns."""
    values = {f"{fiber}_pct": 0.0 for fiber in FIBERS}

    if pd.isna(comp):
        return pd.Series(values)

    comp = comp.lower().replace("\xa0", " ").replace("-", " ")
    comp = re.sub(r"\s+", " ", comp)

    matches = re.findall(r"(\d+\.?\d*)\s*%\s*([a-z\s]+)", comp)
    sorted_fiber_map_items = sorted(FIBER_MAP.items(), key=lambda item: len(item[0]), reverse=True)

    for pct, raw_fiber in matches:
        raw_fiber = raw_fiber.strip()
        raw_fiber = re.sub(r'[^a-z\s]', '', raw_fiber)

        mapped_fiber = None
        for key, canonical in sorted_fiber_map_items:
            if re.search(r'\b' + re.escape(key) + r'\b', raw_fiber):
                mapped_fiber = canonical
                break

        if mapped_fiber is None:
            mapped_fiber = "Other"

        values[f"{mapped_fiber}_pct"] += float(pct) / 100.0

    return pd.Series(values)


def apply_constraints(X, y_pred):
    """Apply physical constraints based on elastane content."""
    y_pred = y_pred.copy()

    if "Elastane_pct" in X.columns:
        no_elastane = X["Elastane_pct"] == 0
    else:
        no_elastane = pd.Series([True] * len(X), index=X.index)

    y_pred.loc[no_elastane, "SL_elastane"] = 0.0
    y_pred.loc[no_elastane, "YC_elastane_Denier"] = 0.0

    return y_pred


def snap_to_valid(value, valid_list):
    """Snap a predicted value to the nearest valid value."""
    if value <= 0 or len(valid_list) == 0:
        return None
    return min(valid_list, key=lambda x: abs(x - value))


def format_stitch_length(row, thr=0.05):
    """Format stitch length output."""
    g = round(float(row["SL_ground"]), 2)
    e = round(float(row["SL_elastane"]), 2)

    if e >= thr:
        return f"{g:.2f}/{e:.2f}"
    else:
        return f"{g:.2f}"


def format_yarn_count(row):
    """Format yarn count output."""
    parts = []

    ne = row["YC_ground_Ne"]
    den = row["YC_ground_Denier"]
    fil = row["YC_ground_Filament"]
    ela = row["YC_elastane_Denier"]

    snapped_ne = snap_to_valid(ne, VALID_COTTON_NE)
    snapped_den = snap_to_valid(den, VALID_DENIER)
    snapped_ela = snap_to_valid(ela, VALID_DENIER)

    if snapped_ne is not None:
        parts.append(f"{int(snapped_ne)}/1")
    elif snapped_den is not None and fil > 5:
        parts.append(f"{int(snapped_den)}D/{int(round(fil))}F")

    if snapped_ela is not None:
        parts.append(f"{int(snapped_ela)}D")

    return ", ".join(parts) if parts else "N/A"


# ================================ USER INPUT INTERFACE ================================

st.subheader("Enter Fabric Properties:")

user_fabrication = st.selectbox(
    "Fabrication",
    options=le.classes_,
    help="Select the type of fabric construction."
)

user_composition = st.text_input(
    "Composition",
    value="95% Cotton, 5% Elastane",
    help="Enter the fiber composition (e.g., 80% Cotton, 15% Recycled Cotton, 5% Elastane)"
)

col1, col2 = st.columns(2)
with col1:
    user_mc_dia = st.number_input("M/C Dia", min_value=1, value=28)
    user_gauge = st.number_input("Gauge", min_value=1, value=24)
with col2:
    user_gsm = st.number_input("GSM", min_value=50, value=180)
    user_color = st.selectbox("Color", options=all_colors, index=5)  # Default: Melange

predict_button = st.button("🔮 Predict", type="primary")

# ================================ PREDICTION LOGIC ================================

if predict_button:
    sample_input = {
        'Fabrication': user_fabrication,
        'Composition': user_composition,
        'M/C Dia': user_mc_dia,
        'Gauge': user_gauge,
        'Color': user_color,
        'GSM': user_gsm,
    }

    input_df = pd.DataFrame([sample_input])

    # 1. Fabrication preprocessing (NO uppercase since dropdown already matches le.classes_)
    # The dropdown uses le.classes_ directly, so just encode it
    input_df['Fabrication'] = le.transform(input_df['Fabrication'])

    # 2. Color one-hot encoding
    for color_option in all_colors:
        input_df[f'Color_{color_option}'] = (input_df['Color'] == color_option)
    input_df.drop(columns=['Color'], inplace=True)

    # 3. Composition parsing
    composition_df = input_df["Composition"].apply(split_composition)
    input_df = pd.concat([input_df.drop(columns=["Composition"]), composition_df], axis=1)

    # 4. Ensure numeric types
    input_df['GSM'] = pd.to_numeric(input_df['GSM'], errors='coerce')
    input_df['M/C Dia'] = pd.to_numeric(input_df['M/C Dia'], errors='coerce')
    input_df['Gauge'] = pd.to_numeric(input_df['Gauge'], errors='coerce')

    # 5. Align columns to X_columns
    aligned_input_df = pd.DataFrame(columns=X_columns)
    for col in X_columns:
        if col in input_df.columns:
            aligned_input_df[col] = input_df[col]
        elif col.startswith('Color_'):
            aligned_input_df[col] = False
        elif col.endswith('_pct'):
            aligned_input_df[col] = 0.0
        else:
            aligned_input_df[col] = 0

    # 6. Convert bool to int for model compatibility
    for col in aligned_input_df.columns:
        if aligned_input_df[col].dtype == 'bool':
            aligned_input_df[col] = aligned_input_df[col].astype(int)

    # 7. Ensure column order matches X_columns
    aligned_input_df = aligned_input_df[X_columns]

    # 8. Make predictions
    y_pred = pd.DataFrame(
        {t: rf_models[t].predict(aligned_input_df) for t in TARGET_COLS},
        index=aligned_input_df.index
    )

    # 9. Apply physical constraints
    y_pred = apply_constraints(aligned_input_df, y_pred)

    # 10. Format output
    formatted_output = pd.DataFrame({
        "Stitch Length": y_pred.apply(format_stitch_length, axis=1),
        "Yarn Count": y_pred.apply(format_yarn_count, axis=1)
    })

    # Display results
    st.subheader("📊 Prediction Results")
    st.table(formatted_output)

    # Optional: Show raw predictions
    with st.expander("🔍 View Raw Predictions"):
        st.dataframe(y_pred.round(3))