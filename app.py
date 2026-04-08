import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- STEP 1: SETTINGS ---
st.set_page_config(page_title="SafeVault Analytics", page_icon="🛡️")

# --- STEP 2: LOAD MODEL ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pcod_predictor.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# --- STEP 3: UI DESIGN ---
st.title("🛡️ SafeVault Analytics")
st.markdown("### Privacy-Preserving PCOD Risk Assessment")
st.write("Leveraging **Synthetic AI** to protect patient privacy.")
st.divider()

# --- STEP 4: SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Symptoms")
age = st.sidebar.slider("Age (yrs)", 15, 50, 25)
weight = st.sidebar.number_input("Weight (Kg)", 30.0, 150.0, 60.0)
cycle_ri = st.sidebar.selectbox("Cycle Type", options=["Regular", "Irregular"])
cycle_val = 1 if cycle_ri == "Regular" else 2
pimple = st.sidebar.radio("Frequent Pimples?", ["No", "Yes"])
pimple_val = 1 if pimple == "Yes" else 0
weight_gain = st.sidebar.radio("Sudden Weight Gain?", ["No", "Yes"])
weight_gain_val = 1 if weight_gain == "Yes" else 0
hair_growth = st.sidebar.radio("Excess Facial/Body Hair?", ["No", "Yes"])
hair_growth_val = 1 if hair_growth == "Yes" else 0
skin_dark = st.sidebar.radio("Skin Darkening?", ["No", "Yes"])
skin_dark_val = 1 if skin_dark == "Yes" else 0

# --- STEP 5: PREDICTION & SUGGESTIONS ---
model = load_model()

if st.button("🔍 Analyze My Risk Score"):
    if model is None:
        st.error("⚠️ Model file not found!")
    else:
        try:
            input_dict = {
                'Age (yrs)': age,
                'Weight (Kg)': weight,
                'Cycle(R/I)': cycle_val,
                'Pimples(Y/N)': pimple_val,
                'Weight gain(Y/N)': weight_gain_val,
                'hair growth(Y/N)': hair_growth_val,
                'Skin darkening (Y/N)': skin_dark_val
            }
            user_data = pd.DataFrame([input_dict])

            # Prediction
            prediction = model.predict(user_data)
            prob = model.predict_proba(user_data)[0][1]

            # Results Display
            st.subheader("🩺 Diagnostic Result")
            
            if prediction[0] == 1:
                st.warning(f"**Higher Risk Pattern Detected.** (Confidence: {prob*100:.1f}%)")
                
                # --- HIGH RISK SUGGESTIONS ---
                st.markdown("### 🥗 Recommendations for Management:")
                st.write("* **Consult a Specialist:** Schedule an appointment with a gynecologist or endocrinologist.")
                st.write("* **Low Glycemic Diet:** Focus on whole grains, leafy greens, and lean proteins to manage insulin.")
                st.write("* **Regular Exercise:** Aim for 30 minutes of moderate activity (like brisk walking) to improve metabolism.")
                st.write("* **Monitor Symptoms:** Keep a digital log of your cycle and any skin changes.")
            else:
                st.success(f"**Low Risk Pattern Detected.** (Confidence: {(1-prob)*100:.1f}%)")
                
                # --- LOW RISK / PREVENTATIVE SUGGESTIONS ---
                st.markdown("### 🌿 Preventative Wellness Tips:")
                st.write("* **Maintain Balanced Nutrition:** Avoid excessive processed sugars and refined carbs.")
                st.write("* **Stress Management:** Practice yoga or meditation to keep cortisol levels in check.")
                st.write("* **Consistent Sleep:** Aim for 7-8 hours of quality sleep to maintain hormonal balance.")
                st.write("* **Routine Checkups:** Continue regular annual health screenings.")

            st.divider()

            # --- STEP 6: SEABORN VISUALIZATION ---
            st.subheader("📊 Model Insight: Symptom Weightage")
            fig, ax = plt.subplots(figsize=(8, 4))
            feat_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
            feat_importances = feat_importances.sort_values(ascending=True)
            
            colors = sns.color_palette("viridis", len(feat_importances))
            feat_importances.plot(kind='barh', color=colors, ax=ax)
            ax.set_title("AI Decision Logic (How it weighted your inputs)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"🚧 Communication Error: {e}")

st.divider()
st.caption("SafeVault Analytics — Developed by Sharmila.S | MCA Data Science Project")