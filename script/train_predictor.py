import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 1: THE SAFEVAULT GPS ---
# We find your data folders no matter where the script is running.
BASE_DIR = Path(__file__).resolve().parent.parent 
RAW_DATA = BASE_DIR / "data" / "raw_pcod.csv"
GHOST_DATA = BASE_DIR / "data" / "synthetic_pcod.csv"
MODEL_FILE = BASE_DIR / "models" / "pcod_predictor.pkl"

print("SafeVault GPS: Synchronizing Real and Synthetic Vaults...")

# --- STEP 2: LOADING & CLEANING ---
if not RAW_DATA.exists() or not GHOST_DATA.exists():
    print("ERROR: Missing data files! Make sure both raw and synthetic CSVs are in the /data folder.")
else:
    real_df = pd.read_csv(RAW_DATA)
    ghost_df = pd.read_csv(GHOST_DATA)

    # Standardize column names by removing hidden spaces
    real_df.columns = real_df.columns.str.strip()
    ghost_df.columns = ghost_df.columns.str.strip()

    # --- STEP 3: THE SMART COLUMN FINDER ---
    # This prevents the 'KeyError' by searching for keywords!
    def find_column(df, keyword):
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None

    # We define the "Golden Symptoms" keywords
    keywords = ['age', 'weight', 'cycle', 'pimple', 'gain', 'hair', 'darkening']
    features = []

    for k in keywords:
        col_name = find_column(real_df, k)
        # We ensure it's a symptom and not the final answer
        if col_name and 'PCOS' not in col_name.upper():
            features.append(col_name)

    # Find the Target (The Result column)
    target = find_column(real_df, 'PCOS')

    print(f"Symptoms Found: {features}")
    print(f"Target Found: {target}")

    # --- STEP 4: THE HYBRID MIX & MANUAL BALANCE ---
    # We clean the data of any "NaN" (empty) rows first
    real_df = real_df.dropna(subset=[target])
    ghost_df = ghost_df.dropna(subset=[target])

    # Combine 100% of Real Data + some Ghost Data for privacy & scale
    X_hybrid = pd.concat([real_df[features], ghost_df[features].head(300)])
    y_hybrid = pd.concat([real_df[target], ghost_df[target].head(300)])

    combined = pd.concat([X_hybrid, y_hybrid], axis=1).dropna()

    # MANUAL BALANCING: We duplicate the 'Yes' cases to match 'No' cases.
    # This stops the AI from guessing and forces it to LEARN.
    df_0 = combined[combined[target] == 0]
    df_1 = combined[combined[target] == 1]

    print(f"Initial Balance: Healthy({len(df_0)}) vs PCOD({len(df_1)})")

    # Upsampling the PCOD cases
    df_1_upsampled = df_1.sample(len(df_0), replace=True, random_state=42)
    final_df = pd.concat([df_0, df_1_upsampled])

    X = final_df[features]
    y = final_df[target]

    # --- STEP 5: TRAINING THE EXPERT JURY ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("The Jury of 500 Doctors is studying the Hybrid Vault...")
    jury = RandomForestClassifier(
        n_estimators=500, 
        max_depth=15, 
        class_weight='balanced',
        random_state=42
    )
    jury.fit(X_train, y_train)

    # --- STEP 6: RESULTS & SAVING ---
    score = jury.score(X_test, y_test)
    print("-" * 40)
    print(f"FINAL SAFEVAULT ACCURACY: {score * 100:.2f}%")
    print("-" * 40)

     # --- STEP 7: VISUAL ANALYTICS FOR VIVA ---
    print("Generating Visual Analytics...")

    # A. Feature Importance Chart
    feat_importances = pd.Series(jury.feature_importances_, index=features)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(7).plot(kind='barh', color='teal')
    plt.title('Which Symptoms are Most Important?')
    plt.savefig(BASE_DIR / 'models' / 'feature_importance.png') # Saves the image!
    plt.show()

    # B. Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, jury.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No PCOD', 'PCOD'])
    disp.plot(cmap='Blues')
    plt.title('SafeVault Accuracy Breakdown')
    plt.savefig(BASE_DIR / 'models' / 'confusion_matrix.png') # Saves the image!
    plt.show()

    # Save the brain so app.py can use it
    os.makedirs(MODEL_FILE.parent, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(jury, f)

    print(f"Knowledge secured at: {MODEL_FILE}")
    print("FINAL STEP: Run 'streamlit run app.py' to launch the website!")
