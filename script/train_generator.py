import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
import pickle
import os

# --- STEP 1: THE SAFEVAULT GPS ---
# Finding the "Vault" where your data lives.
BASE_DIR = Path(__file__).resolve().parent.parent 
RAW_DATA = BASE_DIR / "data" / "raw_pcod.csv"
OUTPUT_DATA = BASE_DIR / "data" / "synthetic_pcod.csv"
MODEL_FILE = BASE_DIR / "models" / "generator_model.pkl"

print(f"📡 SafeVault GPS: Searching for your data at... {RAW_DATA}")

# --- STEP 2: OPENING THE VAULT ---
if not RAW_DATA.exists():
    print("❌ ERROR: I still can't find 'raw_pcod.csv'.")
    print(f"👉 Please make sure your file is sitting inside: {BASE_DIR}/data/")
else:
    print("📂 FOUND IT! Opening the real PCOD data vault now...")
    real_data = pd.read_csv(RAW_DATA)
    
    # Cleaning up any invisible spaces in the column names to avoid errors
    real_data.columns = real_data.columns.str.strip()

    # --- STEP 3: THE RULEBOOK (Metadata) ---
    # We teach the AI the "DNA" of your medical data.
    print("📝 Creating the rulebook so the AI doesn't get confused...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    # ✨ THE POWER-FIX: Labeling Categories
    # This is CRITICAL. It tells the AI which columns are Yes/No choices.
    categorical_columns = [
        'Cycle(R/I)', 'Pimple(Y/N)', 'Weight gain(Y/N)', 
        'hair growth(Y/N)', 'Skin darkening (Y/N)', 'PCOS (Y/N)'
    ]

    for col in categorical_columns:
        if col in real_data.columns:
            metadata.update_column(column_name=col, sdtype='categorical')
            print(f"✅ Marked '{col}' as a category.")

    # --- STEP 4: THE SPEEDY MATHEMATICIAN (GaussianCopula) ---
    # We swapped CTGAN for GaussianCopula because it's much more accurate 
    # for smaller, tabular datasets. It finishes in seconds!
    print("\n🧮 SafeVault is analyzing the mathematical patterns...")
    generator = GaussianCopulaSynthesizer(metadata)
    generator.fit(real_data)

    # --- STEP 5: SAVING THE KNOWLEDGE ---
    # We save the generator so we can make more ghosts later if needed.
    os.makedirs(MODEL_FILE.parent, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(generator, f)

    # --- STEP 6: CREATING THE GHOST PATIENTS ---
    # Now we ask the AI to generate 1,000 brand-new, safe patients.
    print("👻 Creating 1,000 high-accuracy 'Ghost Patients' (Synthetic Data)...")
    ghost_patients = generator.sample(num_rows=1000)
    
    # Save the ghosts to your data folder
    ghost_patients.to_csv(OUTPUT_DATA, index=False)

    print("-" * 50)
    print(f"🎉 SUCCESS! Your high-quality safe data is ready at: {OUTPUT_DATA}")
    print("🚀 NEXT STEP: Run your Predictor Notebook and watch that accuracy fly!")