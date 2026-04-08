import pandas as pd
from pathlib import Path
import os

# --- STEP 1: THE SAFEVAULT GPS ---
# We look for both the Real and Synthetic files to be 100% sure.
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"

# Check for both versions of the vault
RAW_FILE = DATA_FOLDER / "raw_pcod.csv"
GHOST_FILE = DATA_FOLDER / "synthetic_pcod.csv"

def investigate_vault(file_path, label):
    if not file_path.exists():
        print(f"⚠️ Radar Alert: I can't find the {label} at {file_path}")
        return

    print(f"🔍 Investigating {label}...")
    # We only need the first row to see the titles
    df = pd.read_csv(file_path, nrows=0)
    
    print(f"Found {len(df.columns)} columns. Here is your 'Master List':")
    print("-" * 50)
    
    # We use .tolist() to see exactly how they look in the code
    column_list = df.columns.tolist()
    
    for name in column_list:
        # The [ ] brackets will show you if there's a space at the start or end!
        # Example: [ Age] vs [Age ] vs [Age]
        print(f"👉 [ {name} ]")
    
    print("-" * 50)
    print(f"✅ Python Copy-Paste List for {label}:")
    print(column_list)
    print("-" * 50)
    print("\n")

# --- STEP 2: RUN THE INVESTIGATION ---
print("🛡️ SafeVault Radar: Commencing Name Synchronization...")
print("=" * 60)

investigate_vault(RAW_FILE, "REAL DATA VAULT")
investigate_vault(GHOST_FILE, "GHOST DATA VAULT (Synthetic)")

print("💡 HUMAN TIP: If the names above don't match your app.py EXACTLY,")
print("the 'Brain Error' will happen. Copy the name from the [ brackets ] above.")
print("=" * 60)