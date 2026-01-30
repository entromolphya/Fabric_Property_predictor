import joblib
import os

# All your pkl files
model_files = [
    # trained_models folder
    'trained_models/random_forest_model_SL_elastane.pkl',
    'trained_models/random_forest_model_SL_ground.pkl',
    'trained_models/random_forest_model_YC_elastane_Denier.pkl',
    'trained_models/random_forest_model_YC_ground_Denier.pkl',
    'trained_models/random_forest_model_YC_ground_Filament.pkl',
    'trained_models/random_forest_model_YC_ground_Ne.pkl',

    # trained_encoders folder
    'trained_encoders/fabrication_label_encoder.pkl',

    # root folder pkl files
    'valid_cotton_ne.pkl',
    'valid_denier.pkl',
    'X_columns.pkl',
]

print("Starting compression...\n")

for file in model_files:
    if os.path.exists(file):
        # Get original size
        original_size = os.path.getsize(file) / 1024 / 1024

        # Load model
        model = joblib.load(file)

        # Save with compression
        joblib.dump(model, file, compress=('gzip', 3))

        # Get new size
        new_size = os.path.getsize(file) / 1024 / 1024

        print(f"✅ {file}")
        print(f"   Before: {original_size:.2f} MB -> After: {new_size:.2f} MB")
        print(
            f"   Saved: {original_size - new_size:.2f} MB ({((original_size - new_size) / original_size) * 100:.1f}%)\n")
    else:
        print(f"❌ File not found: {file}\n")

print("Done! All models compressed.")