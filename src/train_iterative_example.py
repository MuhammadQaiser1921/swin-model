"""
Example: Iterative Model Development
=====================================

Use this approach in Kaggle to avoid reloading data every time you modify the model.

Steps:
1. Run cells 1-2 once to load and prepare data
2. Edit swin_transformer.py as needed
3. Re-run cells 3-4 to rebuild and retrain WITHOUT reloading data
"""

# ============================================================
# CELL 1: Load Data (RUN ONCE)
# ============================================================
from train_video import load_data

print("Loading data...")
data = load_data()

# Save to avoid losing it if kernel restarts
import pickle
with open('/kaggle/working/data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✓ Data loaded and cached")


# ============================================================
# CELL 2: Prepare Datasets (RUN ONCE)
# ============================================================
from train_video import prepare_datasets

print("Preparing datasets...")
train_ds, val_ds, test_ds = prepare_datasets(data)
print("✓ Datasets prepared")


# ============================================================
# CELL 3: Build Model (RE-RUN AFTER EDITING swin_transformer.py)
# ============================================================
from train_video import build_and_compile_model

print("Building model...")
model = build_and_compile_model()


# ============================================================
# CELL 4: Train Model (RE-RUN TO TRAIN)
# ============================================================
from train_video import train_model

print("Training model...")
history = train_model(model, train_ds, val_ds)

# Show results
print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print("="*60)


# ============================================================
# OPTIONAL: Evaluate on Test Set
# ============================================================
if test_ds is not None:
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {test_accuracy:.4f}")
