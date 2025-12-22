import torch
import coremltools as ct
import sys
import os
import glob

# --- Configuration ---
# 1. Image size: SHARP uses 1536x1536 as its internal resolution.
# This is required by the patch-based encoder architecture.
INPUT_SIZE = (1536, 1536) 
MODEL_PATH = "model/sharp_2572gikvuh.pt" # Check if your file is here or in root
OUTPUT_NAME = "SHARP_VisionPro.mlpackage"

# --- 1. Setup Environment & Imports ---
print(f"🔍 Setting up paths...")
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")

# Add 'src' to python path so we can import the internal modules
if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"   -> Added {src_path} to sys.path")

try:
    print("⏳ Attempting to import SHARP model class...")
    from sharp.models import PredictorParams, create_predictor
    print("   ✅ Successfully imported SHARP model classes.")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("   ⚠️ DEBUG: Listing files in src/ to help you find the import:")
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".py"):
                print(f"      - {os.path.relpath(os.path.join(root, file), src_path)}")
    print("\n   ACTION REQUIRED: Edit the 'from ... import ...' line in this script to match the file structure found above.")
    sys.exit(1)

# --- 2. Load the Model ---
print(f"\n📦 Loading weights from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    # Fallback: check root if not in model/
    if os.path.exists("sharp_2572gikvuh.pt"):
        MODEL_PATH = "sharp_2572gikvuh.pt"
        print(f"   -> Found weights in root: {MODEL_PATH}")
    else:
        print(f"   ❌ Error: Could not find {MODEL_PATH}")
        sys.exit(1)

try:
    # Initialize model with default parameters
    print("   -> Creating model with default parameters...")
    model = create_predictor(PredictorParams())
    
    # Load state dict
    print(f"   -> Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    
    # Sometimes weights are wrapped in "state_dict" or "model" keys
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("   ✅ Model loaded and set to eval mode.")
except Exception as e:
    print(f"   ❌ Model load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 3. Trace the Model ---
print(f"\n🔦 Tracing model with dummy input {INPUT_SIZE}...")
try:
    # Create dummy input: (Batch, Channels, Height, Width)
    dummy_image = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    
    # The model requires a disparity_factor tensor
    # Based on predict.py: disparity_factor = f_px / width
    # For a 1536x1536 image, using a typical focal length assumption
    # f_px is typically around width (for 90deg FOV) or 2*width (for narrower FOV)
    # Using a reasonable default: f_px = width (so disparity_factor = 1.0)
    dummy_disparity_factor = torch.tensor([1.0])
    
    # JIT Trace with both inputs
    print("   -> Tracing with image and disparity_factor...")
    traced_model = torch.jit.trace(model, (dummy_image, dummy_disparity_factor))
    print("   ✅ Model traced successfully.")
except Exception as e:
    print(f"   ❌ Tracing failed: {e}")
    print("   (Hint: If this is a control flow error, try torch.jit.script instead, though trace is preferred for Core ML)")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 4. Convert to Core ML ---
print(f"\n🔄 Converting to Core ML (ML Program)...")
try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=dummy_image.shape, name="color_image"),
            ct.TensorType(shape=dummy_disparity_factor.shape, name="disparity_factor")
        ],
        convert_to="mlprogram", # Essential for AVP/Transformer models
        compute_precision=ct.precision.FLOAT16 # FP16 is standard for Vision Pro
    )
    
    # Set metadata
    mlmodel.short_description = "Apple SHARP Model for 3D Gaussian Splatting"
    mlmodel.author = "Converted Locally"
    mlmodel.input_description["color_image"] = "Input RGB image"
    mlmodel.input_description["disparity_factor"] = "Disparity factor (f_px / width)"
    
    print("   ✅ Conversion complete.")
except Exception as e:
    print(f"   ❌ Core ML Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 5. Save ---
print(f"\n💾 Saving to {OUTPUT_NAME}...")
mlmodel.save(OUTPUT_NAME)
print(f"🎉 Done! File saved to: {os.path.abspath(OUTPUT_NAME)}")
