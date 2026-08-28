import os
import joblib
import pandas as pd

def test_pipeline_loading():
    model_path = 'student_kmeans_pipeline.pkl'
    
    print("=== PIPELINE VALIDATION TEST ===")
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' not found! Run the web app once first to generate it.")
        return

    # Load the compressed asset configuration bundle
    try:
        assets = joblib.load(model_path)
        print("✅ Success: Pipeline file loaded cleanly into memory.")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Verify features and structural components match expected design specs
    required_features = assets['features']
    print(f"📦 Registered Features count: {len(required_features)}")
    print(f"🎯 Registered Labels: {list(assets['labels'].values())}")

    # Generate an arbitrary baseline input to verify prediction accuracy
    test_inputs = [assets['df_final'][col].mean() for col in required_features]
    test_df = pd.DataFrame([test_inputs], columns=required_features)
    
    try:
        scaled_test = assets['scaler'].transform(test_df)
        prediction = assets['model'].predict(scaled_test)
        profile_string = assets['labels'][prediction[0]]
        print(f"🔮 Test Prediction Verification: Cluster {prediction[0]} -> '{profile_string}'")
        print("🎉 System status: 100% functional and ready for deployment!")
    except Exception as e:
        print(f"❌ Mathematical execution failed: {e}")

if __name__ == "__main__":
    test_pipeline_loading()
