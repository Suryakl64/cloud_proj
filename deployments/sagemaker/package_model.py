"""
Packages the ONNX model and custom inference handler into a SageMaker model.tar.gz archive.
"""

import os
import tarfile

def create_model_tar(output_path="deployments/sagemaker/model.tar.gz"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_dir = "model"
    code_dir = "deployments/sagemaker/code"

    print(f"Packaging SageMaker model archive into {output_path}...")
    with tarfile.open(output_path, "w:gz") as tar:
        # Add model files
        model_onnx = os.path.join(model_dir, "model.onnx")
        model_data = os.path.join(model_dir, "model.onnx.data")
        
        if os.path.exists(model_onnx):
            tar.add(model_onnx, arcname="model.onnx")
            print("  Added model.onnx")
        
        if os.path.exists(model_data):
            tar.add(model_data, arcname="model.onnx.data")
            print("  Added model.onnx.data")

        # Also add model_ir9.onnx if present
        model_ir9 = os.path.join(model_dir, "model_ir9.onnx")
        if os.path.exists(model_ir9):
            tar.add(model_ir9, arcname="model_ir9.onnx")
            print("  Added model_ir9.onnx")

        # Add code directory containing inference.py and requirements.txt
        for item in os.listdir(code_dir):
            item_path = os.path.join(code_dir, item)
            tar.add(item_path, arcname=f"code/{item}")
            print(f"  Added code/{item}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] SageMaker archive created successfully: {output_path} ({file_size_mb:.2f} MB)")
    return output_path

if __name__ == "__main__":
    create_model_tar()
