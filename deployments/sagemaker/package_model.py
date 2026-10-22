"""
Optimized packager for SageMaker model.tar.gz archive.
Only includes necessary files with maximum gzip compression to enable rapid upload.
"""

import os
import tarfile

def create_model_tar(output_path="deployments/sagemaker/model.tar.gz"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_dir = "model"
    code_dir = "deployments/sagemaker/code"

    print(f"Packaging optimized SageMaker model archive into {output_path}...")
    with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
        # Add model definition and weights (NO duplicates)
        model_onnx = os.path.join(model_dir, "model.onnx")
        model_data = os.path.join(model_dir, "model.onnx.data")
        
        if os.path.exists(model_onnx):
            tar.add(model_onnx, arcname="model.onnx")
            print("  Added model.onnx")
        
        if os.path.exists(model_data):
            tar.add(model_data, arcname="model.onnx.data")
            print("  Added model.onnx.data")

        # Add code directory containing inference.py and requirements.txt
        for item in sorted(os.listdir(code_dir)):
            item_path = os.path.join(code_dir, item)
            tar.add(item_path, arcname=f"code/{item}")
            print(f"  Added code/{item}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Compact SageMaker archive created: {output_path} ({file_size_mb:.2f} MB)")
    return output_path

if __name__ == "__main__":
    create_model_tar()
