import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    repo_id = "Xenova/bge-small-en-v1.5"
    local_dir = Path(__file__).resolve().parent.parent / "models" / "bge-small-en-v1.5"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading model.onnx to {local_dir}")
    # If the file is in 'onnx' subfolder on the hub, hf_hub_download puts it in local_dir/onnx/model.onnx
    # We will use local_dir=local_dir
    hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx", local_dir=local_dir)
    print("Downloading tokenizer.json")
    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
    print("Done")

if __name__ == "__main__":
    main()
