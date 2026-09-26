import sys
import os

def train_models(train_dir: str, model_dir: str):
    print(f"[INFO] Training pipeline executing on {train_dir}...")

if __name__ == '__main__':
    train_models(sys.argv[1] if len(sys.argv) > 1 else "../../dataset/train", "../../models")