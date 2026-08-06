import torch
import transformers
import datasets
import streamlit
import fastapi
import sklearn
import pandas
import numpy

print("=" * 50)
print("✅ All Libraries Imported Successfully")
print("=" * 50)

print(f"PyTorch Version      : {torch.__version__}")
print(f"Transformers Version : {transformers.__version__}")
print(f"Datasets Version     : {datasets.__version__}")
print(f"Streamlit Version    : {streamlit.__version__}")
print(f"FastAPI Version      : {fastapi.__version__}")

print("=" * 50)

print("CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("=" * 50)