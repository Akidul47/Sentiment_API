import sys, importlib, torch, transformers
print("python:", sys.executable)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
m = importlib.import_module("transformers.generation.utils")
print("Has GenerationMixin?", hasattr(m, "GenerationMixin"))
from transformers import pipeline
clf = pipeline("sentiment-analysis")
print(clf("Finally works ✅"))