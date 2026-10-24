from datasets import load_dataset
import pandas as pd
import numpy as np
import os


print("Connecting to Hugging Face...")
# Load a tiny, public dataset to verify the connection works
dataset = load_dataset("knkarthick/samsum", split="train") #knkarthick/samsum #rotten_tomatoes
df = dataset.to_pandas()

print("\nSuccessfully connected!")
print(f"Dataset preview:\n{dataset[0]}")

print(df.head())

#print(f"Dataset shape: {df.shape()}")