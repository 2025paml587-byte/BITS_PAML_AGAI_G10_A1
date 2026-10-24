from datasets import load_dataset
import pandas as pd
import numpy as np
import os


#print("Connecting to Hugging Face...")
## Load a tiny, public dataset to verify the connection works
#dataset = load_dataset("knkarthick/samsum", split="train") #knkarthick/samsum #rotten_tomatoes
#df = dataset.to_pandas()

#print("\nSuccessfully connected!")
#print(f"Dataset preview:\n{dataset[0]}")

#print(df.head())

#print(f"Dataset shape: {df.shape()}")

# 1. loading the dataset split from Hugging Face
print(f"Loading SamSum dataset from Hugging Face...")
dataset = load_dataset("knkarthick/samsum")
print(f"Dataset loaded successfully! Dataset splits: {dataset.keys()}")

# 2. converting the dataset to a pandas dataframe
train_df = dataset["train"].to_pandas()
print(f"Train dataset shape: {train_df.shape}")
val_df = dataset["validation"].to_pandas()
print(f"Validation dataset shape: {val_df.shape}")
test_df = dataset["test"].to_pandas()
print(f"Test dataset shape: {test_df.shape}")

#3. Text processing and cleaning
#3.1. Define a function to clean the text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = ''.join(e for e in text if e.isalpha() or e.isspace())
    # Remove extra whitespace
    text = ' '.join(text.split())
    if isinstance(text, str):
        #Replace Windows-style line endings with Unix-style line endings
        return text.replace("\r\n", "\n").strip()
    return text

#3.2. Apply the cleaning function to the text columns in the dataframes
for df in [train_df, val_df, test_df]:
    df['dialogue'] = df['dialogue'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)

#4. Create a Controlled Experiment Subset (100 rows)
# We use a fixed random seed (42) to ensure reproducibility of the random sampling
experimental_subset = train_df.sample(n=100, random_state=42).reset_index(drop=True)

#5. Verify Dataset Characteristics for Report Section 3
print(f"\n--- Experimental Subset Statistics ---")
print(f"Total rows Selected: {len(experimental_subset)}")
print(f"Average Dialogue Length (chars): {experimental_subset['dialogue'].str.len().mean():.2f}")
print(f"Average Summary Length (chars): {experimental_subset['summary'].str.len().mean():.2f}")

print(f"\n--- Sample Data from Experimental Subset ---")
print(experimental_subset.head())

#6. Save the experimental subset to a CSV file for further analysis
output_dir = "data"
#6.1 Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)   
experimental_subset_file = os.path.join(output_dir, "experimental_subset.csv")
print(f"\nSaving experimental subset to {experimental_subset_file}...")
experimental_subset.to_csv(experimental_subset_file, index=False)
#6.2 Saving the cleaned full datasets to CSV files for further analysis
print(f"Saving cleaned datasets to {output_dir}...")
train_file = os.path.join(output_dir, "train_cleaned.csv")
val_file = os.path.join(output_dir, "validation_cleaned.csv")
test_file = os.path.join(output_dir, "test_cleaned.csv")
#6.3 Confirming the saving of the experimental and cleaned datasets
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)    
test_df.to_csv(test_file, index=False)
print(f"Cleaned datasets saved successfully!")