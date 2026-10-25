import os
import pandas as pd
import torch
from src.model_pipeline import load_quantized_model

# FIX: Use forward slashes (/) so that the script runs perfectly on Colab's Linux servers
def run_baseline_experiment(data_path: str = "data/experimental_subset.csv"):
    """
    Member 1 Delivery: Experiment E1 (Baseline System).
    Loads 3 sample rows from the experimental subset, passes them through a simple 
    prompt template, and captures the baseline summaries.
    """
    # 1. Load the single source of truth data subset
    if not os.path.exists(data_path):
        # FIX: Changed backslash to forward slash here as well
        raise FileNotFoundError(f"⚠️ {data_path} missing! Run src/DataProcessing_Subsetting.py first.")
    
    df = pd.read_csv(data_path)
    samples = df.head(3)  # Per assignment rule: Evaluate at least 3 outputs
    
    # 2. Check for GPU hardware parity before initializing the core engine
    if not torch.cuda.is_available():
        print("💡 Hardware Check: NVIDIA GPU not detected locally.")
        print("   This script's execution block is designed to be run in Google Colab.")
        return
        
    # 3. Instantiate the engine loader you wrote in model_pipeline.py
    model, tokenizer = load_quantized_model()
    
    baseline_records = []
    print("\n🚀 Executing Baseline Experiment E1 (3 Samples)...")
    
    for idx, row in samples.iterrows():
        dialogue = row['dialogue']
        ground_truth = row['summary']
        
        # 4. Define your simple baseline prompt instruction
        baseline_prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
        
        # 5. Tokenization & Autoregressive Inference Generation Configuration
        inputs = tokenizer(baseline_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,    # Default simple greedy decoding strategy
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Extract and clean generated token sequences
        generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
        # Strip away the prompt prefix to isolate the raw summary response
        response_summary = generated_text.replace(baseline_prompt, "").strip()
        
        print(f"\n--- Processed Record {idx+1}/3 (ID: {row['id']}) ---")
        print(f"Generated Summary:\n{response_summary}")
        
        baseline_records.append({
            "id": row['id'],
            "prompt": baseline_prompt,
            "generated_summary": response_summary,
            "ground_truth": ground_truth
        })
        
    # 6. Export results for Report Section 6 (Baseline Implementation) evidence
    results_df = pd.DataFrame(baseline_records)
    results_df.to_csv("data/e1_baseline_results.csv", index=False)
    print("\n✅ Experiment E1 completed. Results saved to 'data/e1_baseline_results.csv'")

if __name__ == "__main__":
    run_baseline_experiment()
