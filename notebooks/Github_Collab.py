# --- Run these cells in Google Colab ---
# Select Runtime > Change runtime type > T4 GPU before loading the quantized model.

# 1. Clone the project repository
!git clone https://github.com/2025paml587-byte/BITS_PAML_AGAI_G10_A1.git
%cd /content/BITS_PAML_AGAI_G10_A1

# 2. Install project and model dependencies
!pip install -r requirements.txt
!pip install transformers accelerate bitsandbytes

# 3. Import and run the project code
import torch
from src.model_pipeline import load_quantized_model
from src.tools.jira_exporter import export_action_items_to_jira

if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU runtime in Colab before loading the 4-bit model.")

model, tokenizer = load_quantized_model()