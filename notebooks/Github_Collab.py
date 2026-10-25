# --- RUN THIS INSIDE GOOGLE COLAB ---

# 1. Download your code layout from your GitHub repo
!git clone https://github.com
%cd bits-assignment-1

# 2. Install your requirements.txt modules onto the cloud machine
!pip install -r requirements.txt

# 3. Import and execute the foundation code you built in VS Code
from src.model_pipeline import load_quantized_model
from src.tools.jira_exporter import export_action_items_to_jira

# This will run on Colab's cloud hardware seamlessly!
model, tokenizer = load_quantized_model()