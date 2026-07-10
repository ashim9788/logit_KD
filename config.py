import os
import torch

# General Environment Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset Specs
DATASET_ID = "ashim9788/medqa_cot"
DATASET_PATH = "./medqa_cot"  # Local backup path if needed

# Model Identifiers
BASE_7B_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_1_5B_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Artifact Storage Directory Paths
ADAPTER_DIR = "./qwen2.5_med_qlora_output"
LOGITS_DATA_DIR = "./captured_logits_data"
DISTILLED_STUDENT_DIR = "./qwen_1.5b_distilled_student"
EVAL_RESULTS_DIR = "./results"

# Step 1 & 2 Parameters (Teacher Training & Inference)
TOP_K_LOGITS = 50

# Step 3 Parameters (Student Distillation)
TEMPERATURE = 2.0
ALPHA = 0.5
STUDENT_BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
STUDENT_EPOCHS = 3
STUDENT_LR = 5e-5

# Step 4 Parameters (Evaluation)
MAX_NEW_TOKENS = 1024
