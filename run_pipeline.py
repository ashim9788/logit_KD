import sys
from train_teacher import train_teacher
from capture_logits import capture_logits
from train_student import train_student
from evaluate import run_evaluation

def main():
    print("=== STARTING UNIFIED DISTILLATION PIPELINE ===")
    
    print("\n>>> STAGE 1: Fine-tuning Teacher Model...")
    train_teacher()
    
    print("\n>>> STAGE 2: Capturing Soft Targets (Logits)...")
    capture_logits()
    
    print("\n>>> STAGE 3: Distilling Knowledge to Student Model...")
    train_student()
    
    print("\n>>> STAGE 4: Running Comprehensive Pipeline Evaluation...")
    run_evaluation()
    
    print("\n=== PIPELINE EXECUTION SUCCESSFUL ===")

if __name__ == "__main__":
    main()