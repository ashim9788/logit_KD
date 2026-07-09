# Unified LLM Knowledge Distillation Pipeline

This repository contains a structured, modular pipeline for transferring medical knowledge and reasoning capabilities from a large language model (Teacher) to a smaller, more efficient model (Student). The pipeline uses fine-tuning, soft-target logit extraction, and a custom sparse KL-Divergence loss function.

## 📁 Repository Structure

The project has been refactored into a modular layout to isolate environment settings, data formatting, model training, logit harvesting, and downstream validation:

```text
├── config.py              # Centralized hyperparameters, model IDs, and directory paths
├── dataset.py             # Custom PyTorch Dataset and causal masking collate function
├── train_teacher.py       # Step 1: Teacher Fine-Tuning (QLoRA)
├── capture_logits.py      # Step 2: Harvesting dark knowledge (Top-K logit distributions)
├── train_student.py       # Step 3: Sparse Offline Logit Distillation
├── evaluate.py            # Step 4: Hybrid Performance Evaluation Pipeline
└── run_pipeline.py        # Master Orchestration Script