import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import config

def capture_logits():
    os.makedirs(config.LOGITS_DATA_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_7B_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_7B_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Merging fine-tuned LoRA adapters into Teacher...")
    teacher_model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    teacher_model.eval()

    dataset = load_dataset(config.DATASET_ID, split="train")
    print("Processing and capturing teacher logits...")

    for idx, item in enumerate(tqdm(dataset)):
        messages = item["messages"]
        text_content = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        inputs = tokenizer(text_content, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        input_ids = inputs["input_ids"][0].cpu().numpy()

        with torch.no_grad():
            outputs = teacher_model(**inputs)
            logits = outputs.logits[0]
            topk_vals, topk_inds = torch.topk(logits, k=config.TOP_K_LOGITS, dim=-1)
            topk_vals = topk_vals.to(torch.float16).cpu().numpy()
            topk_inds = topk_inds.to(torch.int32).cpu().numpy()

        np.savez_compressed(
            os.path.join(config.LOGITS_DATA_DIR, f"sample_{idx}.npz"),
            input_ids=input_ids,
            topk_vals=topk_vals,
            topk_inds=topk_inds
        )
    print(f"Logit capture complete! Saved to {config.LOGITS_DATA_DIR}")

if __name__ == "__main__":
    capture_logits()