import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import config

def train_teacher():
    print(f"Loading tokenizer for {config.BASE_7B_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_7B_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading base model {config.BASE_7B_ID} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_7B_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print(f"Downloading dataset: {config.DATASET_ID}...")
    train_data = load_dataset(config.DATASET_ID, split="train")
    eval_data = load_dataset(config.DATASET_ID, split="test")

    training_args = SFTConfig(
        output_dir=config.ADAPTER_DIR,
        max_length=2048,
        dataset_text_field="messages",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=3,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )

    print("Starting teacher training execution loop...")
    trainer.train()
    trainer.save_model(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)
    print("Teacher fine-tuning complete!")

if __name__ == "__main__":
    train_teacher()