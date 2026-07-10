import torch
import json
import re, os
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import config

def load_evaluation_dataset(path_or_id):
    try: return load_from_disk(path_or_id)["test"]
    except: 
        dataset = load_dataset(path_or_id)
        return dataset["test"] if "test" in dataset else dataset["train"]

def generate_responses(model, tokenizer, dataset, batch_size=4):
    model.eval()
    responses = []
    prompts = []
    for item in dataset:
        eval_messages = [msg for msg in item["messages"] if msg["role"] != "assistant"] if "messages" in item else [{"role": "user", "content": item["question"]}]
        prompts.append(tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True))

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(config.DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        for j, out_ids in enumerate(generated_ids):
            responses.append(tokenizer.decode(out_ids[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True).strip())
    return responses

def extract_answer_letter(text):
    match = re.search(r"""###\s*Correct\s*Answer:\s*([A-D])""", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match_fallback = re.search(r"\b([A-D])\b\s*$", text.strip())
    return match_fallback.group(1).upper() if match_fallback else "UNKNOWN"

def run_evaluation():
    eval_dataset = load_evaluation_dataset(config.DATASET_ID)
    references = []
    for item in eval_dataset:
        references.append(next((msg["content"] for msg in item["messages"] if msg["role"] == "assistant"), ""))

    models_to_evaluate = {
        "7B_Base_Instruct": {"path": config.BASE_7B_ID, "is_peft": False, "batch_size": 4},
        "7B_Teacher_QLoRA": {"path": config.BASE_7B_ID, "adapter": config.ADAPTER_DIR, "is_peft": True, "batch_size": 4},
        "1.5B_Base_Instruct": {"path": config.BASE_1_5B_ID, "is_peft": False, "batch_size": 32},
        "1.5B_Distilled_Student": {"path": config.DISTILLED_STUDENT_DIR, "is_peft": False, "batch_size": 32}
    }
    
    model_predictions = {}
    for name, cfg in models_to_evaluate.items():
        print(f"\n--- Evaluating Model: {name} ---")
        tokenizer = AutoTokenizer.from_pretrained(cfg["adapter"] if "adapter" in cfg else cfg["path"], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left" 
        
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if cfg.get("is_peft"): model = PeftModel.from_pretrained(model, cfg["adapter"])
            
        model_predictions[name] = generate_responses(model, tokenizer, eval_dataset, batch_size=cfg["batch_size"])
        del model; del tokenizer; torch.cuda.empty_cache()

    # Evaluation Calculation Engine
    bertscore = load("bertscore")
    results_summary = {}
    ref_letters = [extract_answer_letter(ref) for ref in references]

    for name, preds in model_predictions.items():
        correct = sum(1 for idx, p in enumerate(preds) if extract_answer_letter(p) == ref_letters[idx] and ref_letters[idx] != "UNKNOWN")
        score_metrics = bertscore.compute(predictions=preds, references=references, lang="en", model_type="roberta-large")
        
        results_summary[name] = {
            "strict_accuracy": (correct / len(eval_dataset)) * 100,
            "precision": float(torch.tensor(score_metrics["precision"]).mean()),
            "recall": float(torch.tensor(score_metrics["recall"]).mean()),
            "f1": float(torch.tensor(score_metrics["f1"]).mean())
        }

    print("\n======================= FINAL HYBRID RESULTS =======================")
    for name, m in results_summary.items():
        print(f"{name:<25} | Acc: {m['strict_accuracy']:.2f}% | F1: {m['f1']:.4f}")
    
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)  # Creates the directory automatically if missing
    output_file_path = os.path.join(config.EVAL_RESULTS_DIR, "distillation_eval_results.json")
    
    with open(output_file_path, "w") as f:
        json.dump({"metrics": results_summary, "predictions": model_predictions}, f, indent=4)
        
    print(f"\n Results logged safely to '{output_file_path}'")

if __name__ == "__main__":
    run_evaluation()
