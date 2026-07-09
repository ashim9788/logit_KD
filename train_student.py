import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import config
from dataset import OfflineDistillationDataset, collate_fn

def train_student():
    print("Loading student tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_1_5B_ID, trust_remote_code=True)
    assistant_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644

    print(f"Loading Student Model ({config.BASE_1_5B_ID})...")
    student_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_1_5B_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    student_model.gradient_checkpointing_enable()
    student_model.train()

    dataset = OfflineDistillationDataset(config.LOGITS_DATA_DIR)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.STUDENT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id, assistant_start_id=assistant_start_id)
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.STUDENT_LR)
    num_training_steps = (config.STUDENT_EPOCHS * len(dataloader)) // config.ACCUMULATION_STEPS
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    print("Beginning Optimized Logit Distillation...")
    for epoch in range(config.STUDENT_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.STUDENT_EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            target_labels = batch["target_labels"].to("cuda")
            topk_vals = batch["topk_vals"].to("cuda")
            topk_inds = batch["topk_inds"].to("cuda")

            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits.float()

            shift_student = student_logits[..., :-1, :].contiguous()
            shift_labels = target_labels[..., 1:].contiguous()
            shift_topk_vals = topk_vals[..., :-1, :].contiguous()
            shift_topk_inds = topk_inds[..., :-1, :].contiguous()

            true_loss_mask = (shift_labels != -100) & (shift_labels != pad_id)

            loss_ce = F.cross_entropy(
                shift_student.view(-1, shift_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )

            log_p_student_all = F.log_softmax(shift_student / config.TEMPERATURE, dim=-1)
            gathered_student_log_probs = torch.gather(log_p_student_all, dim=-1, index=shift_topk_inds)
            p_teacher = F.softmax(shift_topk_vals / config.TEMPERATURE, dim=-1)

            loss_kl_per_token = F.kl_div(gathered_student_log_probs, p_teacher, reduction="none").sum(dim=-1)
            loss_kl = (loss_kl_per_token * true_loss_mask).sum() / true_loss_mask.sum() if true_loss_mask.sum() > 0 else torch.tensor(0.0, device="cuda")
            loss_kl = loss_kl * (config.TEMPERATURE ** 2)

            loss = (config.ALPHA * loss_kl) + ((1.0 - config.ALPHA) * loss_ce)
            loss = loss / config.ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % config.ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.ACCUMULATION_STEPS
            progress_bar.set_postfix({"Loss": f"{loss.item() * config.ACCUMULATION_STEPS:.4f}", "KL": f"{loss_kl.item():.4f}", "CE": f"{loss_ce.item():.4f}"})

    student_model.save_pretrained(config.DISTILLED_STUDENT_DIR)
    tokenizer.save_pretrained(config.DISTILLED_STUDENT_DIR)
    print("Distillation execution complete!")

if __name__ == "__main__":
    train_student()