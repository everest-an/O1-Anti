import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from mt_physics_loss import MTQuantumCoherenceLoss

# Competition-required base model
MODEL_NAME = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


def main():
    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False
    )
    # Ensure a pad token exists (some Nemotron tokenizers only have eos)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. Model (4-bit NF4 quantised) ────────────────────────────────────────
    print("Loading Model (4-bit NF4)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # T4 supports fp16, not bf16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
        attn_implementation="eager",            # Nemotron Hybrid requires eager
    )

    # ── 3. Prepare for kbit training ─────────────────────────────────────────
    # MUST be called before get_peft_model when using 4-bit + gradient checkpointing
    # It enables input-grad hooks and casts norm layers to fp32 for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── 4. LoRA config (rank ≤ 32 — competition requirement) ─────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    print("Injecting LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Custom Trainer with MT physics loss ────────────────────────────────
    custom_mt_loss = MTQuantumCoherenceLoss(lambda_coherence=0.05)

    class MTCustomTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            # Try to get hidden states for MT coherence penalty.
            # The Omni outer model may not support output_hidden_states — fall back gracefully.
            try:
                outputs = model(**inputs, output_hidden_states=True)
            except TypeError:
                outputs = model(**inputs)

            base_loss = outputs.loss
            if base_loss is None:
                raise ValueError(
                    "Model returned loss=None. Ensure 'labels' are present in inputs. "
                    f"Input keys: {list(inputs.keys())}"
                )

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                mt_penalty = custom_mt_loss(outputs.hidden_states[-1])
            else:
                mt_penalty = 0.0

            total_loss = base_loss + mt_penalty
            return (total_loss, outputs) if return_outputs else total_loss

    # ── 6. Dataset ────────────────────────────────────────────────────────────
    from datasets import load_dataset
    print("Loading prepared dataset...")
    train_dataset = []
    try:
        raw_dataset = load_dataset("json", data_files="train_math.jsonl", split="train")

        def tokenize_function(examples):
            texts = [
                tokenizer.apply_chat_template(msg, tokenize=False)
                for msg in examples["messages"]
            ]
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            # Labels = input_ids but with -100 on padding positions
            # (padding tokens must not contribute to the loss)
            enc["labels"] = [
                [
                    -100 if mask == 0 else token_id
                    for token_id, mask in zip(ids, masks)
                ]
                for ids, masks in zip(enc["input_ids"], enc["attention_mask"])
            ]
            return enc

        train_dataset = raw_dataset.map(
            tokenize_function, batched=True, remove_columns=["messages"]
        )
        print(f"Dataset ready: {len(train_dataset)} samples, columns: {train_dataset.column_names}")
    except Exception as e:
        import traceback
        print(f"Dataset loading failed: {e}")
        traceback.print_exc()

    # ── 7. Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="./nemotron-mt-reasoning-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,
        save_steps=200,         # Save only at the very end (saves disk I/O)
        save_total_limit=1,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT-compatible
        dataloader_pin_memory=False,    # Avoids pinned-memory issues with quantised models
        remove_unused_columns=False,    # Keep all columns for custom collator
        report_to="none",               # Disable wandb / tensorboard
    )

    # Pass-through collator: forwards ALL tokenised columns to the model
    # (input_ids, attention_mask, labels) — SimpleDataCollator used to drop labels!
    class PassThroughCollator:
        def __call__(self, examples):
            batch = {}
            for key in examples[0].keys():
                batch[key] = torch.stack(
                    [torch.tensor(e[key]) for e in examples]
                )
            return batch

    # ── 8. Train ──────────────────────────────────────────────────────────────
    if len(train_dataset) > 0:
        trainer = MTCustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=PassThroughCollator(),
        )
        print("Starting MT-guided LoRA training (200 steps)...")
        trainer.train()

        # ── 9. Save adapter ───────────────────────────────────────────────────
        out_dir = "./submission/adapter"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving LoRA adapter to {out_dir} ...")
        model.save_pretrained(out_dir)
        saved = os.listdir(out_dir)
        print(f"Saved files: {saved}")
        for fname in saved:
            fsize = os.path.getsize(os.path.join(out_dir, fname))
            print(f"  {fname}: {fsize:,} bytes")
    else:
        print("ERROR: Dataset is empty — training skipped. Fix dataset loading above.")


if __name__ == "__main__":
    main()
