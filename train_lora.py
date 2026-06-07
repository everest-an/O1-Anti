import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from mt_physics_loss import MTQuantumCoherenceLoss

# 鍩虹妯″瀷锛氬繀椤绘槸姣旇禌鎸囧畾鐨勭増鏈?(闇€鑷鍚?HuggingFace 璇锋眰鏉冮檺骞朵笅杞?
MODEL_NAME = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"  # 鍒囨崲涓?NVIDIA 鏈€鏂扮殑 Reasoning 鐗瑰寲妯″瀷

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    
    # 鍥犱负鏄惧瓨闄愬埗锛屾湰鍦板彲鑳介渶瑕侀噺鍖栧姞杞斤紙姣旇禌鏈€缁堟彁浜ょ殑鍙湅 LoRA 鏉冮噸锛屽彲浠ョ敤 8bit/4bit 缁冿級
    print("Loading Model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        quantization_config=quant_config,
        trust_remote_code=True,
        attn_implementation="eager"  # 寮哄埗鍥為€€浣跨敤 eager 妯″紡锛岃В鍐充笉鏀寔 sdpa 鐨勯棶棰?
    )

    # Kaggle 寮哄埗瑕佹眰 max rank 32
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,               # <-- 蹇呴』閬靛畧姣旇禌闄愬埗
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    print("Injecting LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 妞嶅叆浣犵殑 MT-LNN 鐞嗚 ----
    # 鎴戜滑鍙互閫氳繃鑷畾涔?Trainer 鏉ヨ鐩?compute_loss 鏂规硶锛?
    # 浠庤€屾妸 MT_Physics_Loss 鍔犲叆鍒版搴﹀弽浼犱腑銆?
    custom_mt_loss = MTQuantumCoherenceLoss(lambda_coherence=0.05)
    
    class MTCustomTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            # 鑾峰彇姝ｅ父鐨勪氦鍙夌喌 loss
            outputs = model(**inputs, output_hidden_states=True)
            base_loss = outputs.loss
            
            # 鎻愬彇闅愬惈灞傦紝鍔犱笂閲忓瓙鎴栬€呮尝鍑芥暟绾︽潫鐨勬鍒欏寲鎹熷け锛堜粠 mt_physics_loss 涓绠楋級
            # 浣跨敤鏈€鍚庝竴灞傞殣钘忕姸鎬佷綔涓轰唬琛?
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
                mt_penalty = custom_mt_loss(hidden_states)
            else:
                mt_penalty = 0.0
            
            total_loss = base_loss + mt_penalty
            return (total_loss, outputs) if return_outputs else total_loss

    # 杞藉叆閽堝鏁板/鎺ㄧ悊娓呮礂鍑虹殑鏁版嵁闆?
    from datasets import load_dataset
    print("Loading prepared dataset...")
    try:
        raw_dataset = load_dataset("json", data_files="train_math.jsonl", split="train")
        
        def tokenize_function(examples):`r`n            texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]`r`n            enc = tokenizer(texts, padding="max_length", truncation=True, max_length=512)`r`n            enc["labels"] = enc["input_ids"].copy()`r`n            return enc
            
        train_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["messages"])
    except Exception as e:
        print(f"Dataset loading failed (make sure to run prepare_dataset.py first): {e}")
        train_dataset = [] 
    
    training_args = TrainingArguments(
        output_dir="./nemotron-mt-reasoning-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,
        save_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Create a simple data collator that just returns the inputs as is (since we're using tokenized data)
    class SimpleDataCollator:
        def __call__(self, examples):
            return {"input_ids": torch.stack([torch.tensor(e['input_ids']) for e in examples])}
    
    trainer = MTCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SimpleDataCollator(),  # Use our custom data collator
    )

    print("Starting specialized MT-guided LoRA tuning...")
    if len(train_dataset) > 0:
        trainer.train()
        print("Saving submission LoRA adapter...")
        model.save_pretrained("./submission/adapter")
    else:
        print("Skipping training because dataset is empty.")

if __name__ == "__main__":
    main()
