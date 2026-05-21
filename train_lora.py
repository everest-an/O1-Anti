import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from mt_physics_loss import MTQuantumCoherenceLoss

# 基础模型：必须是比赛指定的版本 (需自行向 HuggingFace 请求权限并下载)
MODEL_NAME = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"  # 切换为 NVIDIA 最新的 Reasoning 特化模型

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 因为显存限制，本地可能需要量化加载（比赛最终提交的只看 LoRA 权重，可以用 8bit/4bit 练）
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
        attn_implementation="sdpa"  # 避免缺少 flash-attn 库导致的报错，使用 PyTorch 原生高效注意力
    )

    # Kaggle 强制要求 max rank 32
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,               # <-- 必须遵守比赛限制
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    print("Injecting LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 植入你的 MT-LNN 理论 ----
    # 我们可以通过自定义 Trainer 来覆盖 compute_loss 方法，
    # 从而把 MT_Physics_Loss 加入到梯度反传中。
    custom_mt_loss = MTQuantumCoherenceLoss(lambda_coherence=0.05)
    
    class MTCustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # 获取正常的交叉熵 loss
            outputs = model(**inputs, output_hidden_states=True)
            base_loss = outputs.loss
            
            # 提取隐含层，加上量子或者波函数约束的正则化损失（从 mt_physics_loss 中计算）
            # 使用最后一层隐藏状态作为代表
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
                mt_penalty = custom_mt_loss(hidden_states)
            else:
                mt_penalty = 0.0
            
            total_loss = base_loss + mt_penalty
            return (total_loss, outputs) if return_outputs else total_loss

    # 载入针对数学/推理清洗出的数据集
    from datasets import load_dataset
    print("Loading prepared dataset...")
    try:
        raw_dataset = load_dataset("json", data_files="train_math.jsonl", split="train")
        
        def tokenize_function(examples):
            # 将 messages 转换为 token
            texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=1024)
            
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
    )

    trainer = MTCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # data_collator 需配置
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
