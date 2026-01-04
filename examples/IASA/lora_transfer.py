import os
import json
import torch

def extract_and_save_lora_adapter(
    input_ckpt_path: str,
    output_dir: str,
    adapter_name: str,
    lora_config: dict,
):
    """
    提取LoRA权重，保存为PEFT格式(adapter_config.json+adapter_model.bin)
    """
    adapter_dir = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_dir, exist_ok=True)
    ckpt = torch.load(input_ckpt_path, map_location='cpu')
    lora_ckpt = {}
    for k, v in ckpt.items():
        if '.lora_A.' in k or '.lora_B.' in k:
            k_new = k.replace('.lora_A.default.weight', f'.lora_A.{adapter_name}.weight')
            k_new = k_new.replace('.lora_B.default.weight', f'.lora_B.{adapter_name}.weight')
            lora_ckpt[k_new] = v
    torch.save(lora_ckpt, os.path.join(adapter_dir, 'adapter_model.bin'))
    with open(os.path.join(adapter_dir, 'adapter_config.json'), 'w') as f:
        json.dump(lora_config, f, indent=2)
    print(f"[SUCCESS] Saved {adapter_name} adapter to {adapter_dir}")

# 用你的LoRA配置
lora_config = {
    "r": 128,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "init_lora_weights": "kaiming",
    "target_modules": ["q", "k", "v", "o", "ffn.0", "ffn.2"],
    "training_strategy": "auto",
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM"
}

# 保存 teacher 和 student
extract_and_save_lora_adapter(
    input_ckpt_path="./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    output_dir="./checkpoints/Original",
    adapter_name="teacher",
    lora_config=lora_config
)
extract_and_save_lora_adapter(
    input_ckpt_path="./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    output_dir="./checkpoints/Original",
    adapter_name="student",
    lora_config=lora_config
)
