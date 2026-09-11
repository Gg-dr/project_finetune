import os
import yaml
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()
    config = load_config(args.config)
    
    print(f"Loading tokenizer {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    
    # Qwen tokenizer might not have a pad token set by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading datasets from {config['dataset_path']}...")
    train_dataset = load_from_disk(os.path.join(config['dataset_path'], "train"))
    val_dataset = load_from_disk(os.path.join(config['dataset_path'], "val"))

    # Select a small subset to run faster in our educational lab
    print("Using a subset of the dataset for faster lab execution (5000 train / 500 val)")
    train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
    val_dataset = val_dataset.select(range(min(500, len(val_dataset))))
    
    # QLoRA configuration
    bnb_config = None
    if config.get("use_qlora", False):
        print("Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        print("Using standard LoRA (bfloat16)...")

    print(f"Loading model {config['model_id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if config.get("use_qlora", False):
        model = prepare_model_for_kbit_training(model)
        
    # LoRA config
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config['target_modules']
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Standard language modeling collator (computes loss on all tokens)
    # This ensures 100% compatibility with the latest bleeding-edge transformers & trl libraries
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = SFTConfig(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        num_train_epochs=config['num_train_epochs'],
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        bf16=config['bf16'],
        report_to="none", # Turn off wandb to keep it simple for the lab
        max_seq_length=512, # Keep max length small to fit in 16GB
    )
    
    def formatting_prompts_func(example):
        """
        Uses the tokenizer's chat template to format the list of messages into a single string.
        """
        # SFTTrainer expects a list of strings if we are batching.
        output_texts = []
        for i in range(len(example['messages'])):
            text = tokenizer.apply_chat_template(example['messages'][i], tokenize=False)
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving adapter to {config['output_dir']}...")
    trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    
    print("Training complete!")

if __name__ == "__main__":
    main()
