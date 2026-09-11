import os
import yaml
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokenize_and_format(example, tokenizer, max_length=512):
    """
    Applies the chat template, tokenizes, and builds labels.
    Labels are set equal to input_ids so the model learns to predict every token.
    """
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(
        text, truncation=True, max_length=max_length, padding=False
    )
    # For causal LM training, labels = input_ids (the trainer shifts them internally)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()
    config = load_config(args.config)

    print(f"Loading tokenizer {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    # Qwen tokenizer might not have a pad token set by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading datasets from {config['dataset_path']}...")
    train_dataset = load_from_disk(os.path.join(config["dataset_path"], "train"))
    val_dataset = load_from_disk(os.path.join(config["dataset_path"], "val"))

    # Select a small subset to run faster in our educational lab
    print("Using a subset of the dataset for faster lab execution (5000 train / 500 val)")
    train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
    val_dataset = val_dataset.select(range(min(500, len(val_dataset))))

    # Tokenize the datasets up-front so the Trainer receives plain tensors
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda ex: tokenize_and_format(ex, tokenizer), remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda ex: tokenize_and_format(ex, tokenizer), remove_columns=val_dataset.column_names
    )

    # --- Model loading ---
    bnb_config = None
    if config.get("use_qlora", False):
        print("Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("Using standard LoRA (bfloat16)...")

    print(f"Loading model {config['model_id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if config.get("use_qlora", False):
        model = prepare_model_for_kbit_training(model)

    # --- LoRA adapter ---
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Standard data collator for causal language modelling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training arguments (uses the rock-stable transformers.TrainingArguments) ---
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        bf16=config["bf16"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {config['output_dir']}...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print("Training complete!")


if __name__ == "__main__":
    main()
