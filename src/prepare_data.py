import os
from datasets import load_dataset

def format_dolly(example):
    """
    Formats the databricks-dolly-15k example into a ChatML list of messages.
    Qwen expects the conversation format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    instruction = example['instruction']
    context = example['context']
    response = example['response']
    
    # Combine instruction and context if context is provided
    if context.strip():
        user_content = f"{instruction}\n\nContext:\n{context}"
    else:
        user_content = instruction
        
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    return {"messages": messages}

def main():
    print("Loading databricks/databricks-dolly-15k...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    print("Formatting dataset...")
    # Apply format and remove old columns to save space
    dataset = dataset.map(format_dolly, remove_columns=dataset.column_names)
    
    print("Splitting dataset into 80/10/10 train/val/test...")
    # First split: 80% train, 20% temp
    train_temp = dataset.train_test_split(test_size=0.2, seed=42)
    # Second split: 50% val, 50% test from the temp (so 10% / 10% of total)
    val_test = train_temp['test'].train_test_split(test_size=0.5, seed=42)
    
    os.makedirs("./data/processed", exist_ok=True)
    
    print("Saving to ./data/processed ...")
    train_temp['train'].save_to_disk("./data/processed/train")
    val_test['train'].save_to_disk("./data/processed/val")
    val_test['test'].save_to_disk("./data/processed/test")
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
