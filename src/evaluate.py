import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_inference_model(base_model_id, adapter_path=None, use_qlora=False):
    print(f"Loading model (adapter={adapter_path}, QLoRA={use_qlora})")
    
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    # Extract only the user prompt to pass to the model
    user_messages = [msg for msg in messages if msg['role'] != 'assistant']
    prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)
        
    # Decode and remove the prompt portion from the output
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "./data/processed"
    
    print("Loading test dataset...")
    test_dataset = load_from_disk(os.path.join(dataset_path, "test")).select(range(20)) # test on 20 examples for simplicity
    
    rouge = evaluate.load('rouge')
    
    # 1. Base Model
    print("\n--- BASE MODEL ---")
    model_base, tokenizer_base = load_inference_model(base_model_id)
    base_responses = [generate_response(model_base, tokenizer_base, ex['messages']) for ex in test_dataset]
    del model_base; torch.cuda.empty_cache()
    
    # 2. LoRA Model
    print("\n--- LORA MODEL ---")
    if os.path.exists("./models/lora"):
        model_lora, tokenizer_lora = load_inference_model(base_model_id, "./models/lora", use_qlora=False)
        lora_responses = [generate_response(model_lora, tokenizer_lora, ex['messages']) for ex in test_dataset]
        del model_lora; torch.cuda.empty_cache()
    else:
        print("LoRA adapter not found. Skipping...")
        lora_responses = ["Model not found"] * len(test_dataset)
        
    # 3. QLoRA Model
    print("\n--- QLORA MODEL ---")
    if os.path.exists("./models/qlora"):
        model_qlora, tokenizer_qlora = load_inference_model(base_model_id, "./models/qlora", use_qlora=True)
        qlora_responses = [generate_response(model_qlora, tokenizer_qlora, ex['messages']) for ex in test_dataset]
        del model_qlora; torch.cuda.empty_cache()
    else:
        print("QLoRA adapter not found. Skipping...")
        qlora_responses = ["Model not found"] * len(test_dataset)
        
    ground_truths = [ex['messages'][-1]['content'] for ex in test_dataset]
    prompts = [ex['messages'][0]['content'] for ex in test_dataset]
    
    # Compute metrics
    print("\nComputing ROUGE scores...")
    print("Base ROUGE:", rouge.compute(predictions=base_responses, references=ground_truths)['rougeL'])
    if os.path.exists("./models/lora"):
        print("LoRA ROUGE:", rouge.compute(predictions=lora_responses, references=ground_truths)['rougeL'])
    if os.path.exists("./models/qlora"):
        print("QLoRA ROUGE:", rouge.compute(predictions=qlora_responses, references=ground_truths)['rougeL'])
    
    # Save markdown comparison
    with open("results_comparison.md", "w") as f:
        f.write("# Model Response Comparison\n\n")
        for i in range(len(test_dataset)):
            f.write(f"## Prompt {i+1}\n**User**: {prompts[i]}\n\n")
            f.write(f"**Ground Truth**: {ground_truths[i]}\n\n")
            f.write(f"**Base Model**: {base_responses[i]}\n\n")
            f.write(f"**LoRA Model**: {lora_responses[i]}\n\n")
            f.write(f"**QLoRA Model**: {qlora_responses[i]}\n\n")
            f.write("---\n")
            
    print("\nEvaluation complete! Results saved to results_comparison.md")

if __name__ == "__main__":
    main()
