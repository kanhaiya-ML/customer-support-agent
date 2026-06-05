from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

model_path = r"C:\Users\Kanhaiya\OneDrive\Desktop\local_projects\chatbot\support_agent\t5_customer_support_optimized"

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()
print("Model loaded ✅")

def predict(text):
    input_text = "customer support: " + text
    
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=64,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    
    # return f"Customer: {text}"
    return f"{response}\n"