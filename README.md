---
base_model: gpt2
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:gpt2
  - lora
  - transformers
datasets:
  - custom
metrics:
  - perplexity
language:
  - ti
new_version: 1.0
---

# GPT-2 Tigrinya (LoRA Fine-Tuned)

## Model Details

### Model Description

This is a **GPT-2 small** (124M parameters) model fine-tuned using **LoRA (Low-Rank Adaptation)** on a custom Tigrinya dataset.  
It is intended for **Tigrinya text generation**, including chatbots, storytelling, and general text continuation.  

- **Developed by:** Abrhaley (Warsaw University of Technology, MSc student)  
- **Funded by [optional]:** N/A  
- **Shared by [optional]:** Abrhaley  
- **Model type:** Causal Language Model (decoder-only Transformer)  
- **Language(s) (NLP):** Tigrinya (`ti`)  
- **License:** MIT  
- **Finetuned from model [optional]:** [gpt2](https://huggingface.co/gpt2)  
- **Framework versions:** Transformers + PEFT 0.17.1  

### Model Sources

- **Repository:** [GitHub](https://github.com/abrhaleyarefaine1997/gpt2-tigrinya-lora)  
- **Paper [optional]:** N/A  
- **Demo [optional]:** N/A  

---

## Uses

### Direct Use

- Generate Tigrinya text  
- Chatbot or dialogue systems  
- Storytelling and content creation  

### Downstream Use [optional]

- Fine-tuning for domain-specific Tigrinya applications (news, education, cultural storytelling)  

### Out-of-Scope Use

- Generating harmful, offensive, or misleading content  
- Decision-making in critical scenarios without human supervision  

---

## Bias, Risks, and Limitations

- Dataset may not include all Tigrinya dialects  
- Possible generation of biased or incoherent text  
- Not suitable for factual question-answering tasks  

### Recommendations

Verify outputs before using in real-world scenarios and avoid sensitive applications.  

---

## How to Get Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "abrhaley/gpt2-tigrinya-lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "ኣብ ኣዲስ ኣበባ"
outputs = generator(prompt, max_length=100, do_sample=True, temperature=0.8, top_k=50, top_p=0.95, num_return_sequences=3)

for i, out in enumerate(outputs, 1):
    print(f"\n=== Sample {i} ===\n{out['generated_text']}\n")

###Citation

If you use this model in your work, please cite it as:

```bibtex
@misc{abrhaley2025gpt2tigrinya,
  title   = {GPT-2 Tigrinya LoRA Fine-Tuned},
  author  = {Abrhaley},
  year    = {2025},
  url     = {https://huggingface.co/abrhaley/gpt2-tigrinya-lora}
}

### APA format:

Abrhaley. (2025). GPT-2 Tigrinya LoRA Fine-Tuned. https://github.com/abrhaleyarefaine1997/gpt2-tigrinya-lora
