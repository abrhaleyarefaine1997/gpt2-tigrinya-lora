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

This model is a **GPT-2 small** (124M parameters) fine-tuned using **LoRA (Low-Rank Adaptation)** on a custom Tigrinya dataset.  
It is designed for **Tigrinya text generation**, including chatbot/dialogue, storytelling, and text continuation.  

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

- Tigrinya text generation  
- Chatbot / dialogue systems  
- Story or content generation  

### Downstream Use [optional]

- Fine-tuning for domain-specific Tigrinya applications (news, education, cultural storytelling)  

### Out-of-Scope Use

- Generating harmful, offensive, or misleading content  
- Critical decision-making without human supervision  

---

## Bias, Risks, and Limitations

- Dataset may not cover all Tigrinya dialects  
- May generate biased, offensive, or incoherent outputs  
- Not suitable for factual QA tasks  

### Recommendations

Users should verify outputs before real-world use and avoid sensitive applications.  

---

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "abrhaley/gpt2-tigrinya-lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "ኣብ ኣዲስ ኣበባ"
print(generator(prompt, max_length=100, do_sample=True))



## Citation

@misc{abrhaley2025gpt2tigrinya,
  title   = {GPT-2 Tigrinya LoRA Fine-Tuned},
  author  = {Abrhaley},
  year    = {2025},
  url     = {https://huggingface.co/abrhaley/gpt2-tigrinya-lora}
}


