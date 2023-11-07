# https://huggingface.co/docs/transformers/generation_strategies
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')

print(model.generation_config)
text="Am I cool?" 
encoded = tokenizer(text, return_tensors='pt').to('cuda')
print(encoded)

outputs = model.generate(**encoded)
print(outputs)
print(tokenizer.decode(outputs[0]))