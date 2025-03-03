from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('/home/cvgroup/myz/zmx/modelDemo/models/gpt2')
model = GPT2Model.from_pretrained('/home/cvgroup/myz/zmx/modelDemo/models/gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
