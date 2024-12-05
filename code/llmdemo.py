
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的GPT-2模型和tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 输入文本（我们将基于这个文本生成后续内容）
input_text = "Once upon a time"

# 使用tokenizer将文本编码为模型的输入格式
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 使用模型生成后续文本
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的文本
print(generated_text)
