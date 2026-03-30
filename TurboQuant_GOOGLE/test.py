import torch
from transformers import AutoTokenizer
from my_qwen3 import Qwen3ForCausalLM, Qwen3Config

local_model_path = r"C:\Users\lihaodong\.cache\modelscope\hub\models\Qwen\Qwen3-1___7B"

# 加载配置
config = Qwen3Config.from_pretrained(local_model_path)
config.use_turboquant = True   # 启用
config.turboquant_bits = 3     # 设置比特数

# 加载模型
model = Qwen3ForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# 加载 tokenizer（必须支持 apply_chat_template）
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

# 准备输入
prompt = "你是谁"
messages = [{"role": "user", "content": prompt}]

# 使用聊天模板，启用思考模式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Qwen3 tokenizer 支持此参数
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解析思考内容
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # 151668 是 </think> 的 token id
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)