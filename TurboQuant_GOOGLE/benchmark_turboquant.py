import gc
import time

import torch
from transformers import AutoTokenizer

from my_qwen3 import Qwen3Config, Qwen3ForCausalLM


LOCAL_MODEL_PATH = r"C:\Users\lihaodong\.cache\modelscope\hub\models\Qwen\Qwen3-1___7B"
PROMPT_TOKEN_LIST = [2048, 4096, 8192]
DECODE_TOKENS = 8


def build_long_prompt(tokenizer, min_tokens):
    seed = "请详细分析人工智能系统在部署、量化、缓存和长上下文推理中的关键工程权衡。"
    pieces = []
    token_count = 0
    while token_count < min_tokens:
        pieces.append(seed)
        text = "".join(pieces)
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
    return text, token_count


def load_model(config_override):
    config = Qwen3Config.from_pretrained(LOCAL_MODEL_PATH)
    for key, value in config_override.items():
        setattr(config, key, value)
    return Qwen3ForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )


def prefill_and_decode(tokenizer, prompt, description, config_override, max_new_tokens=DECODE_TOKENS):
    model = load_model(config_override)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**2

    start_time = time.time()
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    prefill_time = time.time() - start_time
    prefill_mem = torch.cuda.memory_allocated() / 1024**2
    prefill_peak = torch.cuda.max_memory_allocated() / 1024**2

    past_key_values = prefill_outputs.past_key_values
    next_token = prefill_outputs.logits[:, -1:, :].argmax(dim=-1)
    decode_start = time.time()
    decode_peak = prefill_peak
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=torch.ones(
                    (input_ids.shape[0], past_key_values.get_seq_length() + 1),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        decode_peak = max(decode_peak, torch.cuda.max_memory_allocated() / 1024**2)
    decode_time = time.time() - decode_start

    result = {
        "description": description,
        "prompt_tokens": int(attention_mask.sum().item()),
        "generated_tokens": max_new_tokens,
        "prefill_time_sec": prefill_time,
        "gen_time_sec": decode_time,
        "start_mem_mb": start_mem,
        "prefill_mem_mb": prefill_mem,
        "prefill_peak_mb": prefill_peak,
        "end_mem_mb": torch.cuda.memory_allocated() / 1024**2,
        "peak_mem_mb": decode_peak,
    }

    del outputs
    del prefill_outputs
    del past_key_values
    del input_ids
    del attention_mask
    del inputs
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    for prompt_tokens in PROMPT_TOKEN_LIST:
        prompt, estimated_tokens = build_long_prompt(tokenizer, min_tokens=prompt_tokens)
        print(f"\nPrepared prompt tokens: {estimated_tokens}")

        baseline = prefill_and_decode(
            tokenizer,
            prompt,
            "Baseline fp16 KV",
            {"use_turboquant": False},
        )
        turbo = prefill_and_decode(
            tokenizer,
            prompt,
            "TurboQuant 3-bit KV",
            {"use_turboquant": True, "turboquant_bits": 3, "turboquant_block_size": 256},
        )

        print("=" * 88)
        print(
            f"{'Config':<22} {'PromptTok':<10} {'GenTok':<8} {'Prefill(s)':<10} {'Gen(s)':<10} "
            f"{'PrefillMem':<12} {'PeakMem':<12} {'EndMem':<12}"
        )
        print("-" * 88)
        for row in (baseline, turbo):
            print(
                f"{row['description']:<22} {row['prompt_tokens']:<10} {row['generated_tokens']:<8} "
                f"{row['prefill_time_sec']:<10.2f} {row['gen_time_sec']:<10.2f} "
                f"{row['prefill_mem_mb']:<12.2f} {row['peak_mem_mb']:<12.2f} {row['end_mem_mb']:<12.2f}"
            )
        print("=" * 88)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    main()


'''
========================================================================================
Config                 PromptTok  GenTok   Prefill(s) Gen(s)     PrefillMem   PeakMem      EndMem      
----------------------------------------------------------------------------------------
Baseline fp16 KV       2082       8        0.35       0.22       4122.63      4245.37      4123.79     
TurboQuant 3-bit KV    2082       8        0.95       5.24       3955.12      3980.33      3955.57     
========================================================================================

========================================================================================
Config                 PromptTok  GenTok   Prefill(s) Gen(s)     PrefillMem   PeakMem      EndMem      
----------------------------------------------------------------------------------------
Baseline fp16 KV       4129       8        0.69       0.83       4938.99      6376.65      4940.15     
TurboQuant 3-bit KV    4129       8        2.22       8.00       4595.55      4620.76      4595.97     
========================================================================================

========================================================================================
Config                 PromptTok  GenTok   Prefill(s) Gen(s)     PrefillMem   PeakMem      EndMem      
----------------------------------------------------------------------------------------
Baseline fp16 KV       8223       8        45.44      61.03      6573.25      14190.60     6575.22     
TurboQuant 3-bit KV    8223       8        7.13       11.64      5873.59      5905.71      5878.44     
========================================================================================

'''