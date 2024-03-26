from mytools.tool_utils import FileUtils, TorchUtils

import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "../llama30B_hf",
    lora_weights: str = "",
    prompt_template: str = "alpaca",
    input_data_path: str = "",
    save_data_path: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.15,
    top_p: float = 0.95,
    do_sample: bool = True,
    field: str = "validation",
    batch_size: int = 1,
    enable_bf16: bool = False,
    hf_cache_dir: str = ""

):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter((prompt_template))
    tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=hf_cache_dir)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16 if not enable_bf16 else torch.bfloat16,
            device_map="auto",
            cache_dir=hf_cache_dir
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16 if not enable_bf16 else torch.bfloat16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16 if not enable_bf16 else torch.bfloat16,
            cache_dir=hf_cache_dir
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16 if not enable_bf16 else torch.bfloat16,
            cache_dir=hf_cache_dir
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True, 
            torch_dtype=torch.float16 if not enable_bf16 else torch.bfloat16,
            cache_dir=hf_cache_dir
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights if not enable_bf16 else torch.bfloat16,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    data = FileUtils.load_file(input_data_path)
    instructions = [it["instruction"] for it in data[field]]
    inputs = [it["input"] for it in data[field]]

    results = []
    max_batch_size = batch_size
    for i in range(0, len(instructions), max_batch_size):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        print(f"Processing batch {(i + max_batch_size-1) // max_batch_size} of {(len(instructions) + max_batch_size-1) // max_batch_size}...")
        start_time = time.time()
    
        prompts = [prompter.generate_prompt(instruction, None) for instruction, input in zip(instruction_batch, input_batch)]
        batch_results = evaluate(prompter, prompts, model, tokenizer, max_new_tokens, temperature, top_p, do_sample)
            
        results.extend(batch_results)
        print(f"Finished processing batch {i // max_batch_size + 1}. Time taken: {time.time() - start_time:.2f} seconds")

    for i in range(len(results)):
        data[field][i]['hypothesis'] = results[i]
    
    FileUtils.save_file(data, save_data_path)


def evaluate(prompter, prompts, model, tokenizer, max_new_tokens=256, temperature=0.15, top_p=0.95, do_sample=True):
    batch_outputs = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generation_output = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1,
                                           max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
        
        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        resp = prompter.get_response(output)
        batch_outputs.append(resp)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

