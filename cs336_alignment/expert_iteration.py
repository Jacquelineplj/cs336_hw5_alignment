from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
import wandb
import torch
from cs336_alignment.utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from argparse import ArgumentParser
import re
from typing import Callable, List
from unittest.mock import patch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm,build_prompts,calculate_metrics
from cs336_alignment.sft import load_policy_into_vllm_instance,init_vllm,extract_reference_answer,sft_model


DATASET_PATH = "data/gsm8k/train.jsonl"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
QWEN_MATH_BASE_PATH = 'models/Qwen2.5-Math-1.5B'
TRAIN_DATASET_PATH = "data/gsm8k/train.jsonl"
EVAL_DATASET_PATH = "data/gsm8k/test.jsonl"
OUTPUT_PATH = "models/expert_iteration_model"
n_ei_steps = 5
micro_batch_size = 4
n_grad_accum_steps = 64
n_sft_steps = 128
eval_steps = 5
device_train = "cuda:0"
device_vllm = "cuda:1"
SEED = 69
ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
sampling_params = SamplingParams(
                    temperature = 1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
                )

def generate_data(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams):

    info_dict_list = evaluate_vllm(vllm_model, reward_fn, prompts, answers, eval_sampling_params)
    fully_correct = [result for result in info_dict_list if result["reward"] == 1.0]
    prompt = [result["prompt"] for result in fully_correct]
    response = [result["response"] for result in fully_correct]
    return prompt, response

def get_train_data(prompt:List[str], response:List[str], tokenizer:PreTrainedTokenizerBase) -> List[str]:
    tokenized_train_data = tokenize_prompt_and_output(prompt_strs=prompt,output_strs=response,tokenizer=tokenizer)
    input_ids = tokenized_train_data["input_ids"].to(device_train)
    labels = tokenized_train_data["labels"].to(device_train)
    response_mask = tokenized_train_data["response_mask"].to(device_train)
    return input_ids, labels, response_mask

def vllm_loop(model:torch.nn.Module, vllm:LLM, tokenizer:PreTrainedTokenizerBase, dataset_path:str):
    load_policy_into_vllm_instance(model, vllm)
    prompt, answer, _ = build_prompts(PROMPT_PATH, dataset_path)
    prompt, response = generate_data(vllm, r1_zero_reward_fn, prompt, answer, sampling_params)
    input_ids, labels, response_mask = get_train_data(prompt, response, tokenizer)
    return input_ids, labels, response_mask

def get_batch(input_ids:torch.Tensor, labels:torch.Tensor, response_mask:torch.Tensor):
    batch_indices = random.sample(range(len(input_ids)), micro_batch_size)
    input_ids = input_ids[batch_indices]
    labels = labels[batch_indices]
    response_mask = response_mask[batch_indices]
    return input_ids, labels, response_mask


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    responses = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for response, prompt, answer in zip(responses, prompts, answers):
        extracted_answer = extract_reference_answer(answer)
        reward_dict = reward_fn(response.outputs[0].text, extracted_answer)
        reward_dict["prompt"] = prompt
        reward_dict["answer"] = answer
        reward_dict["extracted_answer"] = extracted_answer
        reward_dict["response"] = response.outputs[0].text
        results.append(reward_dict)
    return results

def main():
    wandb.init(project="cs336-sft",
        name=f"expert_iteration_batch_size_{micro_batch_size*n_grad_accum_steps}",
        config={
            'batch_size': micro_batch_size*n_grad_accum_steps
            }
        )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_train
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)
    global_step = 0
    for i_ei_step in range(n_ei_steps):
        input_ids, labels, response_mask = vllm_loop(model, vllm, tokenizer, TRAIN_DATASET_PATH)
        # input_ids, labels, response_mask = vllm_loop(model, vllm, tokenizer, EVAL_DATASET_PATH)

        for i_sft_step in range(n_sft_steps):
            loss_list = []
            entropy_list = []
            response_tokens_list = []
            mask_fraction_list = []

            
            for j_grad_accum_step in range(n_grad_accum_steps):
                input_ids_batch, labels_batch, response_mask_batch = get_batch(input_ids, labels, response_mask)
                with amp_ctx:
                    loss, entropy, metadata = sft_model(model, input_ids_batch, labels_batch, response_mask_batch, n_grad_accum_steps)
                    loss_list.append(loss.item())
                    entropy_list.append(entropy.mean().item())
                    response_tokens_list.append(metadata["response_tokens"])
                    mask_fraction_list.append(metadata["mask_fraction"])
                    

                    if j_grad_accum_step == n_grad_accum_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                        entropy = sum(entropy_list) / n_grad_accum_steps
                        response_tokens = sum(response_tokens_list) / n_grad_accum_steps
                        mask_fraction = sum(mask_fraction_list) / n_grad_accum_steps
                        print (f"Training summary at step {global_step + 1}:")
                        print (f"loss: {loss:.6f}")
                        print (f"Global Entropy: {entropy:.6f}")
                        print (f"response_tokens: {response_tokens}")
                        print (f"mask_fraction: {mask_fraction}")
                        wandb.log({
                            "train/loss": loss,
                            "train/entropy": entropy,
                            "train/response_tokens": response_tokens,
                            "train/mask_fraction": mask_fraction,
                            "train_step": global_step + 1
                        })
            
            global_step += 1
            if (global_step % eval_steps == 0):
                load_policy_into_vllm_instance(model, vllm)

                test_prompt, test_answer, _ = build_prompts(PROMPT_PATH, EVAL_DATASET_PATH)
                overview = evaluate_vllm(vllm, r1_zero_reward_fn, test_prompt, test_answer, sampling_params)
                overview = calculate_metrics(overview)

                print (f"evaluation at step {global_step}:")
                print (f"total number: {overview["total"]}")
                print (f"Accurancy: {overview["fully_correct"] / overview["total"] * 100:.2f}%")
                print (f"Wrong answer with correct format:{overview["only_format_correct"]}")
                print (f"Wrong format:{overview["fully_wrong"]}")

                wandb.log({
                    "eval/correct": overview["fully_correct"],  
                    "eval/wrong format": overview["only_format_correct"],
                    "eval/wrong format": overview["fully_wrong"],
                    "eval/accuracy": overview["fully_correct"] / overview["total"],
                    "eval_step": global_step+1
                })

                model.save_pretrained(save_directory=OUTPUT_PATH)
                tokenizer.save_pretrained(save_directory=OUTPUT_PATH)
if __name__ == "__main__":
    main()