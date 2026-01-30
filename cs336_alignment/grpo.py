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
from cs336_alignment.sft import load_policy_into_vllm_instance,init_vllm,extract_reference_answer,run_vllm
from cs336_alignment.expert_iteration import *
from tqdm import tqdm
import typer

device_train = "cuda:0"
device_vllm = "cuda:1"

n_grpo_steps: int = 200
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 3 # On-policy
train_batch_size: int = 256 # On-policy
gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
gpu_memory_utilization: float = 0.85
eval_steps: int = 10
loss_type: Literal[
"no_baseline",
"reinforce_with_baseline",
"grpo_clip",
] = "grpo_clip"
use_std_normalization: bool = True
cliprange: float = 0.2
SEED = 69
QWEN_MATH_BASE_PATH = 'models/Qwen2.5-Math-1.5B'

PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
TRAIN_DATASET_PATH = "data/gsm8k/train.jsonl"
OUTPUT_PATH = "models/grpo_model"
EVAL_DATASET_PATH = "data/gsm8k/test.jsonl"



sampling_params = SamplingParams(
                    temperature = sampling_temperature, top_p=1.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens, stop=["</answer>"], include_stop_str_in_output=True
                )
ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def mem0(tag=""):
    torch.cuda.synchronize(0)
    print(tag, "alloc0=", torch.cuda.memory_allocated(0)/1e9, "reserved0=", torch.cuda.memory_reserved(0)/1e9)

def sample_qa_pairs(sample_size: int):
    prompts, answers, questions = build_prompts(PROMPT_PATH, TRAIN_DATASET_PATH)
    answers = [extract_reference_answer(answer) for answer in answers]
    idx = random.sample(range(len(prompts)), sample_size)
    return [prompts[i] for i in idx], [answers[i] for i in idx]

def grpo_model(model:torch.nn.Module, input_ids:torch.Tensor, labels:torch.Tensor, response_mask:torch.Tensor, old_log_probs:torch.Tensor, raw_rewards:torch.Tensor, advantages:torch.Tensor, cliprange:float, use_masked_mean: bool = True):
    out = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    policy_log_probs,token_entropy = out["log_probs"], out["token_entropy"]
    if use_masked_mean:
        loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    else:
        loss, metadata = grpo_microbatch_train_step_normalize(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    return loss, token_entropy, metadata

def train(learning_rate: float, use_masked_mean: bool = True, epochs_per_rollout_batch: int = 3, train_batch_size: int = 128):
    print(f"learning_rate: {learning_rate}, masked_mean: {masked_mean}")
    run = wandb.init(project="cs336-grpo",
    name=f"learning_rate_{learning_rate}_math_grpo_masked_mean_{masked_mean}_epochs_per_rollout_batch_{epochs_per_rollout_batch}_train_batch_size_{train_batch_size}",
    config={
        "learning_rate": learning_rate,
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
    device_map=device_train)

    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.0,betas=(0.9, 0.95))
    vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=gpu_memory_utilization)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)
    global_step = 0

    assert train_batch_size % gradient_accumulation_steps == 0, ("train_batch_size must be divisible by gradient_accumulation_steps")
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, ("rollout_batch_size must be divisible by group_size")
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, ("train_batch_size must be greater than or equal to group_size")
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    for i_grpo_steps in range(n_grpo_steps):

        prompts, answers = sample_qa_pairs(n_prompts_per_rollout_batch)
        prompts = [p for p in prompts for _ in range(group_size)]
        answers = [a for a in answers for _ in range(group_size)]

        load_policy_into_vllm_instance(model, vllm)
        responses = run_vllm(vllm, prompts, sampling_params)
        print('[debug] responses: ', len(responses)) #expect to be rollout_batch_size = n_prompts_per_rollout_batch * group_size
        normalized_rewards, rewards, rewards_metadata = compute_group_normalized_rewards(r1_zero_reward_fn, responses, answers, group_size, advantage_eps, use_std_normalization)
        normalized_rewards = normalized_rewards.unsqueeze(-1)
        raw_rewards = rewards.unsqueeze(-1)

        input_ids, labels, response_mask = get_train_data(prompts, responses, tokenizer)
        print ("---------examples of prompt, response, answer-----------")
        for i in range(3):
            print (f"prompt:{prompts[i]}")
            print (f"rollouts:{responses[i]}")
            print (f"answers:{answers[i]}")
            print (f"reward:{raw_rewards[i].item():.6f}")
        print ("--------grpo step rollout example done")

        for i_epochs_per_rollout_batch in range(epochs_per_rollout_batch):
            chunk = 6
            with torch.no_grad():
                lp_chunks = []
                for s in range(0, input_ids.size(0), chunk):
                    out = get_response_log_probs(
                        model,
                        input_ids[s:s+chunk],
                        labels[s:s+chunk],
                        return_token_entropy=False,
                    )
                    lp_chunks.append(out["log_probs"])
                old_log_probs = torch.cat(lp_chunks, dim=0)

            loss_list = []
            entropy_list = []
            rewards_list = []

            for i_microbatches_per_rollout_batch in tqdm(range(n_microbatches_per_rollout_batch), desc="Training"):
                start = i_microbatches_per_rollout_batch * micro_train_batch_size
                end = start + micro_train_batch_size
                input_ids_batch, labels_batch, response_mask_batch = (x[start:end] for x in (input_ids, labels, response_mask))
                raw_rewards_batch = rewards[start:end]
                advantages_batch = normalized_rewards[start:end]
                old_log_probs_batch = old_log_probs[start:end]
                with amp_ctx:
                    loss, entropy, metadata = grpo_model(model, input_ids_batch, labels_batch, response_mask_batch, old_log_probs_batch, raw_rewards_batch, advantages_batch, cliprange)
                    loss_list.append(loss.item())
                    entropy_list.append(entropy.mean().item())
                    rewards_list.append(raw_rewards_batch.mean().item())
                    
                    
                    if (i_microbatches_per_rollout_batch+1) % gradient_accumulation_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if loss_type == "grpo_clip":
                            clip_fraction = (metadata["is_clipped"] & response_mask_batch).sum() / response_mask_batch.sum()
                            clip_fraction = clip_fraction.item()
                        optimizer.step()
                        optimizer.zero_grad()

                        # 只使用最近 gradient_accumulation_steps 个 microbatches 的数据
                        recent_loss_list = loss_list[-gradient_accumulation_steps:]
                        recent_entropy_list = entropy_list[-gradient_accumulation_steps:]
                        recent_rewards_list = rewards_list[-gradient_accumulation_steps:]
                        
                        loss = sum(recent_loss_list) / len(recent_loss_list)
                        entropy = sum(recent_entropy_list) / len(recent_entropy_list)
                        rewards_mean = sum(recent_rewards_list) / len(recent_rewards_list)
                        print (f"Training summary at step {global_step + 1}:")
                        print (f"loss: {loss:.6f}")
                        print (f"Global Entropy: {entropy:.6f}")
                        print (f"Grad Norm: {grad_norm.item():.6f}")
                        print (f"Rewards Mean: {rewards_mean:.6f}")
                        log_dict = {
                            "train/loss": loss,
                            "train/entropy": entropy,
                            "train/grad_norm": grad_norm.item(),
                            "train_step": global_step + 1,
                            "train/rewards_mean": rewards_mean,
                        }
                        if loss_type == "grpo_clip":
                            log_dict["train/clip_fraction"] = clip_fraction
                        wandb.log(log_dict)
            
                        global_step += 1
                        
                        # 清空列表，为下一个梯度累积步骤做准备
                        loss_list.clear()
                        entropy_list.clear()
                        rewards_list.clear()        

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
                                "eval/fully wrong": overview["fully_wrong"],
                                "eval/accuracy": overview["fully_correct"] / overview["total"],
                                "eval_step": global_step
                            })
                                        

    run.finish()

if __name__ == "__main__":
    typer.run(train)