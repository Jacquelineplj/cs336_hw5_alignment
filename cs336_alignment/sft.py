from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
import wandb
import torch
from cs336_alignment.utils import tokenize_prompt_and_output
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from argparse import ArgumentParser


QWEN_MATH_BASE_PATH = 'models/Qwen2.5-Math-1.5B'
TRAIN_DATASET_PATH = "data/gsm8k/train.jsonl"
EVAL_DATASET_PATH = "data/gsm8k/test.jsonl"
PROMPT_PATH = 'cs336_alignment/prompts/r1_zero.prompt'
OUTPUT_PATH = "models/sft_model"
SEED = 69
n_sft_steps = 256
n_grad_accum_steps = 8
eval_steps = 16
device_train = "cuda:0"
device_vllm = "cuda:1"
micro_batch_size = 8

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )
def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm:LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_jsonl(file_path:str)->list[str]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_qa(data:list[str], prompt_path:str)->list[str]:
    formated_q = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for d in data:
        extracted_answer = extract_reference_answer(d["answer"])
        pair = {}
        pair["prompt"] = prompt.format(question = d["question"])
        pair["answer"] = d["answer"] + "</think> <answer>" + extracted_answer+ "</answer>"
        formated_q.append(pair)
    return formated_q

def perpare_dataset(train_sample: int):
    train_data = load_jsonl(TRAIN_DATASET_PATH)
    train_data = train_data[:train_sample]
    train_data = format_qa(train_data, PROMPT_PATH)

    test_data = load_jsonl(EVAL_DATASET_PATH)
    test_data = format_qa_prompt(test_data, PROMPT_PATH)
    return train_data, test_data


def format_qa_prompt(data:list[str], prompt_path:str)->list[str]:
    formated_q = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for d in data:
        pair = {}
        pair["prompt"] = prompt.format(question = d["question"])
        pair["answer"] = d["answer"]
        formated_q.append(pair)
    return formated_q

def get_batch(tokenized_train_data: dict[str, torch.Tensor], batch_size: int, device: str):
    batch_indices = random.sample(range(len(tokenized_train_data["input_ids"])), batch_size)
    return {k: v[batch_indices].to(device) for k,v in tokenized_train_data.items()}


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
):
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        extracted_answer = extract_reference_answer(answer)
        reward_dict = reward_fn(response, extracted_answer)
        allinfo_dict_list.append(reward_dict)
    overview = {"correct":0, "format_wrong":0, "answer_wrong":0, "count":0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview

def run_vllm(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    texts = [output.outputs[0].text.strip() for output in result]
    return texts

def sft_model(model:torch.nn.Module, input_ids:torch.Tensor, labels:torch.Tensor, response_mask:torch.Tensor, n_grad_accum_steps:int):
    response_log_probs = get_response_log_probs(model, input_ids, labels, True)
    log_probs = response_log_probs["log_probs"]
    entropy = response_log_probs["token_entropy"]
    loss, metadata = sft_microbatch_train_step(log_probs, response_mask, n_grad_accum_steps)
    return loss, entropy, metadata


def train(train_samples: int, dataset_type: str):
        wandb.init(project="cs336-sft",
        name=f"train_sample_{train_sample}_dataset_{dataset_type}_math_sft",
        config={
            "train_sample": train_sample,
            "dataset_type": dataset_type
            }
        )

        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

        tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)

        #prepare dataset
        train_qa, test_prompt = perpare_dataset(train_sample)
        tokenized_train_data = tokenize_prompt_and_output(prompt_strs=[data["prompt"] for data in train_qa],
                                                    output_strs=[data["response"] for data in train_qa],
                                                    tokenizer=tokenizer)
        train_batch = get_batch(tokenized_train_data, micro_batch_size, device_train)
        input_ids = train_batch["input_ids"].to(device_train)
        labels = train_batch["labels"].to(device_train)
        response_mask = train_batch["response_mask"].to(device_train)

        #prepare model
        model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_train
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)
        vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)
        for i_sft_step in range(n_sft_steps):
            loss_list = []
            entropy_list = []
            response_tokens_list = []
            mask_fraction_list = []
            for j_grad_accum_step in range(n_grad_accum_steps):
                with amp_ctx:
                    loss, entropy, metadata = sft_model(model, input_ids, labels, response_mask, n_grad_accum_steps)
                    loss_list.append(loss)
                    entropy_list.append(entropy.mean().item())
                    metadata_list.append(metadata)
                    response_tokens_list.append(metadata["response_tokens"])
                    mask_fraction_list.append(metadata["mask_fraction"])
                    if j_grad_accum_step == n_grad_accum_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        loss = sum(loss_list) / n_grad_accum_steps
                        entropy = sum(entropy_list) / n_grad_accum_steps
                        response_tokens = sum(response_tokens_list) / n_grad_accum_steps
                        mask_fraction = sum(mask_fraction_list) / n_grad_accum_steps
                        print (f"Training summary at step {i_sft_step + 1}:")
                        print (f"loss: {loss:.6f}")
                        print (f"Global Entropy: {entropy:.6f}")
                        print (f"response_tokens: {response_tokens}")
                        print (f"mask_fraction: {mask_fraction}")
                        wandb.log({
                            "train/loss": to_float(loss),
                            "train/entropy": to_float(entropy.mean()),
                            "train/response_tokens": response_tokens,
                            "train/mask_fraction": mask_fraction,
                            "train_step": i_sft_step + 1
                        })
            if i_sft_step % eval_steps == 0:
                load_policy_into_vllm_instance(model, vllm)
                sampling_params = SamplingParams(
                    temperature = 1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
                )
                overview = evaluate_vllm(
                    vllm_model=vllm,
                    reward_fn=r1_zero_reward_fn,
                    prompts = [data["prompt"] for data in test_prompt],
                    answers = [data["answer"] for data in test_prompt],
                    eval_sampling_params = sampling_params
                )
                accuracy = overview["correct"] / overview["count"]
                print (f"evaluation at step {i_sft_step+1}")
                print (f"Correct answer:{overview['correct']}")
                print (f"Accuracy: {accuracy:.4f}")
                print (f"Wrong answer with correct format:{overview['answer_wrong']}")
                print (f"Wrong format:{overview['format_wrong']}")

                wandb.log({
                    "eval/correct": overview["correct"],
                    "eval/wrong answer": overview["answer_wrong"],
                    "eval/wrong format": overview["format_wrong"],
                    "eval/accuracy": accuracy,
                    "eval_step": i_sft_step + 1
                })
                model.save_pretrained(save_directory=OUTPUT_PATH)
                tokenizer.save_pretrained(save_directory=OUTPUT_PATH)





    
if __name__ == "__main__":
    dataset_type = "raw"
    train_samples = [128, 256, 512, 1024]
    for train_sample in train_samples:
        main(train_sample, dataset_type)
