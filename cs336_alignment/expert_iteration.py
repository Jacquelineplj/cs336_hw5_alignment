from cs336.math_baseline import evaluate_vllm,build_prompts,calculate_metrics
from cs336.utils import tokenize_prompt_and_output
from transformers import PreTrainedTokenizerBase
from cs336.sft import load_policy_into_vllm_instance,init_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

DATASET_PATH = "data/gsm8k/train.jsonl"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
QWEN_MATH_BASE_PATH = 'models/Qwen2.5-Math-1.5B'
TRAIN_DATASET_PATH = "data/gsm8k/train.jsonl"
EVAL_DATASET_PATH = "data/gsm8k/test.jsonl"
n_ei_steps = 5
micro_batch_size = 8
n_grad_accum_steps = 8
device_train = "cuda:0"
device_vllm = "cuda:1"
SEED = 69

def generate_data(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams):

    info_dict_list = evaluate_vllm(vllm_model, reward_fn, prompts, answers, eval_sampling_params)
    fully_correct = [result for result in results if result["reward"] == 1.0]
    prompt = [result["prompt"] for result in fully_correct]
    response = [result["response"] for result in fully_correct]
    return prompt, response

def get_train_data(prompt:List[str], response:List[str], tokenizer:PreTrainedTokenizerBase) -> List[str]:
    tokenized_train_data = tokenize_prompt_and_output(prompt_strs=prompt,output_strs=response,tokenizer=tokenizer)
    input_ids = train_batch["input_ids"].to(device_train)
    labels = train_batch["labels"].to(device_train)
    response_mask = train_batch["response_mask"].to(device_train)
    return input_ids, labels, response_mask

def vllm_loop(model:torch.nn.Module, vllm:LLM, tokenizer:PreTrainedTokenizerBase, dataset_path:str):
    load_policy_into_vllm_instance(model, vllm)
    sampling_params = SamplingParams(
                    temperature = 1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
                )
    prompt, answer = build_prompts(PROMPT_PATH, dataset_path)
    prompt, response = generate_data(vllm, r1_zero_reward_fn, prompt, answer, sampling_params)
    input_ids, labels, response_mask = get_train_data(prompt, response, tokenizer)
    return input_ids, labels, response_mask

def get_batch(input_ids:torch.Tensor, labels:torch.Tensor, response_mask:torch.Tensor):
    batch_indices = random.sample(range(len(input_ids)), micro_batch_size)
    input_ids = input_ids[batch_indices]
    labels = labels[batch_indices]
    response_mask = response_mask[batch_indices]
    return input_ids, labels, response_mask

def main():
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_train
    )
    vllm = init_vllm(QWEN_MATH_BASE_PATH, device_vllm, SEED, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    amp_ctx = torch.amp.autocast(device_type=device_train, dtype=torch.bfloat16)
    global_step = 0
    for i_ei_step in range(n_ei_steps):
        input_ids, labels, response_mask = vllm_loop(model, vllm, tokenizer, TRAIN_DATASET_PATH)
        for i_sft_step in range(n_sft_steps):
            loss_list = []
            entropy_list = []
            response_tokens_list = []
            mask_fraction_list = []

            input_ids_batch, labels_batch, response_mask_batch = get_batch(input_ids, labels, response_mask)
            for j_grad_accum_step in range(n_grad_accum_steps):
                with amp_ctx:
                    loss, entropy, metadata = sft_model(model, input_ids_batch, labels_batch, response_mask_batch, n_grad_accum_steps)
                    loss_list.append(loss)
                    entropy_list.append(entropy.mean().item())
                    response_tokens_list.append(metadata["response_tokens"])
                    mask_fraction_list.append(metadata["mask_fraction"])
                    
                    if j_grad_accum_step == n_grad_accum_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        print (f"Training summary at step {global_step + 1}:")
                        print (f"loss: {loss:.6f}")
                        print (f"Global Entropy: {entropy.mean().item():.6f}")
                        print (f"response_tokens: {metadata["response_tokens"]}")
                        print (f"mask_fraction: {metadata["mask_fraction"]}")
            
            global_step += 1
            if (global_step % eval_steps == 0):
                load_policy_into_vllm_instance(model, vllm)

                test_prompt, test_answer = build_prompts(PROMPT_PATH, EVAL_DATASET_PATH)
                overview = evaluate_vllm(vllm, r1_zero_reward_fn, test_prompt, test_answer, sampling_params)
                overview = calculate_metrics(overview)

                print (f"evaluation at step {global_step}:")
                print (f"total number: {overview["total"]}")
                print (f"Accurancy: {overview["fully_correct"] / overview["total"] * 100:.2f}%")
                print (f"Wrong answer with correct format:{overview["only_format_correct"]}")
                print (f"Wrong format:{overview["fully_wrong"]}")

                wandb.log({
                    "eval/correct": overview["fully_correct"],  
                    "eval/correct format with wrong answer": overview["only_format_correct"],
                    "eval/wrong format": overview["fully_wrong"],
                    "eval/accuracy": overview["fully_correct"] / overview["total"],
                    "eval_step": global_step+1
                }

                model.save_pretrained(save_directory=OUTPUT_PATH)
                tokenizer.save_pretrained(save_directory=OUTPUT_PATH)