from vllm import LLM, SamplingParams
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
import os
import re
ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def create_vllm_model(model_path: str) -> LLM:
    llm = LLM(model = model_path)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True
    return llm, sampling_params

def build_prompts(prompts_path: str, data_path: str) -> List[str]:
    with open(prompts_path, "r") as f:
        prompt_template = f.read()
    prompts = []
    answers = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(prompt_template.format(question=data["question"]))
            answers.append(data["answer"])
    return prompts, answers


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams) -> None:
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

def calculate_metrics(results: List[dict], debug: bool = False) -> dict:
    total = len(results)
    fully_correct = [result for result in results if result["reward"] == 1.0]
    only_format_correct = [result for result in results if result["format_reward"] == 1.0 and result["answer_reward"] == 0.0]
    fully_wrong = [result for result in results if result["format_reward"] == 0.0 and result["answer_reward"] == 0.0]
    if debug:
        print('Only format correct')
        for i in range(min(10,len(only_format_correct))):
            print(only_format_correct[i]["prompt"])
            print(only_format_correct[i]["response"])
            print(only_format_correct[i]["answer"])
            print('-'*100)
        print('Fully wrong')
        for i in range(min(10,len(fully_wrong))):
            print(fully_wrong[i]["prompt"])
            print(fully_wrong[i]["response"])
            print(fully_wrong[i]["answer"])
            print('-'*100)
    return {
        "total": total,
        "fully_correct": len(fully_correct),
        "only_format_correct": len(only_format_correct),
        "fully_wrong": len(fully_wrong),
    }

def main():
    model_path = "models/Qwen2.5-Math-1.5B"
    llm, sampling_params = create_vllm_model(model_path)
    prompts, answers = build_prompts("cs336_alignment/prompts/r1_zero.prompt", "data/gsm8k/test.jsonl")
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
    os.makedirs("results", exist_ok=True)
    with open("results/math_baseline_results.jsonl", "w") as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    print(f"Saved {len(results)} results to results/math_baseline_results.jsonl")
    metrics = calculate_metrics(results, debug=True)
    print(metrics)

if __name__ == "__main__":
    main()