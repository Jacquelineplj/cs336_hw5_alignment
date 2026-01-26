import json

def main():
    correct = []
    format_correct_answer_wrong = []
    format_wrong_answer_correct = []
    format_wrong_answer_wrong = []
    with open("baseline_result.jsonl") as f:
        for line in f:
            line = json.loads(line.strip())
            if line["format_reward"] == 1 and line["answer_reward"] == 1:
                correct.append(line)
            elif line["format_reward"] == 0 and line["answer_reward"] == 1:
                format_wrong_answer_correct.append(line)
            elif line["format_reward"] == 1 and line["answer_reward"] == 0:
                format_correct_answer_wrong.append(line)
            else:
                format_wrong_answer_wrong.append(line)
    correct_dict_list = []
    for i in correct:
        correct_dict = {}
        correct_dict["question"] = i["question"]
        correct_dict["answer"] = i["answer"]
        correct_dict_list.append(correct_dict)

    with open("./data/gsm8k/train_correct.jsonl", "w") as f:
        for i in correct_dict_list:
            json.dump(i, f)
            f.write("\n")