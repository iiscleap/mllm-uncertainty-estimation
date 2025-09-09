import argparse

def compute_negation_accuracy(vanilla_file, negated_file, answer_file):
    # load gold answers
    gold_answers = {}
    with open(answer_file, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            gold_answers[idx] = ans.strip("()")

    # load vanilla predictions
    vanilla_preds = {}
    with open(vanilla_file, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            vanilla_preds[idx] = ans[0]

    # load negated predictions
    negated_preds = {}
    with open(negated_file, "r") as f:
        for line in f:
            idx, ans = line.strip().split(" ", maxsplit=1)
            negated_preds[idx] = ans[0]

    total = 0
    vanilla_correct = 0
    negation_correct = 0

    for idx in gold_answers.keys():
        gold = gold_answers[idx]
        # flip the gold answer for the negated version
        gold_negated = "B" if gold == "A" else "A"

        vanilla_pred = vanilla_preds.get(idx, "X")
        negated_pred = negated_preds.get(idx, "X")

        if vanilla_pred == gold:
            vanilla_correct += 1
        if negated_pred == gold_negated:
            negation_correct += 1
        total += 1

    print(total)
    vanilla_acc = vanilla_correct / total
    negation_acc = negation_correct / max(vanilla_correct, 1)
    return vanilla_acc, negation_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset

    if dataset == "blink":
        answer_file = f"answer_list.txt"
    else:
        answer_file = f"dataset/VSR/answer_list.txt"

    vanilla_file = f"vanilla_output/{model}_vanilla_{dataset}.txt"
    negated_file = f"vanilla_output/negated_{model}_vanilla_{dataset}.txt"

    vanilla_acc, negation_acc = compute_negation_accuracy(vanilla_file, negated_file, answer_file)

    print(f"{model}-{dataset} vanilla accuracy: {vanilla_acc:.4f}")
    print(f"{model}-{dataset} negation accuracy: {negation_acc:.4f}")
