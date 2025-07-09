import json
import os
from pathlib import Path
from collections import defaultdict


def _calculate_metrics(tp, fp, fn):
    """a helper function to calculate precision, recall and F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_f1_scores_multi_class(jsonl_file_path, dir_name, hd_dir, model_name=None, choices=["A", "B", "C", "D"]):
    stats = {choice: defaultdict(int) for choice in choices}

    error_ids = []
    failed_ids = []

    if model_name and dir_name:
        right_path = os.path.join(hd_dir, f"{model_name}", f"{dir_name}", "right")
        false_path = os.path.join(hd_dir, f"{model_name}", f"{dir_name}", "false")
        os.makedirs(right_path, exist_ok=True)
        os.makedirs(false_path, exist_ok=True)

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                if data.get("status") != "success":
                    failed_ids.append(line_num)
                    continue
                image_id = data.get("image", f"line_{line_num}")
                true_choice = data.get("choice")
                pred_ans_str = data.get("pred_ans")
                if not all([true_choice, pred_ans_str]) or true_choice not in choices:
                    error_ids.append(f"{image_id} (missing required fields or invalid true_choice)")
                    continue
                pred_choice_letter = None
                stripped_pred = pred_ans_str.strip()
                if ":" in stripped_pred:
                    pred_choice_letter = stripped_pred.split(":", 1)[0].strip()
                elif "." in stripped_pred and stripped_pred.find(".") < 3:
                    pred_choice_letter = stripped_pred.split(".", 1)[0].strip()
                elif ")" in stripped_pred and stripped_pred.find(")") < 3:
                    pred_choice_letter = stripped_pred.split(")", 1)[0].strip()
                else:
                    first_char = stripped_pred[:1]
                    if first_char.isalpha() and first_char.isupper():
                        pred_choice_letter = first_char

                if pred_choice_letter not in choices:
                    error_ids.append(f"{image_id} (could not extract valid choice from '{pred_ans_str}')")
                    pred_choice_letter = None

                if pred_choice_letter == true_choice:
                    # TP
                    stats[true_choice]["TP"] += 1
                    if model_name and dir_name:
                        with open(os.path.join(right_path, f"{line_num}.json"), "w") as fw:  # type: ignore
                            json.dump(data, fw, indent=4, ensure_ascii=False)
                else:
                    # FN
                    stats[true_choice]["FN"] += 1

                    if pred_choice_letter is not None:
                        # FP
                        stats[pred_choice_letter]["FP"] += 1

                    if model_name and dir_name:
                        with open(os.path.join(false_path, f"{line_num}.json"), "w") as fw:  # type: ignore
                            json.dump(data, fw, indent=4, ensure_ascii=False)

            except json.JSONDecodeError:
                error_ids.append(f"line_{line_num} (JSON decode error)")
            except Exception as e:
                error_ids.append(f"line_{line_num} (unexpected error: {e})")

    # F1 of different choices to calculate Macro-F1
    per_class_metrics = {}
    for choice in choices:
        tp = stats[choice]["TP"]
        fp = stats[choice]["FP"]
        fn = stats[choice]["FN"]
        per_class_metrics[choice] = _calculate_metrics(tp, fp, fn)

    # Macro-F1
    macro_f1 = sum(metrics["f1"] for metrics in per_class_metrics.values()) / len(choices)

    # Micro-F1
    total_tp = sum(stats[choice]["TP"] for choice in choices)
    total_fp = sum(stats[choice]["FP"] for choice in choices)
    total_fn = sum(stats[choice]["FN"] for choice in choices)

    micro_metrics = _calculate_metrics(total_tp, total_fp, total_fn)
    micro_f1 = micro_metrics["f1"]

    total_samples = total_tp + total_fn  # number of total samples
    accuracy = total_tp / total_samples if total_samples > 0 else 0.0

    # return final results
    results = {
        "overall_metrics": {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        },
        "micro_details": {**micro_metrics, "TP": total_tp, "FP": total_fp, "FN": total_fn},
        "per_class_stats": {choice: {**stats[choice], **per_class_metrics[choice]} for choice in choices},
        "errors": {
            "parsing_errors": error_ids,
            "failed_status_records": failed_ids,
        },
    }

    return results


if __name__ == "__main__":
    dir_names = ["HallucinationDetection", "FactChecking"]
    # dir_name = "HallucinationDetection"
    # dir_name = "FactChecking"

    model_names = ["qwen", "qwen_25_7b", "qwen_25_32b"]
    # model_name = "qwen"  #  "qvq_72b"   "qwen_25_7b"   "qwen_25_32b"   "qwen_25_72b"
    # model_name = "qvq_72b"
    # model_name = "qwen_25_7b"
    # model_name = "qwen_25_32b"
    # model_name = "qwen_25_72b"

    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    parent_dir = current_dir.parent
    hd_result_dir = os.path.join(parent_dir, "HD_result")

    for model_name in model_names:
        for dir_name in dir_names:
            jsonl_file = os.path.join(hd_result_dir, f"{model_name}", f"result_{dir_name}_{model_name}.jsonl")
            results = calculate_f1_scores_multi_class(jsonl_file, dir_name, hd_result_dir, model_name=model_name)
            print(results)
            with open(
                os.path.join(hd_result_dir, f"{model_name}", f"f1_{dir_name}_result.json"),
                "w",
            ) as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
