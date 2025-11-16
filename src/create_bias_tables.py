import os
import json

RESULTS_DIR = "results_new"

languages = ["en", "es", "tr", "nl"]

non_control_categories = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Physical_appearance",
    "SES",
    "Sexual_orientation"
]

control_categories = [c + "_control" for c in non_control_categories]

eval_modes = ["short_answer", "cot", "reasoning"]

model_for_mode = {
    "short_answer": ["Llama-2-7b-chat-hf", "Llama-3.1-8B-Instruct"],             # example
    "cot":          ["Llama-2-7b-chat-hf", "Llama-3.1-8B-Instruct"],  # example
    "reasoning":    ["DeepSeek-R1-Distill-Llama-8B"]
}

model_for_mode_name = {
    "short_answer": ["Llama-2-7b", "Llama-3.1-8B-Instruct"],
    "cot":          ["Llama-2-7b", "Llama-3.1-8B-Instruct"],
    "reasoning":    ["DeepSeek-R1-8B"]
}


def load_metrics(model, lang, category, mode):
    fname = f"results_{model}_{lang}_{category}_{mode}.json"
    path = os.path.join(RESULTS_DIR, fname)
    
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        d = json.load(f)
    
    acc = d["metrics"]["accuracy_metrics"]["valid_accuracy"]
    amb = d["metrics"]["bias_metrics"]["bias_amb"]
    dis = d["metrics"]["bias_metrics"]["bias_dis"]

    return acc, amb, dis


def latex_cell(metrics):
    if metrics is None:
        return "- & - & -"
    acc, amb, dis = metrics
    return f"{acc:.3f} & {amb:.3f} & {dis:.3f}"

def load_raw_metrics(model, lang, category, mode):
    """Load raw metrics (ACC, BiasAmb, BiasDis, weights) from file."""
    fname = f"results_{model}_{lang}_{category}_{mode}.json"
    path = os.path.join(RESULTS_DIR, fname)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        d = json.load(f)

    acc = d["metrics"]["accuracy_metrics"]["valid_accuracy"]
    bias_amb = d["metrics"]["bias_metrics"]["bias_amb"]
    bias_dis = d["metrics"]["bias_metrics"]["bias_dis"]

    # weights
    N_acc = d["metrics"]["accuracy_metrics"]["total_examples"]
    N_amb = d["metrics"]["bias_metrics"]["n_amb_not_unk"]
    N_dis = d["metrics"]["bias_metrics"]["n_disamb_not_unk"]

    return acc, bias_amb, bias_dis, N_acc, N_amb, N_dis

def compute_weighted_all(categories, lang, mode):
    """
    Weighted ALL row computed using:
    - N_acc: total_examples
    - N_amb: n_amb_not_unk
    - N_dis: n_disamb_not_unk
    """
    results = {}

    for model in model_for_mode[mode]:

        sum_acc = 0
        sum_acc_w = 0

        sum_amb = 0
        sum_amb_w = 0

        sum_dis = 0
        sum_dis_w = 0

        for cat in categories:
            m = load_raw_metrics(model, lang, cat, mode)
            if m is None:
                continue

            acc, amb, dis, N_acc, N_amb, N_dis = m

            # Accuracy weighted
            sum_acc += acc * N_acc
            sum_acc_w += N_acc

            # BiasAmb weighted
            if N_amb > 0:
                sum_amb += amb * N_amb
                sum_amb_w += N_amb

            # BiasDis weighted
            if N_dis > 0:
                sum_dis += dis * N_dis
                sum_dis_w += N_dis

        if sum_acc_w == 0:
            results[model] = None
        else:
            acc_all = sum_acc / sum_acc_w
            amb_all = sum_amb / sum_amb_w if sum_amb_w > 0 else 0
            dis_all = sum_dis / sum_dis_w if sum_dis_w > 0 else 0
            results[model] = (acc_all, amb_all, dis_all)

    return results

def generate_mode_table(categories, lang, mode, table_title, label_prefix):
    models = model_for_mode[mode]
    num_cols = 1 + 3 * len(models)  # Category + each model's ACC/Amb/Dis
    models_name = model_for_mode_name[mode]

    # Begin table
    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{table_title} (Language = {lang.upper()})}}")
    latex.append(f"\\label{{tab:{label_prefix}_{lang}_{mode}}}")
    latex.append("")
    latex.append("\\small")
    latex.append(f"\\begin{{tabularx}}{{\\textwidth}}{{l *{{{3 * len(models)}}}{{c}}}}")
    latex.append("\\toprule")

    # Column headers
    header = ["Category"]
    for model in models_name:
        header.append(f"\\multicolumn{{3}}{{c}}{{{model}}}")
    latex.append(" & ".join(header) + " \\\\")

    # cmidrules
    col_start = 2
    for _ in models:
        latex.append(f"\\cmidrule(lr){{{col_start}-{col_start+2}}}")
        col_start += 3

    # Subheaders
    sub = [" "]
    for _ in models:
        sub.extend(["ACC", "BiasAmb", "BiasDis"])
    latex.append(" & ".join(sub) + " \\\\")
    latex.append("\\midrule")

    # Rows
    for cat in categories:
        row = [cat.replace("_", "\\_")]
        for model in models:
            metrics = load_metrics(model, lang, cat, mode)
            row.append(latex_cell(metrics))
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\midrule")
    all_row = ["ALL"]
    weighted = compute_weighted_all(categories, lang, mode)

    for model in models:
        metrics = weighted[model]
        all_row.append(latex_cell(metrics))

    latex.append(" & ".join(all_row) + " \\\\")

    # End table
    latex.append("\\bottomrule")
    latex.append("\\end{tabularx}")
    latex.append("")
    latex.append("\\end{table}")
    latex.append("")



    return "\n".join(latex)


def main():
    for lang in languages:
        print("\n============================================")
        print(f"LATEX TABLES FOR LANGUAGE = {lang.upper()}")
        print("============================================\n")

        for mode in eval_modes:
            print("----- NON-CONTROL:", mode, "-----")
            print(generate_mode_table(
                non_control_categories,
                lang,
                mode,
                f"Bias and Accuracy Results for Non-Control Categories ({mode})",
                "noncontrol"
            ))

            print("----- CONTROL:", mode, "-----")
            print(generate_mode_table(
                control_categories,
                lang,
                mode,
                f"Bias and Accuracy Results for Control Categories ({mode})",
                "control"
            ))


if __name__ == "__main__":
    main()