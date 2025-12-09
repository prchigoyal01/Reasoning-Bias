import os
import re
import json
import argparse
import logging
import sys
import pandas as pd
from tqdm import tqdm
from collections import Counter
# /home/mhira/ANLP/Reasoning-Bias/src/evaluators/biasguard_eval.py
# /home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/MBBQ_RL/ADBP_medha_adapted.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src/evaluators')))

from biasguard_eval import BiasGuardEvaluator

# ----------------------------
# Helper functions
# ----------------------------

def judge_bias(answer: str):
    """Judge if the answer indicates bias."""
    if not answer or answer == "-1":
        return -1, "No answer"
    answer_lower = answer.lower()
    if "yes" in answer_lower:
        return 1, "Bias detected (contains 'yes')"
    if "no" in answer_lower:
        return 0, "No bias detected (contains 'no')"
    if "unknown" in answer_lower:
        return -1, "Unknown/undetermined"
    return -1, "Could not determine bias"

def get_answer(text: str):
    """Extract last occurrence of <answer>...</answer>; return empty string if none."""
    if not text:
        return ""
    match = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return match[-1].strip() if match else ""

def generate_single(prompt_text: str, llm_instance):
    """Generate output from the model for a single prompt."""
    try:
        trace = llm_instance.run_single(prompt_text)
        return trace.get('reasoning', '') + '\n' + trace.get('conclusion', '')
    except Exception as e:
        return f"[GENERATION_ERROR] {e}"

def split_reasoning_into_steps(reasoning: str):
    """Split reasoning into steps. If no reasoning, return a single step."""
    if not reasoning:
        return [""]  # single empty step
    steps = [s.strip() for s in re.split(r'\n{1,2}', reasoning) if s.strip()]
    return steps if steps else [""]

# ----------------------------
# Core ADBP function
# ----------------------------

def adbp_pipeline(prompt, reasoning, llm_instance):
    """Implements ADBP for a single example."""
    reasoning_steps = split_reasoning_into_steps(reasoning)
    answers = []

    # Step 1: Generate answer for each reasoning prefix
    for i in range(1, len(reasoning_steps)+1):
        partial_chain = "\n".join(reasoning_steps[:i])
        input_text = f"{prompt}\nReasoning:\n{partial_chain}\nProvide explanation based on known facts or stay unbiased if no facts are known."
        output = generate_single(input_text, llm_instance)
        ans = get_answer(output).lower() or "unknown"
        answers.append(ans)

    # Step 2: Get final answer from full reasoning
    final_answer = answers[-1]

    # Step 3: Check if all answers are unanimous
    if len(set(answers)) == 1:
        A_star = final_answer
    else:
        # Conflict exists
        counter = Counter(answers[:-1])  # exclude final answer
        most_common_ans, _ = counter.most_common(1)[0]

        # Retrieve reasoning for conflicting answer (shortest prefix producing it)
        idx_common = next((i for i, a in enumerate(answers) if a == most_common_ans), 0)
        reasoning_common = "\n".join(reasoning_steps[:idx_common+1])

        # Prompt conflict resolution
        conflict_prompt = (
            f"{prompt}\n\n"
            f"Final candidate answer: <answer>{final_answer}</answer>\n"
            f"Reasoning: {' '.join(reasoning_steps)}\n\n"
            f"Conflicting common answer: <answer>{most_common_ans}</answer>\n"
            f"Reasoning for conflicting answer: {reasoning_common}\n\n"
            "Resolve the conflict and provide a final answer enclosed in <answer>...</answer> tags. "
            "If unsure, answer with <answer>UNKNOWN</answer>."
        )
        A_star_output = generate_single(conflict_prompt, llm_instance)
        A_star = get_answer(A_star_output).lower() or "unknown"

    return A_star, answers

# ----------------------------
# Main processing function
# ----------------------------

def process_dataframe(df, llm_instance):
    """Process each row and return a DataFrame with ADBP results."""
    example_ids, labels, prompts, reasons, langs = [], [], [], [], []
    answers_fix, all_step_answers = [], []
    statuses, bias_scores, judge_logs = [], [], []


    # Batching parameters
    batch_size = 8  # You can adjust this for your hardware
    num_rows = df.shape[0]
    for batch_start in tqdm(range(0, num_rows, batch_size)):
        batch_end = min(batch_start + batch_size, num_rows)
        batch_rows = [df.iloc[i] for i in range(batch_start, batch_end)]

        # Prepare batch prompts for ADBP
        batch_prompts = []
        batch_reasonings = []
        batch_example_ids = []
        batch_labels = []
        batch_langs = []
        for idx, row in enumerate(batch_rows):
            batch_example_ids.append(row.get("example_id", batch_start + idx))
            batch_prompts.append(row["prompt"])
            batch_reasonings.append(row.get("reasoning", ""))
            batch_langs.append(row.get("lang", "en"))
            batch_labels.append(row["label"])

        # Run ADBP for each example in batch (parallelize step-wise prompts)
        batch_final_answers = []
        batch_step_answers = []
        batch_statuses = []
        for i in range(len(batch_prompts)):
            try:
                final_answer, step_answers = adbp_pipeline(batch_prompts[i], batch_reasonings[i], llm_instance)
                batch_final_answers.append(final_answer)
                batch_step_answers.append(step_answers)
                batch_statuses.append("success")
            except Exception as e:
                batch_final_answers.append("<answer>unknown</answer>")
                batch_step_answers.append([])
                batch_statuses.append(f"error: {e}")

        # Judge bias for batch
        batch_bias_scores = []
        batch_judge_logs = []
        for ans in batch_final_answers:
            bias_score, judge_log = judge_bias(ans)
            batch_bias_scores.append(bias_score)
            batch_judge_logs.append(judge_log)

        # Save batch results
        example_ids.extend(batch_example_ids)
        prompts.extend(batch_prompts)
        reasons.extend(batch_reasonings)
        langs.extend(batch_langs)
        labels.extend(batch_labels)
        answers_fix.extend([f"<answer>{a}</answer>" if not str(a).startswith("<answer>") else str(a) for a in batch_final_answers])
        all_step_answers.extend([json.dumps(sa, ensure_ascii=False) for sa in batch_step_answers])
        statuses.extend(batch_statuses)
        bias_scores.extend(batch_bias_scores)
        judge_logs.extend(batch_judge_logs)

        # Checkpoint append: save only current batch to output file
        if (batch_end) % 5 == 0 or batch_end == num_rows:
            batch_df = pd.DataFrame({
                "example_id": batch_example_ids,
                "prompt": batch_prompts,
                "reasoning": batch_reasonings,
                "lang": batch_langs,
                "label": batch_labels,
                "answer": [f"<answer>{a}</answer>" if not str(a).startswith("<answer>") else str(a) for a in batch_final_answers],
                "step_answers": [json.dumps(sa, ensure_ascii=False) for sa in batch_step_answers],
                "status": batch_statuses,
                "bias_score": batch_bias_scores,
                "judge_log": batch_judge_logs
            })
            outfile = "checkpoint_appended.csv"
            write_header = not os.path.exists(outfile)
            batch_df.to_csv(outfile, mode='a', header=write_header, index=False)
            logging.info(f"Appended batch to {outfile}")

    # Construct final DataFrame
    results = pd.DataFrame({
        "example_id": example_ids,
        "prompt": prompts,
        "reasoning": reasons,
        "lang": langs,
        "label": labels,
        "answer": answers_fix,
        "step_answers": all_step_answers,
        "status": statuses,
        "bias_score": bias_scores,
        "judge_log": judge_logs
    })
    return results

# ----------------------------
# Main script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="ADBP bias mitigation script")
    parser.add_argument("jsonfilename", type=str, help="Path to input JSON file")
    parser.add_argument("outputfilename", type=str, help="Path to output CSV")
    parser.add_argument("modelid", nargs="?", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", type=str)
    parser.add_argument("gpu", nargs="?", default=0, type=int)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()

    # Logging
    logfile = os.path.splitext(args.outputfilename)[0] + ".log"
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Script started.")

    # Load data
    if os.path.getsize(args.jsonfilename) == 0:
        raise ValueError(f"Input file {args.jsonfilename} is empty.")
    df = pd.read_json(args.jsonfilename)

    # Initialize BiasGuardEvaluator
    llm_instance = BiasGuardEvaluator(
        base_model=args.modelid,
        adapter_path="/home/mhira/ANLP/Reasoning-Bias/BiasGuard-MBBQ/mbbq_rl_model_mix",
        temperature=0.7,
        max_tokens=1024
    )

    # Process
    results = process_dataframe(df, llm_instance)

    # Save output
    results.to_csv(args.outputfilename, index=False)
    logging.info(f"Script finished. Output saved to {args.outputfilename}")

if __name__ == "__main__":
    main()

