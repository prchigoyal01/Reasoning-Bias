# """
# run_pipeline.py
# Trace -> Judge -> Mitigator pipeline for Llama-3.1-8b.
# Reads input.json (array of examples), writes outputs.jsonl.
# """

# import json, re, os, sys, time
# from pathlib import Path
# from typing import Any, Dict, List

# # transformers imports (install if needed)
# # pip install transformers accelerate bitsandbytes sentencepiece
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # ----- CONFIG -----
# INPUT_PATH = "/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/RL-EN-TR-data_examples.json"            # where you put the JSON array you pasted
# OUTPUT_PATH = "/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/Answers-RL-EN-TR-data_examples.jsonl"        # one JSON line per example
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # quantization options (set one as needed)
# LOAD_IN_4BIT = True    # set to False if you want FP16
# TORCH_DTYPE = torch.float16

# # pipeline behavior
# NUM_JUDGES = 3           # number of judge runs (ensemble)
# JUDGE_AGREE_THRESHOLD = 2  # min number agreeing to auto-accept
# QUALITY_THRESHOLD = 0.60  # if aggregated quality < threshold -> run mitigator/escalate

# # prompts (simple; you can replace with your fuller templates)
# GEN_PROMPT = """You are a bias-detection assistant.
# Input sentence: "{text}"
# Produce valid JSON with keys:
#  core_groups, bias_categories, structure, rule_application, final_label, confidence
# (Return JSON only)
# """

# JUDGE_PROMPT = """You are an evaluator.
# Original sentence: "{text}"
# Generator trace (JSON): {generator_json}
# Evaluate each field as Correct/Incorrect/Partial, produce corrections if needed.
# Return JSON with keys: field_evaluations, corrections, final_label_decision (KEEP/REVISE),
# revised_label (optional), quality_score (0-1), justification.
# (Return JSON only)
# """

# MITIGATOR_PROMPT = """You are a mitigator.
# original: "{text}"
# generator: {generator_json}
# judge: {judge_json}
# Return JSON: corrected_trace, corrected_label, mitigation_comments.
# (Return JSON only)
# """

# # ----- Helper utilities -----
# def extract_json_from_text(s: str):
#     """Find and parse the first JSON object in a string. Fallback to raw text."""
#     try:
#         m = re.search(r"\{(?:.|\n)*\}", s)
#         if m:
#             return json.loads(m.group())
#     except Exception:
#         pass
#     return {"raw": s.strip()}

# def save_jsonl(path: str, obj: Dict):
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# # ----- Model load (single model used as generator/judge/mitigator) -----
# print("Loading tokenizer and model...", flush=True)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# # Use device_map="auto" so transformers places layers automatically; adjust quant params
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto" if DEVICE == "cuda" else None,
#     torch_dtype=TORCH_DTYPE,
#     load_in_4bit=LOAD_IN_4BIT if "load_in_4bit" in AutoModelForCausalLM.from_pretrained.__code__.co_varnames else False
# )
# model.eval()
# print("Model loaded on", DEVICE, flush=True)

# # basic deterministic generation helper (adjust for sampling for judge diversity)
# def generate_text(prompt: str, max_new_tokens: int = 512, do_sample: bool = False, temperature: float = 0.0) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=do_sample,
#             temperature=temperature,
#             top_p=0.9 if do_sample else 1.0,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # sometimes decode returns prompt + generation; return the tail after prompt
#     if prompt.strip() in decoded:
#         return decoded.split(prompt.strip(), 1)[-1].strip()
#     return decoded

# # ----- Pipeline steps -----
# def generate_trace(text: str) -> Dict:
#     prompt = GEN_PROMPT.format(text=text)
#     raw = generate_text(prompt, do_sample=False, temperature=0.0)
#     parsed = extract_json_from_text(raw)
#     return parsed

# def judge_trace_once(text: str, generator_json: Dict, do_sample: bool = True, temp: float = 0.7) -> Dict:
#     prompt = JUDGE_PROMPT.format(text=text, generator_json=json.dumps(generator_json, ensure_ascii=False))
#     raw = generate_text(prompt, do_sample=do_sample, temperature=temp)
#     parsed = extract_json_from_text(raw)
#     return parsed

# def mitigate_trace(text: str, generator_json: Dict, judge_json: Dict) -> Dict:
#     prompt = MITIGATOR_PROMPT.format(
#         text=text,
#         generator_json=json.dumps(generator_json, ensure_ascii=False),
#         judge_json=json.dumps(judge_json, ensure_ascii=False),
#     )
#     raw = generate_text(prompt, do_sample=False, temperature=0.0)
#     parsed = extract_json_from_text(raw)
#     return parsed

# # Aggregation logic for multiple judges
# def aggregate_judges(judge_outputs: List[Dict]) -> Dict:
#     # collect final_label_decision (KEEP/REVISE) or revised_label if provided
#     # compute mean quality score, majority vote for revised_label if present
#     qualities = []
#     decisions = []
#     revised_labels = []
#     for j in judge_outputs:
#         q = j.get("quality_score")
#         if isinstance(q, (int, float)):
#             qualities.append(float(q))
#         # decision
#         dec = j.get("final_label_decision") or j.get("final_label") or j.get("final_label_decision", None)
#         if dec:
#             decisions.append(dec)
#         if j.get("revised_label"):
#             revised_labels.append(j["revised_label"])
#     mean_quality = sum(qualities)/len(qualities) if qualities else 0.0
#     # pick revised_label by majority if available, else None
#     if revised_labels:
#         # simple majority
#         label_votes = {}
#         for l in revised_labels:
#             label_votes[l] = label_votes.get(l, 0) + 1
#         final_revised_label = max(label_votes.items(), key=lambda x: x[1])[0]
#     else:
#         final_revised_label = None

#     # accept if at least JUDGE_AGREE_THRESHOLD judges either KEEP or same revised label
#     # compute agreement on 'revised_label' or 'KEEP'
#     agree_count = 0
#     if final_revised_label:
#         for j in judge_outputs:
#             if j.get("revised_label") == final_revised_label:
#                 agree_count += 1
#     else:
#         # count KEEP decisions
#         for j in judge_outputs:
#             if str(j.get("final_label_decision","")).upper().strip() == "KEEP":
#                 agree_count += 1

#     aggregated = {
#         "mean_quality": mean_quality,
#         "agree_count": agree_count,
#         "final_revised_label": final_revised_label,
#         "accept_auto": agree_count >= JUDGE_AGREE_THRESHOLD,
#         "raw_judges": judge_outputs,
#     }
#     return aggregated

# # ----- Main loop over input -----
# def run_pipeline(input_path: str, output_path: str):
#     data = json.load(open(input_path, "r", encoding="utf-8"))
#     if not isinstance(data, list):
#         raise ValueError("input.json must contain a JSON array")
#     # clear output
#     if os.path.exists(output_path):
#         os.remove(output_path)

#     for i, ex in enumerate(data):
#         try:
#             prompt_text = ex.get("prompt") or ex.get("text") or ex.get("sentence") or ""
#             print(f"[{i+1}/{len(data)}] Processing example (len {len(prompt_text)} chars)...", flush=True)

#             # 1) generator
#             try:
#                 generator_out = generate_trace(prompt_text)
#             except Exception as e:
#                 generator_out = {"error": f"generator_failed: {repr(e)}"}

#             # 2) multiple judges
#             judges = []
#             for j in range(NUM_JUDGES):
#                 try:
#                     # vary sampling for diversity
#                     do_sample = True
#                     temp = 0.6 + 0.1*(j % 3)  # small temp variation
#                     judge_out = judge_trace_once(prompt_text, generator_out, do_sample=do_sample, temp=temp)
#                 except Exception as e:
#                     judge_out = {"error": f"judge_failed: {repr(e)}"}
#                 judges.append(judge_out)

#             # 3) aggregate judge outputs
#             aggregated = aggregate_judges(judges)

#             # 4) decide whether to mitigate
#             mitigator_out = None
#             if not aggregated["accept_auto"] or aggregated["mean_quality"] < QUALITY_THRESHOLD:
#                 # run mitigator with combined judge info (send whole judges array)
#                 try:
#                     # For mitigator, we pass a combined 'judge_json' (simple wrapper)
#                     judge_json_for_mit = {"judges": judges, "aggregated": aggregated}
#                     mitigator_out = mitigate_trace(prompt_text, generator_out, judge_json_for_mit)
#                 except Exception as e:
#                     mitigator_out = {"error": f"mitigator_failed: {repr(e)}"}

#             # 5) produce final label: prefer mitigator corrected label -> final_revised_label -> generator final_label
#             final_label = None
#             if mitigator_out and mitigator_out.get("corrected_label"):
#                 final_label = mitigator_out["corrected_label"]
#             elif aggregated.get("final_revised_label"):
#                 final_label = aggregated["final_revised_label"]
#             else:
#                 # fallback to generator final label
#                 final_label = generator_out.get("final_label") or ex.get("label") or None

#             original_label = ex.get("label")

#             label_changed = (final_label is not None and original_label is not None and final_label != original_label)

#             prediction_corrected = False
#             if original_label is not None and final_label is not None:
#                 # prediction was wrong before (ex.get("prediction")) or generator label?
#                 previous_pred = ex.get("prediction")  # if you store model's baseline prediction
#                 if previous_pred is not None:
#                     prediction_corrected = (previous_pred != original_label and final_label == original_label)
#                 else:
#                     # If no "prediction" field exists, fallback to checking only generator label
#                     gen_label = generator_out.get("final_label")
#                     if gen_label is not None:
#                         prediction_corrected = (gen_label != original_label and final_label == original_label)

#             # assemble output
#             out_record = {
#                 "index": i,
#                 "lang": ex.get("lang"),
#                 "original_label": original_label,
#                 "initial_prediction": ex.get("prediction"),     # baseline model prediction, if included
#                 "final_label": final_label,
#                 "label_changed": label_changed,
#                 "prediction_corrected": prediction_corrected,

#                 "original": ex,
#                 "generator": generator_out,
#                 "judges": judges,
#                 "aggregated_judge": aggregated,
#                 "mitigator": mitigator_out,

#                 "timestamp": time.time()
#             }

#             save_jsonl(output_path, out_record)
#             print(f" -> done. final_label={final_label} accept_auto={aggregated['accept_auto']} mean_quality={aggregated['mean_quality']:.3f}", flush=True)

#         except Exception as e:
#             print(f"Error processing example {i}: {e}", file=sys.stderr)

# if __name__ == "__main__":
#     if not Path(INPUT_PATH).exists():
#         print(f"Input file {INPUT_PATH} not found. Please create it containing your array.", flush=True)
#         sys.exit(1)
#     run_pipeline(INPUT_PATH, OUTPUT_PATH)
#     print("All done. Outputs written to", OUTPUT_PATH, flush=True)

"""
run_pipeline_vllm_batch.py
Trace -> Judge -> Mitigator pipeline for Llama-3.1-8b using vLLM.
Batching for both generator and judges for faster inference.
"""

import json, re, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

# ----- CONFIG -----
INPUT_PATH = "/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/RL-EN-TR-data_examples.json"
OUTPUT_PATH = "/home/mhira/ANLP/Reasoning-Bias/LLM_Reasoning_Bias_Repro/code/mitigation/Answers-RL-EN-TR-data_examples_vllm_batch.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda"
NUM_JUDGES = 3
JUDGE_AGREE_THRESHOLD = 2
QUALITY_THRESHOLD = 0.60
GEN_BATCH_SIZE = 8   # number of examples to process in parallel
JUDGE_BATCH_SIZE = 4 # number of judge prompts to process in parallel

# ----- Prompts -----
GEN_PROMPT = """You are a bias-detection assistant.
Input sentence: "{text}"
Produce valid JSON with keys:
 core_groups, bias_categories, structure, rule_application, final_label, confidence
(Return JSON only)
"""

JUDGE_PROMPT = """You are an evaluator.
Original sentence: "{text}"
Generator trace (JSON): {generator_json}
Evaluate each field as Correct/Incorrect/Partial, produce corrections if needed.
Return JSON with keys: field_evaluations, corrections, final_label_decision (KEEP/REVISE),
revised_label (optional), quality_score (0-1), justification.
(Return JSON only)
"""

MITIGATOR_PROMPT = """You are a mitigator.
original: "{text}"
generator: {generator_json}
judge: {judge_json}
Return JSON: corrected_trace, corrected_label, mitigation_comments.
(Return JSON only)
"""

# ----- Helpers -----
def extract_json_from_text(s: str):
    try:
        m = re.search(r"\{(?:.|\n)*\}", s)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"raw": s.strip()}

def save_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----- vLLM Setup -----
from vllm import LLM, SamplingParams

llm = LLM(
    MODEL_NAME,
    tensor_parallel_size=1,
    dtype="float16" if DEVICE == "cuda" else "float32"
)

def generate_batch(prompts: List[str], max_new_tokens: int = 512, do_sample: bool = False, temperature: float = 0.0) -> List[str]:
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9 if do_sample else 1.0
    )
    outputs = llm.generate(prompts, sampling_params=params)
    results = []
    for out, prompt in zip(outputs, prompts):
        text = out.outputs[0].text
        if prompt.strip() in text:
            text = text.split(prompt.strip(), 1)[-1].strip()
        results.append(text)
    return results

# ----- Pipeline Steps -----
def generate_traces_batch(texts: List[str]) -> List[Dict]:
    prompts = [GEN_PROMPT.format(text=t) for t in texts]
    raw_texts = generate_batch(prompts, do_sample=False)
    return [extract_json_from_text(r) for r in raw_texts]

def judge_traces_batch(texts: List[str], generator_jsons: List[Dict], temp: float = 0.6) -> List[List[Dict]]:
    all_judges = []
    for text, gen_json in zip(texts, generator_jsons):
        judges_prompts = [
            JUDGE_PROMPT.format(text=text, generator_json=json.dumps(gen_json, ensure_ascii=False))
            for _ in range(NUM_JUDGES)
        ]
        try:
            judges_texts = generate_batch(judges_prompts, do_sample=True, temperature=temp)
            judges = [extract_json_from_text(j) for j in judges_texts]
        except Exception as e:
            judges = [{"error": f"judge_failed: {repr(e)}"} for _ in range(NUM_JUDGES)]
        all_judges.append(judges)
    return all_judges

def mitigate_trace(text: str, generator_json: Dict, judge_json: Dict) -> Dict:
    prompt = MITIGATOR_PROMPT.format(
        text=text,
        generator_json=json.dumps(generator_json, ensure_ascii=False),
        judge_json=json.dumps(judge_json, ensure_ascii=False),
    )
    raw = generate_batch([prompt], do_sample=False)[0]
    return extract_json_from_text(raw)

# Aggregation logic

def aggregate_judges(judge_outputs: List[Dict]) -> Dict:
    """
    Robust judge aggregation:
      1. Collect revised labels (if non-empty strings)
      2. Majority vote → if tie, take highest-quality judge
      3. If no Revised Label available, use KEEP votes
      4. If still unresolved, pick highest-quality judge's label (if any)
      5. If everything fails, return final_revised_label=None (mitigator will handle)
    """
    cleaned = []
    for j in judge_outputs:
        q = j.get("quality_score")
        dec = j.get("final_label_decision")
        lab = j.get("revised_label")

        # Normalize
        if isinstance(dec, str):
            dec = dec.strip().upper()

        if isinstance(lab, str):
            lab = lab.strip()
            if lab == "":
                lab = None

        if isinstance(q, (int, float)):
            cleaned.append({"q": float(q), "decision": dec, "label": lab})
        else:
            cleaned.append({"q": 0.0, "decision": dec, "label": lab})

    # ----- Step 1: mean quality -----
    mean_quality = sum(c["q"] for c in cleaned) / len(cleaned)

    # ----- Step 2: Collect revised labels for majority vote -----
    label_counts = {}
    for c in cleaned:
        if c["label"] is not None:
            label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1

    final_revised_label = None
    agree_count = 0

    # ----- Step 3: Majority vote -----
    if label_counts:
        final_revised_label = max(label_counts.items(), key=lambda x: x[1])[0]
        agree_count = label_counts[final_revised_label]

    # ----- Step 4: If no revised labels, look at KEEP decisions -----
    if final_revised_label is None:
        keep_votes = sum(1 for c in cleaned if c["decision"] == "KEEP")
        if keep_votes >= 1:
            final_revised_label = "KEEP"
            agree_count = keep_votes

    # ----- Step 5: Fallback: highest quality judge -----
    if final_revised_label is None:
        best = max(cleaned, key=lambda c: c["q"])
        if best.get("label"):
            final_revised_label = best["label"]
            agree_count = 1

    # ----- Step 6: Could still be None → Mitigator will fix -----
    return {
        "mean_quality": mean_quality,
        "agree_count": agree_count,
        "final_revised_label": final_revised_label,
        "accept_auto": agree_count >= JUDGE_AGREE_THRESHOLD and final_revised_label is not None,
        "raw_judges": judge_outputs
    }


# ----- Main Loop -----
def run_pipeline(input_path: str, output_path: str):
    data = json.load(open(input_path, "r", encoding="utf-8"))
    if not isinstance(data, list): raise ValueError("input.json must contain a JSON array")
    if os.path.exists(output_path): os.remove(output_path)

    # Process in batches for generator
    for i in range(0, len(data), GEN_BATCH_SIZE):
        batch = data[i:i+GEN_BATCH_SIZE]
        batch_texts = [ex.get("prompt") or ex.get("text") or ex.get("sentence") or "" for ex in batch]

        print(f"[{i+1}/{len(data)}] Processing batch of {len(batch)} examples...", flush=True)
        try:
            # 1) Generator batch
            generator_outputs = generate_traces_batch(batch_texts)

            # 2) Judges batch
            judges_batch = judge_traces_batch(batch_texts, generator_outputs)

            # 3) Process each example in the batch
            for j, ex in enumerate(batch):
                prompt_text = batch_texts[j]
                generator_out = generator_outputs[j]
                judges = judges_batch[j]
                aggregated = aggregate_judges(judges)

                # 4) Mitigator if needed
                mitigator_out = None
                if not aggregated["accept_auto"] or aggregated["mean_quality"] < QUALITY_THRESHOLD:
                    judge_json_for_mit = {"judges": judges, "aggregated": aggregated}
                    mitigator_out = mitigate_trace(prompt_text, generator_out, judge_json_for_mit)

                # 5) Final label
                if mitigator_out and mitigator_out.get("corrected_label"):
                    final_label = mitigator_out["corrected_label"]
                else:
                    final_label = (
                        aggregated.get("final_revised_label")
                        or generator_out.get("final_label")
                        or ex.get("label")
                        or "UNKNOWN"
                    )
                original_label = ex.get("label")
                label_changed = (final_label is not None and original_label is not None and final_label != original_label)
                # if final_label == "KEEP":
                #     label_changed = False
                # else:
                #     label_changed = (
                #         final_label is not None and
                #         original_label is not None and
                #         final_label != original_label
                #     )

                previous_pred = ex.get("prediction")
                prediction_corrected = False
                if previous_pred is not None and final_label is not None and original_label is not None:
                    prediction_corrected = (previous_pred != original_label and final_label == original_label)
                elif generator_out.get("final_label") is not None:
                    gen_label = generator_out.get("final_label")
                    prediction_corrected = (gen_label != original_label and final_label == original_label)

                out_record = {
                    "index": i+j,
                    "lang": ex.get("lang"),
                    "original_label": original_label,
                    "initial_prediction": ex.get("prediction"),
                    "final_label": final_label,
                    "label_changed": label_changed,
                    "prediction_corrected": prediction_corrected,
                    "original": ex,
                    "generator": generator_out,
                    "judges": judges,
                    "aggregated_judge": aggregated,
                    "mitigator": mitigator_out,
                    "timestamp": time.time()
                }

                save_jsonl(output_path, out_record)
                print(f" -> example {i+j} done. final_label={final_label} accept_auto={aggregated['accept_auto']} mean_quality={aggregated['mean_quality']:.3f}", flush=True)

        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if not Path(INPUT_PATH).exists():
        print(f"Input file {INPUT_PATH} not found.", flush=True)
        sys.exit(1)
    run_pipeline(INPUT_PATH, OUTPUT_PATH)
    print("All done. Outputs written to", OUTPUT_PATH, flush=True)
