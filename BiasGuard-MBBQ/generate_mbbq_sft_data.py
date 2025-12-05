"""
Generate SFT training data for BiasGuard-MBBQ.

Uses reasoning from existing CoT evaluation results, truncating at answer markers.
"""

import json
import os
import re
from tqdm import tqdm
from config_mbbq import (
    RESULTS_DIR, MBBQ_DATA_DIR, CATEGORIES,
    SFT_DATA_PATH, PROMPT_TEMPLATE_PATH, 
    BIAS_TYPES_PATH, STANDARDS_PATH
)
from convert_mbbq_to_biasguard import convert_mbbq_to_biasguard_format, load_mbbq_category

BIASED = "Yes, the sentence is biased."
UNBIASED = "No, the sentence is unbiased."


def get_mbbq_system_instruction():
    """Load MBBQ-specific bias types and standards."""
    
    with open(BIAS_TYPES_PATH, 'r') as f:
        bias_dict = json.load(f)
        bias_types = ""
        for bias_name, bias in bias_dict.items():
            bias_types += "\n" + bias_name + ":\n"
            for k, v in bias.items():
                bias_types += f"{k}: {v}\n"
    
    with open(STANDARDS_PATH, 'r') as f:
        standards_dict = json.load(f)
        standards = ""
        for _, standard in standards_dict.items():
            standards += "\n"
            for k, v in standard.items():
                standards += f"{k}: {v}\n"

    with open(PROMPT_TEMPLATE_PATH, 'r') as f:
        prompt_template = json.load(f)
        prompt = ""
        for key, value in prompt_template.items():
            value = value.replace("{standards}", standards)
            value = value.replace("{bias_types}", bias_types)
            prompt += f"{key}\n{value}\n\n"

    return prompt


def format_prompt(example, sys_inst):
    """Format example with system instruction."""
    messages = [
        {"role": "SYSTEM", "content": sys_inst},
        {"role": "USER", "content": example["prompt"]},
    ]
    text = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])
    text += "\n[ASSISTANT]: "
    return {"prompt_text": text}


def truncate_reasoning_at_answer(reasoning_text):
    """
    Truncate reasoning text at answer markers to avoid hallucinations.
    
    Looks for patterns like:
    - "answer:"
    - "answer is"
    - "Answer:"
    - "Answer is"
    - "Final Answer:"
    - "final answer:"
    """
    if not reasoning_text:
        return ""
    
    # Patterns to match (case-insensitive)
    patterns = [
        r'\banswer\s*:\s*',
        r'\banswer\s+is\s+',
        r'\bfinal\s+answer\s*:\s*',
        r'\bfinal\s+answer\s+is\s+',
        r'\bAnswer\s*:\s*',
        r'\bAnswer\s+is\s+',
        r'\bFinal\s+Answer\s*:\s*',
        r'\bFinal\s+Answer\s+is\s+',
    ]
    
    text_lower = reasoning_text.lower()
    min_idx = len(reasoning_text)
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            idx = match.start()
            if idx < min_idx:
                min_idx = idx
    
    if min_idx < len(reasoning_text):
        # Truncate at the answer marker
        truncated = reasoning_text[:min_idx].strip()
        return truncated
    
    return reasoning_text.strip()


def extract_conclusion_from_reasoning(reasoning_text, label):
    """
    Extract or construct conclusion from reasoning.
    
    Args:
        reasoning_text: Full reasoning text
        label: 0 (unbiased) or 1 (biased)
    
    Returns:
        Conclusion string in BiasGuard format
    """
    # Look for existing conclusion
    conclusion_patterns = [
        r'##\s*Conclusion\s*##\s*:?\s*',
        r'Conclusion\s*:?\s*',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, reasoning_text, re.IGNORECASE)
        if match:
            # Extract conclusion after the marker
            conclusion = reasoning_text[match.end():].strip()
            # Take first sentence or first 200 chars
            if '\n' in conclusion:
                conclusion = conclusion.split('\n')[0]
            conclusion = conclusion[:200].strip()
            
            # Check if it already has the right format
            if BIASED in conclusion or UNBIASED in conclusion:
                return conclusion
    
    # Construct conclusion if not found
    if label == 0:
        conclusion = f"## Conclusion ##: {UNBIASED}"
    else:
        conclusion = f"## Conclusion ##: {BIASED}"
    
    return conclusion


def load_reasoning_results(category, results_dir=RESULTS_DIR):
    """
    Load reasoning outputs from CoT evaluation results.
    
    Note: Each MBBQ example has one reasoning output, but we create 3 BiasGuard examples.
    We'll use the same reasoning for all 3 choices of the same MBBQ example.
    
    Returns:
        Dict mapping example_id to reasoning text
    """
    # Find the results file
    results_file = os.path.join(
        results_dir, 
        f"results_Llama-3.1-8B-Instruct_en_{category}_cot.json"
    )
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return {}
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reasoning_outputs = data.get('reasoning_outputs', [])
    
    # Load original MBBQ data to map indices
    mbbq_examples = load_mbbq_category(category)
    
    # Create mapping: example_id -> reasoning
    # reasoning_outputs[i] corresponds to mbbq_examples[i] (in file order)
    # We'll filter ambig examples later when processing
    reasoning_map = {}
    
    for i, reasoning in enumerate(reasoning_outputs):
        if i < len(mbbq_examples):
            example_id = mbbq_examples[i].get('example_id', i)
            reasoning_map[example_id] = reasoning
    
    return reasoning_map


def generate_sft_data_mbbq():
    """Main function to generate SFT data from MBBQ using existing reasoning."""
    
    print("="*60)
    print("Generating SFT Data for BiasGuard-MBBQ")
    print("="*60)
    
    # Load system instruction
    print("\nLoading system instruction...")
    sys_inst = get_mbbq_system_instruction()
    
    # Convert MBBQ to BiasGuard format
    print("\nConverting MBBQ to BiasGuard format...")
    all_biasguard_examples = []
    category_example_map = {}  # Map category to list of examples
    
    for category in CATEGORIES:
        print(f"\nProcessing {category}...")
        mbbq_examples = load_mbbq_category(category)
        
        # Load reasoning results for this category
        reasoning_map = load_reasoning_results(category)
        print(f"  Loaded {len(reasoning_map)} reasoning outputs")
        
        category_examples = []
        for example in mbbq_examples:
            # Filter out ambiguous examples (same as in convert_mbbq_to_biasguard.py)
            if example.get('context_condition') == "ambig":
                continue
            
            # Convert to BiasGuard format (this will filter out unknown answers)
            # Returns list of 0-2 examples (0 if all answers were unknown, 1-2 if some were valid)
            biasguard_examples = convert_mbbq_to_biasguard_format(example)
            
            # Skip if no valid examples (all answers were unknown)
            if not biasguard_examples:
                continue
            
            # Attach reasoning to each BiasGuard example
            example_id = example.get('example_id', -1)
            
            # Get reasoning for this MBBQ example (same reasoning used for all choices)
            reasoning = reasoning_map.get(example_id, "")
            
            for bg_ex in biasguard_examples:
                
                # Truncate at answer markers
                reasoning_truncated = truncate_reasoning_at_answer(reasoning)
                
                # Extract or construct conclusion
                conclusion = extract_conclusion_from_reasoning(reasoning_truncated, bg_ex['prompt_label'])
                
                # Combine reasoning and conclusion
                if reasoning_truncated:
                    response = reasoning_truncated + "\n\n" + conclusion
                else:
                    response = conclusion
                
                bg_ex['response'] = response
                bg_ex['reasoning'] = reasoning_truncated
                bg_ex['conclusion'] = conclusion
                bg_ex['category'] = category
                
                category_examples.append(bg_ex)
        
        category_example_map[category] = category_examples
        all_biasguard_examples.extend(category_examples)
        print(f"  Created {len(category_examples)} examples for {category}")
    
    print(f"\nTotal examples: {len(all_biasguard_examples)}")
    
    # Format prompts
    print("\nFormatting prompts...")
    formatted_examples = []
    for ex in tqdm(all_biasguard_examples):
        formatted = format_prompt(ex, sys_inst)
        formatted['prompt'] = ex['prompt']
        formatted['prompt_label'] = ex['prompt_label']
        formatted['response'] = ex['response']
        formatted['conclusion'] = ex['conclusion']
        formatted['category'] = ex.get('category', '')
        formatted_examples.append(formatted)
    
    # Filter valid examples
    print("\nFiltering valid examples...")
    valid_examples = []
    for ex in tqdm(formatted_examples):
        response = ex['response']
        conclusion = ex['conclusion']
        label = ex['prompt_label']
        
        # Check if conclusion matches label
        is_correct = (
            (label == 0 and UNBIASED in conclusion and BIASED not in conclusion) or
            (label == 1 and BIASED in conclusion and UNBIASED not in conclusion)
        )
        
        # Basic quality checks
        if is_correct and len(response) > 50:  # Minimum length
            valid_examples.append({
                "prompt": ex['prompt'],
                "label": label,
                "response": response,
                "conclusion": conclusion,
                "category": ex['category'],
            })
    
    print(f"Valid examples: {len(valid_examples)} / {len(formatted_examples)}")
    
    # Save to file
    print(f"\nSaving to {SFT_DATA_PATH}...")
    with open(SFT_DATA_PATH, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    label_counts = {}
    category_counts = {}
    for ex in valid_examples:
        label = ex['label']
        category = ex['category']
        label_counts[label] = label_counts.get(label, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count}")
    
    print(f"\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    print(f"\n✓ SFT data saved to {SFT_DATA_PATH}")
    print("="*60)


if __name__ == "__main__":
    generate_sft_data_mbbq()

