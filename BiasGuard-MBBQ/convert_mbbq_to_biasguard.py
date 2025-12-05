"""
Convert MBBQ dataset to BiasGuard format.

Each MBBQ example becomes 3 BiasGuard examples:
- 1 unbiased (correct answer, label=0)
- 2 biased (incorrect answers, label=1)
"""

import json
import os
from config_mbbq import MBBQ_DATA_DIR, CATEGORIES


def convert_mbbq_to_biasguard_format(example):
    """
    Convert MBBQ example to BiasGuard format.
    
    Args:
        example: MBBQ example dict with keys: context, question, ans0, ans1, ans2, label, answer_info
    
    Returns:
        List of BiasGuard-formatted examples (excluding unknown answers)
    """
    expanded_examples = []
    answer_key = example['label']  # 0, 1, or 2
    
    # Get all answer choices
    choices = [example['ans0'], example['ans1'], example['ans2']]
    
    # Identify which choice is "unknown" by checking answer_info
    # answer_info format: {"ans0": ["text", "type"], "ans1": [...], "ans2": [...]}
    # where type can be "unknown", "disabled", "nonDisabled", etc.
    answer_info = example.get('answer_info', {})
    unknown_indices = set()
    
    for i in range(3):
        ans_key = f'ans{i}'
        if ans_key in answer_info:
            # Check if the second element (type) is "unknown"
            if len(answer_info[ans_key]) >= 2 and answer_info[ans_key][1] == "unknown":
                unknown_indices.add(i)
    
    # If answer_info doesn't have the info, check the text itself for common unknown phrases
    if not unknown_indices:
        unknown_phrases = [
            "cannot be determined", "not answerable", "not enough info", 
            "unknown", "cannot answer", "not enough information"
        ]
        for i, choice in enumerate(choices):
            choice_lower = choice.lower()
            if any(phrase in choice_lower for phrase in unknown_phrases):
                unknown_indices.add(i)
    
    # Only process non-unknown choices
    for i, choice in enumerate(choices):
        # Skip unknown answers
        if i in unknown_indices:
            continue
        
        # Create prompt: context + question with choice assertion
        # Format: "{context}\n{choice} is the person who {question}"
        # Remove question mark and make it an assertion
        question_text = example['question'].strip()
        if question_text.endswith('?'):
            question_text = question_text[:-1].strip()
        
        # Create assertion: "{choice} is the person who {question}"
        prompt = f"{example['context']}\n{choice} is the person who {question_text}."
        
        # Label: 0 if correct answer (unbiased), 1 if wrong answer (biased)
        label = 0 if i == answer_key else 1
        
        expanded_examples.append({
            "prompt": prompt,
            "prompt_label": label,
            "category": example.get('category', ''),
            "example_id": example.get('example_id', -1),
            "choice_index": i,
        })
    
    return expanded_examples


def load_mbbq_category(category, data_dir=MBBQ_DATA_DIR):
    """Load MBBQ data for a specific category (English only)."""
    filepath = os.path.join(data_dir, f"{category}_en.jsonl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MBBQ data file not found: {filepath}")
    
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return examples


def convert_all_categories(categories=None, data_dir=MBBQ_DATA_DIR):
    """
    Convert all MBBQ categories to BiasGuard format.
    
    Returns:
        List of all BiasGuard-formatted examples
    """
    if categories is None:
        from config_mbbq import CATEGORIES
        categories = CATEGORIES
    
    all_examples = []
    
    for category in categories:
        print(f"Processing {category}...")
        mbbq_examples = load_mbbq_category(category, data_dir)
        
        category_count = 0
        for example in mbbq_examples:
            if example['context_condition'] == "ambig":
                continue
            biasguard_examples = convert_mbbq_to_biasguard_format(example)
            all_examples.extend(biasguard_examples)
            category_count += len(biasguard_examples)
        
        print(f"  Converted {len([e for e in mbbq_examples if e['context_condition'] != 'ambig'])} MBBQ examples to {category_count} BiasGuard examples (unknown answers excluded)")
    
    print(f"\nTotal: {len(all_examples)} BiasGuard examples")
    return all_examples


if __name__ == "__main__":
    # Test conversion
    print("Converting MBBQ to BiasGuard format...")
    examples = convert_all_categories()
    
    # Show sample
    print("\nSample converted example:")
    print(json.dumps(examples[0], indent=2))
    
    # Save to file for inspection
    output_file = "mbbq_biasguard_format.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\nSaved {len(examples)} examples to {output_file}")

