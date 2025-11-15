"""Evaluator for the RedditBias dataset (flexible loader + HF model support).

Usage (examples):
  python reddit_bias.py \
	--model facebook/bart-large-mnli \
	--dataset path/to/reddit_bias.jsonl \
	--text-field text --label-field label \
	--task classification \
	--of-labels other

The script supports two modes:
 - classification: uses a Hugging Face text-classification pipeline (expects model to output labels)
 - generation: uses a text-generation pipeline and maps generated text to labels using a simple string match

Metrics produced:
 - accuracy: fraction of predictions equal to gold labels
 - OF score: fraction of predictions that are considered 'other-favoring' (labels passed via --of-labels)

Notes / assumptions:
 - If the dataset is a local JSONL file, it should have one JSON object per line. Use `--text-field` and `--label-field` to map fields.
 - For generation mode you may need to provide a mapping from generated answers to label names with `--map`.
 - This is a flexible evaluator; if you want a precise reproduction of the RedditBias repo metrics, tell me how that repo defines OF and I will adapt.
"""

import json
import argparse
import os
from typing import List, Dict, Any, Optional

try:
	from datasets import load_dataset
except Exception:
	load_dataset = None

try:
	from transformers import pipeline, AutoTokenizer
except Exception:
	pipeline = None


def load_data(path_or_id: str, text_field: str, label_field: str, max_examples: Optional[int] = None):
	"""Load dataset from a local file (json/jsonl/csv) or a HuggingFace dataset id.
	Returns list of dicts with keys 'text' and 'label'."""
	# Local file
	if os.path.exists(path_or_id):
		# try to use datasets if available
		if load_dataset:
			# choose json or csv by extension
			ext = os.path.splitext(path_or_id)[1].lower()
			if ext in ('.json', '.jsonl'):
				ds = load_dataset('json', data_files=path_or_id, split='train')
			elif ext in ('.csv', '.tsv'):
				ds = load_dataset('csv', data_files=path_or_id, split='train')
			else:
				# fallback to manual load
				items = []
				with open(path_or_id, 'r', encoding='utf-8') as f:
					for i, line in enumerate(f):
						if max_examples and i >= max_examples:
							break
						items.append(json.loads(line))
				return [{'text': it[text_field], 'label': it[label_field]} for it in items]

			items = []
			for i, ex in enumerate(ds):
				if max_examples and i >= max_examples:
					break
				items.append({'text': ex[text_field], 'label': ex[label_field]})
			return items
		else:
			items = []
			with open(path_or_id, 'r', encoding='utf-8') as f:
				for i, line in enumerate(f):
					if max_examples and i >= max_examples:
						break
					items.append(json.loads(line))
			return [{'text': it[text_field], 'label': it[label_field]} for it in items]

	# HuggingFace dataset id
	if load_dataset:
		ds = load_dataset(path_or_id, split='train')
		items = []
		for i, ex in enumerate(ds):
			if max_examples and i >= max_examples:
				break
			items.append({'text': ex[text_field], 'label': ex[label_field]})
		return items

	raise RuntimeError("datasets library not available and local file not found: %s" % path_or_id)


def build_pipeline(model_name: str, task: str, device: int = -1):
	if pipeline is None:
		raise RuntimeError("transformers library is required to run evaluation (install 'transformers').")
	if task == 'classification':
		return pipeline('text-classification', model=model_name, return_all_scores=False, device=device)
	elif task == 'generation':
		# generation pipeline
		return pipeline('text-generation', model=model_name, device=device)
	else:
		raise ValueError('Unsupported task: %s' % task)


def predict_batch_clf(pipe, examples: List[Dict[str, Any]], text_field='text'):
	preds = []
	for ex in examples:
		out = pipe(ex[text_field], truncation=True)
		# pipeline returns list of dicts (label, score)
		if isinstance(out, list) and len(out) > 0 and 'label' in out[0]:
			preds.append(out[0]['label'])
		else:
			preds.append(str(out))
	return preds


def predict_batch_gen(pipe, examples: List[Dict[str, Any]], prompt_template: Optional[str] = None):
	preds = []
	for ex in examples:
		prompt = prompt_template.format(text=ex['text']) if prompt_template else ex['text'] + '\nAnswer:'
		out = pipe(prompt, max_new_tokens=64, do_sample=False)
		if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
			gen = out[0]['generated_text']
		else:
			# fallback
			gen = str(out)
		preds.append(gen.strip())
	return preds


def compute_metrics(preds: List[str], golds: List[str], of_labels: List[str]):
	total = len(golds)
	correct = sum(1 for p, g in zip(preds, golds) if p == g)
	accuracy = correct / total if total else 0.0
	of_count = sum(1 for p in preds if p in of_labels)
	of_score = of_count / total if total else 0.0
	return {'accuracy': accuracy, 'of_score': of_score, 'total': total}


def main():
	parser = argparse.ArgumentParser(description='Evaluate a model on RedditBias-like datasets')
	parser.add_argument('--model', required=True, help='HuggingFace model id or local path')
	parser.add_argument('--dataset', required=True, help='Local path to dataset (jsonl/csv) or HF dataset id')
	parser.add_argument('--text-field', default='text', help='Field name for input text')
	parser.add_argument('--label-field', default='label', help='Field name for gold label')
	parser.add_argument('--task', choices=['classification', 'generation'], default='classification')
	parser.add_argument('--of-labels', default='other', help='Comma-separated labels considered "other-favoring" (OF)')
	parser.add_argument('--max-examples', type=int, default=None, help='Limit number of examples for a quick run')
	parser.add_argument('--device', type=int, default=-1, help='Device for pipelines (use -1 for CPU)')
	parser.add_argument('--prompt-template', default=None, help='Template for generation prompts, use {text} to insert example text')
	args = parser.parse_args()

	print('Loading data...')
	examples = load_data(args.dataset, args.text_field, args.label_field, max_examples=args.max_examples)
	print(f'Loaded {len(examples)} examples')

	print('Building model pipeline...')
	pipe = build_pipeline(args.model, args.task, device=args.device)

	print('Running predictions...')
	if args.task == 'classification':
		preds = predict_batch_clf(pipe, examples, text_field='text')
	else:
		preds = predict_batch_gen(pipe, examples, prompt_template=args.prompt_template)

	golds = [ex['label'] for ex in examples]
	of_labels = [s.strip() for s in args.of_labels.split(',') if s.strip()]

	metrics = compute_metrics(preds, golds, of_labels)
	print('\nResults:')
	print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
	main()

