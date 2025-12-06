PROMPT_TEMPLATE_PATH = "prompt_templates/original.json"
BIAS_TYPES_PATH = "prompt_templates/mbbq_bias_types.json"
STANDARDS_PATH = "prompt_templates/standards.json"

TEACHER_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BATCH_SIZE = 10

# MBBQ data paths
MBBQ_DATA_DIR = "../MBBQ_data"
RESULTS_DIR = "../src/results_new"
MBBQ_BIASGUARD_PATH = "mbbq_biasguard_format.jsonl"

# Categories to use (without _control suffix)
CATEGORIES = ['Disability_status', 'Physical_appearance', 'SES', 'Sexual_orientation']

# Output paths
SFT_DATA_PATH = "mbbq_sft_data.jsonl"
SFT_MODEL_PATH = "mbbq_sft_model"

RL_DATA_PATH = "mbbq_rl_data.jsonl"
RL_MODEL_PATH = "mbbq_rl_model"

