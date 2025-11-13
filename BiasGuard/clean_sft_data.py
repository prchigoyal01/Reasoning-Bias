
import json 
import matplotlib.pyplot as plt

with open("sft_data.jsonl", "r") as f:
    lines = f.readlines()

with open("sft_data_clean.jsonl", "w") as f:
    lens = []
    for line in lines:
        data = json.loads(line)
        if len(data["response"]) > len(data["prompt"]) + len(data["conclusion"]) + 50:
            f.write(line)