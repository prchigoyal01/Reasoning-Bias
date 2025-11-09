import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_mbbq_dataset(
    data_dir: str = "MBBQ_data",
    language: str = None,
    category: str = None
) -> List[Dict[str, Any]]:
    """
    Load MBBQ dataset from local JSONL files.
    
    Args:
        data_dir: Directory containing MBBQ JSONL files
        language: Optional language filter (eg en, es, nl, tr)
        category: Optional category filter (eg Age, Gender_identity)
        
    Returns:
        List of dictionaries containing the dataset
    """
    data_dir = os.path.abspath(data_dir)
    records = []

    if language is None or category is None:
        raise ValueError("Provide language and category to dataloader")

    fname = f"{category}_{language}.jsonl"

    fpath = os.path.join(data_dir, fname)
    try:
        records = load_jsonl(str(fpath))
    except Exception as e:
        print(f"Could not read {fpath}: {e}")
        
    return records