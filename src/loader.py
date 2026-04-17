import pandas as pd
import json
from typing import List, Dict

def load_corpus(jsonl_path: str) -> List[Dict]:
    docs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                docs.append({"doc_id": data["doc_id"], "text": data["text"]})
    return docs

def load_case1(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["reference_doc_ids"] = df["reference_doc_ids"].str.split("|")
    return df

def load_case2(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["reference_doc_ids"] = df["reference_doc_ids"].str.split("|")
    return df