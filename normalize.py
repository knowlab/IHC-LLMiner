import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher

from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

def normalize_dataframe(df):
    # Normalize: strip whitespace, lowercase content
    return df.applymap(lambda x: str(x).strip().lower()).sort_values(by=df.columns.tolist()).reset_index(drop=True)

def parse_markdown_table(md_table):
    lines = md_table.strip().split('\n')
    headers = [h.strip().lower() for h in lines[0].strip('|').split('|')]
    # Check if the second line is a separator
    if len(lines) > 1 and all(char in "- |" for char in lines[1]):
        data_start_index = 2  # Skip the separator line
    else:
        data_start_index = 1  # No separator line, data starts immediately
    
    rows = [
        dict(zip(headers, [cell.strip() for cell in row.strip('|').split('|')]))
        for row in lines[data_start_index:]
    ]
    return normalize_dataframe(pd.DataFrame(rows))#.to_dict()#orient='records')

class UMLSNormalizer:
    def __init__(self, model_path, bs=512, device='cuda:0'):
        self.bs = bs
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda(device)

    def load_umls_mappings(self, mapping_file):
        with open(mapping_file) as f:
            data = f.readlines()

        self.umls_id_pairs = [tuple(line.strip().split('||')[1::-1]) for line in tqdm(data)]
        self.all_names = [p[0] for p in self.umls_id_pairs]
        self.all_ids = [p[1] for p in self.umls_id_pairs]
        self.all_reps_emb = self._embed_texts(self.all_names)

    def _embed_texts(self, texts):
        all_reps = []
        for i in tqdm(range(0, len(texts), self.bs)):
            toks = self.tokenizer.batch_encode_plus(
                texts[i:i+self.bs],
                padding="max_length", max_length=32, truncation=True, return_tensors="pt"
            )
            toks_cuda = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                output = self.model(**toks_cuda)
            cls_rep = output[0][:, 0, :].cpu().numpy()
            all_reps.append(cls_rep)
        return np.concatenate(all_reps, axis=0)

    def normalize_column_names(self, table_columns):
        self.umls_mapping = {}
        for i in tqdm(range(0, len(table_columns), self.bs)):
            toks = self.tokenizer.batch_encode_plus(
                table_columns[i:i+self.bs],
                padding="max_length", max_length=32, truncation=True, return_tensors="pt"
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                cls_rep = self.model(**toks)
            cls_rep = cls_rep[0][:, 0, :].cpu()

            all_reps_emb_tensor = torch.tensor(self.all_reps_emb)
            distances = torch.cdist(cls_rep.unsqueeze(0), all_reps_emb_tensor)[0]
            nearest_indices = torch.argmin(distances, dim=-1)
            del toks
            del cls_rep
            del all_reps_emb_tensor
            del distances
            for name, idx in zip(table_columns[i:i+self.bs], nearest_indices):
                self.umls_mapping[name] = self.umls_id_pairs[idx]
            del nearest_indices

    def normalize_table(self, df):
        new_rows = []
        for row in df.to_dict(orient='records'):
            new_row = {}
            for k, v in row.items():
                if not k or not v:
                    continue
                if k not in ['tumor type', 'tumor site']:
                    if k in self.umls_mapping:
                        new_row[self.umls_mapping[k][0]] = v
                else:
                    if v in self.umls_mapping:
                        new_row[k] = self.umls_mapping[v][0]
            new_rows.append(new_row)
        return pd.DataFrame(new_rows)

    def save_output(self, normalized_tables, output_file):
        all_data = pd.concat(normalized_tables, ignore_index=True)
        all_data.replace('', pd.NA, inplace=True)
        all_data.dropna(axis=1, how='all', inplace=True)
        all_data = all_data[all_data['tumor site'].fillna('').str.lower() != 'na']
        all_data = all_data[all_data['tumor site'].fillna('').str.lower() != 'sodium']
        all_data.to_csv(output_file, sep='\t', index=False)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='knowlab-research/IHC-LLMiner-align-GatorTronS', help="Path to SapBERT model")
    parser.add_argument("--mapping_file", type=str, default='evaluation_file_umls2025aa.txt', help="UMLS concept file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to TSV input file with abstracts extraction results")
    parser.add_argument("--output_file", type=str, default="inference_umls_mapped_data.tsv")
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    normalizer = UMLSNormalizer(model_path=args.model_path, device=args.device)
    normalizer.load_umls_mappings(args.mapping_file)

    # Load and parse all tables
    tables=pd.read_csv(args.input_file, sep='\t')
    tables=tables.dropna()
    tables=tables.output.apply(parse_markdown_table)
    
    # Extract all unique column names (for normalization)
    column_set = set()
    for table in tables:
        column_set.update(table.columns)
        try:
            column_set.update(table['tumor site'].tolist())
            column_set.update(table['tumor type'].tolist())
        except:
            pass
    column_set = [c for c in column_set if c and not c.replace('ihc marker', '').strip().isnumeric()]
    normalizer.normalize_column_names(sorted(set(column_set)))

    # Normalize each table and append PMIDs
    normalized_tables=[]
    for idx, table in tqdm(enumerate(tables),total=len(tables)):
        normalized_tables.append(pd.DataFrame(normalizer.normalize_table(table)))

    normalizer.save_output(normalized_tables, args.output_file)
    print(f"Saved normalized output to {args.output_file}")

if __name__ == "__main__":
    main()
