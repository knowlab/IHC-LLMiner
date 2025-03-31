import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

class IHCInformationExtractor:
    def __init__(self, model_path, device='cuda:0', max_new_tokens=1024, temperature=0):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.system_prompt = "Act as a histopathology expert."
        self.user_prompt = (
            "Examine the following scientific abstract carefully. "
            "Print a 'markdown' table with a row for each different tumor type tested and a column for each immunohistochemical marker examined "
            "(even if there is only one marker and one tumour type). For each tumor, specify the tumour type and tumor site (if given, "
            "if not state NA) separately. For each result (for each tumor type and immunohistochemical marker), report the data in the form X/Y "
            "where X are the number of positive cases and Y are the total number of cases tested. "
            "If percentages are given and it is possible to calculate absolute numbers, do so. "
            "Sometimes results may be given for a group of tumors together – only report the results for individual counts. "
            "If the distribution is unclear, mark as NA. Include only IHC findings. "
            "Minimize additional comments. Do not extrapolate or assume results.\n\nAbstract: "
        )
        self._load_pipeline()

    def _load_pipeline(self):
        is_4bit = "medium" in self.model_path
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
            device=self.device,
            trust_remote_code=True,
            load_in_4bit=is_4bit if is_4bit else False,
        )
        self.parameters = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
            "temperature": self.temperature,
        }

    def load_data(self, input_file):
        with open(input_file) as f:
            data = json.loads(f.read())

        return [d for d in data['prediction'] if d.lower().strip()=='include']

    def extract(self, data):
        results = []
        print(f"Running extraction on {len(data)} abstracts...")
        for row in tqdm(data):
            abstract = row['abstract']
            final_prompt = f"<|user|>\n{self.system_prompt}\n{self.user_prompt}\n{abstract}<|end|>\n<|assistant|>"
            output = self.pipe(final_prompt, **self.parameters)[0]['generated_text']
            results.append({
                'pmid': row['pmid'],
                'url': row['url'],
                'abstract': abstract,
                'label': row.get('prediction', ''),
                'marker': row.get('marker', ''),
                'output': output or ''
            })
        return pd.DataFrame(results)

    def save_output(self, df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(self.model_path.rstrip("/"))
        filename = f"inference_{model_name}_{self.max_new_tokens}_{self.temperature}.tsv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IHC Information Extraction using a Generative LLM")
    parser.add_argument("--model_path", type=str, default='', help="Path to the fine-tuned LLM")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON input file with abstracts")
    parser.add_argument('--output', default='extracted_abstract.tsv', help='Output TSV file.')
    args = parser.parse_args()

    extractor = IHCInformationExtractor(
        model_path=args.model_path
    )

    data = extractor.load_data(args.input_json)
    df = extractor.extract(data)
    extractor.save_output(df, args.output_dir)


if __name__ == "__main__":
    main()
