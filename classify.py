import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline
import json
import argparse

class AbstractClassifier:
    """
    A class to classify PubMed abstracts using a fine-tuned Gemma-2 model.
    """

    def __init__(self, model_path, device='cuda:0', max_new_tokens=4, temperature=0):
        self.device = device
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        self.parameters = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": temperature
        }


        self.system_prompt="Act as a histopathology expert."
        self.user_prompt = """The following is an abstract of an abstract of a biomedical paper. Your task is to classify whether the input abstract meets the following criteria:
            1. It reports the positivity rate of one or more immunohistochemical markers in one or more tumor types. Case reports describing immunohistochemical findings in a single patient, even without exact numbers, are acceptable.
            2. If not a single-patient case report, it provides the exact number of patients that are positive/negative for each marker in each tumor type.
            3. It is not a review article or a meta-analysis.
        If the article meets all these criteria, output 'Include'. Otherwise, output 'Exclude'.
        \nAbstract: """     
    
    def classify(self, abstract, system_prompt, user_prompt):
        """
        Classify a single abstract as 'include' or 'exclude'.
        """   
        full_prompt = f"<|user|>\n{system_prompt}\n{user_prompt}\n{abstract}<|end|>\n<|assistant|>"
        response = self.pipe(full_prompt, **self.parameters)[0]['generated_text']
        return 'include' if 'include' in response.lower() else 'exclude'

    def run(self, input_path, output_path):
        """
        Run classification on a dataset and save results.
        """
        df = pd.read_csv(input_path, sep='\t')
        results = []
        
        print(f"Running classification on {len(df)} abstracts...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            abstract = row['abstract']
            label = self.classify(abstract, self.system_prompt, self.user_prompt)
            results.append(label)

        df['prediction'] = results
        with open(output_path, 'w') as f:
            f.write(json.dumps(df.to_dict(orient='records')))
        print(f"Classification complete. Results saved to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description='Classify IHC abstracts using a fine-tuned Gemma-2 model.')
    parser.add_argument('--model_path', default='knowlab-research/IHC-LLMiner-CLS', help='Path to the fine-tuned Gemma-2 model.')
    parser.add_argument('--input_file', required=True, help='TSV file with a column named "abstract".')
    parser.add_argument('--output_file', default='classified_abstracts.json', help='Output JSON file with predictions.')
    parser.add_argument('--device', default='cuda:0', help='GPU device.')

    args = parser.parse_args()

    classifier = AbstractClassifier(model_path=args.model_path, 
        device=args.device)
    classifier.run(
        input_path=args.input_file,
        output_path=args.output_file
    )


if __name__ == '__main__':
    main()
