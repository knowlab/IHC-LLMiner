# IHC-LLMiner

[**Paper**]() | 
[**🤗 Dataset**]() | 
[**🤗 Classification Model**]() |
[**🤗 Extraction Model**]()

## Description

IHC-LLMiner is a Python module for automatically extracting immunohistochemistry (IHC) marker-tumour profiles from PubMed abstracts. It leverages LLMs and BERT-based models for:
- Downloading abstracts for specific IHC markers
- Classifying abstract relevance
- Extracting structured IHC marker data
- Normalising entity mentions using UMLS

## Installation
Python 3.10
```bash
git clone https://github.com/knowlab/IHC-LLMiner.git
cd IHC-LLMiner
conda create -n ihcllminer python=3.10
conda activate ihcllminer
pip install .
```

## Download Abstracts
```bash
python download.py --markers ER PR CD34 --max_per_marker 9999 --max_per_marker --output pmid_list_w_abstract.tsv
```

## Classify Abstracts
```bash
python classify_abstracts.py \
  --input_file pmid_list_w_abstract.tsv \
  --output_file predictions.json
```

## Extract IHC Profiles
```bash
python extract_information.py \
  --input_json predictions.json \
  --output extract_result.tsv
```

## Normalise the Extracted Results
> You would need UMLS metathesaurus downloaded. For this, you would need to log in with your own credential.
> 
```bash
python normalize_umls.py \
  --model_path /path/to/sapbert_model \
  --mapping_file evaluation_file_umls2024ab.txt \
  --input_file extract_result.tsv \
  --output_file inference_umls_mapped_data.tsv
```

## Example for downstream analysis of the normalised results
> Please refer to example/data_analysis.ipynb

## Reference

```bibtex
```
