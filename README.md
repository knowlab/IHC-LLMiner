# IHC-LLMiner

[**Paper**]() | 
[**🤗 Model and Dataset**](https://huggingface.co/collections/knowlab-research/ihc-llminer-67ebc27792449023e7123f2d) | 

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
python download.py --markers BOB1 TTF1 --max_per_marker 9999 --output_file pmid_list_w_abstract.tsv
```

## Classify Abstracts
```bash
python classify.py \
  --input_file pmid_list_w_abstract.tsv \
  --output_file predictions.json
```

## Extract IHC Profiles
```bash
python extract.py \
  --input_file predictions.json \
  --output_file extraction_result.tsv
```

## Preparation of the UMLS file
> You would need UMLS metathesaurus downloaded. For this, you would need to log in with your own credential.
> Download MRCONSO.RRF from [here](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
> then run generate_UMLS_data.ipynb

## Normalise the Extracted Results
```bash
python normalize.py \
  --input_file extraction_result.tsv \
  --output_file inference_umls_mapped_data.tsv
```

## Example for downstream analysis of the normalised results
> Please refer to data_analysis.ipynb

## Hardware
The code was tested with A5000 GPU 24GB memory.

## Reference
```bibtex
```
