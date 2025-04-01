import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from langdetect import detect
import argparse
import warnings
warnings.filterwarnings("ignore")

class PubMedDownloader:
    """
    Class to download PMIDs and abstracts from PubMed based on IHC marker search terms.
    """

    def __init__(self, term_list, max_per_marker=20, output_file='pmid_list_w_abstract.tsv'):
        self.term_list = [term.upper().strip() for term in term_list]
        self.max_per_marker = max_per_marker
        self.output_file = output_file
        self.unique_pmids = set()
        self.results = []

    def get_pmids(self, term):
        if int(self.max_per_marker)<10000:
            r=requests.get(f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term} immunohisto*&RetMax={self.max_per_marker}')
            soup=BeautifulSoup(r.text)
            try:
                return [i.get_text(strip=True) for i in soup.find_all('id')]
            except:
                return []
        else:
            pmid_list=[]
            for i in range(0, int(self.max_per_marker), 9999):
                r=requests.get(f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term} immunohisto*&RetMax=9999&retstart={i}')
                soup=BeautifulSoup(r.text)
                try:
                    pmid_list=pmid_list+[i.get_text(strip=True) for i in soup.find_all('id')]
                except:
                    return []
            return pmid_list

    def get_abstract(self, pmid):
        """
        Retrieve the abstract text for a given PMID.
        """
        url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}'
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        try:
            return '\n'.join([tag.get_text(strip=True) for tag in soup.find_all('abstracttext')])
        except:
            return ""

    def run(self):
        """
        Run the full pipeline to retrieve PMIDs and extract relevant abstracts.
        """
        print("Downloading PMIDs and abstracts...")
        for term in tqdm(self.term_list, desc="Markers"):
            pmids = self.get_pmids(term)
            count = 0
            for pmid in pmids:
                if pmid in self.unique_pmids:
                    continue
                abstract = self.get_abstract(pmid)
                if not abstract:
                    continue
                try:
                    if detect(abstract) != 'en':
                        continue
                except:
                    continue
                self.results.append({
                    'marker': term,
                    'pmid': pmid,
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'abstract': abstract
                })
                self.unique_pmids.add(pmid)
                count += 1
                if count >= self.max_per_marker:
                    break

        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, sep='\t', index=False)
        print(f"Saved {len(df)} abstracts to {self.output_file}.")


def main():
    parser = argparse.ArgumentParser(description='Download PubMed abstracts for IHC markers.')
    parser.add_argument('--markers', nargs='+', required=True, help='List of IHC marker terms.')
    parser.add_argument('--max_per_marker', type=int, default=20, help='Max abstracts per marker.')
    parser.add_argument('--output_file', default='pmid_list_w_abstract.tsv', help='Output TSV file.')

    args = parser.parse_args()
    downloader = PubMedDownloader(term_list=args.markers,
                            max_per_marker=args.max_per_marker,
                            output_file=args.output_file)
    downloader.run()


if __name__ == "__main__":
    main()
