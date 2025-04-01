from setuptools import setup, find_packages

setup(
    name='ihc-llminer',
    version='0.1.0',
    author='Yunsoo Kim',
    author_email='yunsoo.kim.23@ucl.ac.uk',
    description='IHC-LLMiner: A pipeline for extracting immunohistochemistry profiles from PubMed abstracts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/knowlab/IHC-LLMiner',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'tqdm',
        'pandas',
        'beautifulsoup4',
        'requests',
        'langdetect',
        'scikit-learn',
        'scipy',
        'lxml',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)