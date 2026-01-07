# !conda activate colbert

import argparse
import datetime
import json
import sys

# import faiss.contrib.torch_utils  # use if you want to use PyTorch tensors
import time
from pathlib import Path

sys.path.insert(0, "edel_repo_cache/ColBERT/")

from typing import Dict, Generator, List, Tuple

import faiss
import lxml.etree as etree
import numpy as np
import numpy.typing as npt
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from utils.utils import process_pubmed_xmls

if __name__ == "__main__":
    # If dataset exists in local cache, load it from there, otherwise create it
    dataset_file = "edel_repo_cache/datasets/pubmed.dataset"
    if Path(dataset_file).exists():
        print("Loading dataset from local cache")
        dataset = load_from_disk(dataset_file)
        print("Dataset loaded")
    else:
        shards = [
            f"edel_repo_cache/pubmed/pubmed23n{i:04d}.xml"
            for i in range(1, 1167)
        ]
        dataset = Dataset.from_generator(
            process_pubmed_xmls,
            gen_kwargs={"shards": shards},
            num_proc=16,
        )
        dataset.save_to_disk(dataset_file)

    dataset_docs = [
        # "PMID: " + x["pmid"] + ". " + x["title"] + ". " + x["abstract"]
        # for x in dataset
        # TODO: Change when re-indexing
        # PMID is actually not necessary. ColBERT will index the documents by their order in the dataset.
        x["title"] + "[SEP]" + x["abstract"]
        for x in dataset
    ]

    # Index with pretrained ColBERT model into FAISS
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 384  # truncate passages at 384 tokens

    index_name = f"colbert_v2.pubmed.{doc_maxlen}len.{nbits}bits"
    checkpoint = "edel_repo_cache/pretrained_llm/colbertv2.0"
    with Run().context(
        RunConfig(
            nranks=4,  # nranks specifies the number of processes to use (might be greater than the number of GPUs?, does not seem true actually)
            index_root="edel_repo_cache/colbert_2.0_index/",
        )
    ):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4
        )  # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
        # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=dataset_docs, overwrite=True)
