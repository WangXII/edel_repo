import json
import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from datasets import Dataset, load_from_disk
from deprecated import deprecated
from tqdm import tqdm

import models.margin_config as margin_class
from models.transformers import InputExample
from po_datasets.create_bm25_examples import (
    get_count_dict,
)
from po_datasets.dataset import DatasetExamples
from utils.utils import process_pubmed_xmls

random.seed(42)


class Retriever(ABC):
    def __init__(
        self,
        examples: DatasetExamples,
        synonyms_in_query: bool = False,
        use_non_matching_texts: bool = True,
        use_supervised_examples: bool = True,
        use_distant_bm_25_examples: bool = False,
        bm25_k: int = 10,
        bm25_repeat_seen_pmids: bool = True,
        margin_config: margin_class.MarginConfig = margin_class.margin_classes_v3,
        use_batch_negatives: bool = False,
        use_random_negatives: bool = False,
        cache_file_prefix: str = "all_po_datasets_synonyms",
        cache_dir: str = "edel_repo_cache/retriever_examples",
        cache: bool = True,
        use_previous_faiss_examples: bool = False,
        include_positive_faiss_examples=False,
        faiss_examples_json: List[str] = [
            "data/retriever_results/faiss_example_genes_found_pmid.json"
        ],
        add_beir_datasets: bool = False,
        only_beir_datasets: bool = False,
        beir_dataset_files: List[str] = ["BioASQ-training12b/training12b_new.json"],
        max_ratio_negatives: int = 20,
        max_negatives: int = 50,
        all_negatives: bool = False
    ):
        # Create input examples
        # TODO: These are not really batch negatives, but rather random negatives
        self.examples = examples
        self.mode = self.examples.mode
        self.synonyms_in_query = synonyms_in_query
        self.use_non_matching_texts = use_non_matching_texts  # Only used with synonyms
        self.use_supervised_examples = use_supervised_examples
        self.use_distant_bm_25_examples = use_distant_bm_25_examples
        self.use_batch_negatives = use_batch_negatives
        self.use_random_negatives = use_random_negatives
        self.max_ratio_negatives = max_ratio_negatives
        self.max_negatives = max_negatives
        self.all_negatives = all_negatives
        self.bm25_k = bm25_k
        self.bm25_repeat_seen_pmids = bm25_repeat_seen_pmids
        self.cache = cache
        self.cache_dir = cache_dir
        self.cache_file_prefix = cache_file_prefix

        self.train = None
        self.dev = None
        self.test = None
        self.all_pmids_texts = None

        self.query_id_dict = {}
        self.current_query_id = -1

        # if self.use_batch_negatives:
        #     self.max_ratio_negatives = 10
        #     self.use_non_matching_texts = False

        self.use_previous_faiss_examples = use_previous_faiss_examples
        self.previous_faiss_examples = {"mapping": {}, "pmid_texts": {}}
        if self.use_previous_faiss_examples:
            for example_json in faiss_examples_json:
                assert Path(example_json).exists(), f"{example_json} does not exist"
                # Add all examples to the same dictionary
                with open(example_json, "r") as f:
                    current_dict = json.load(f)
                    self.add_previous_faiss_examples(current_dict)

        self.include_positive_faiss_examples = include_positive_faiss_examples

        self.add_beir_datasets = add_beir_datasets
        self.only_beir_datasets = only_beir_datasets
        self.beir_dataset_files = beir_dataset_files
        if self.add_beir_datasets:
            self.pubmed_cache = self.load_pubmed_cache()
            self.pl_dataframe = pl.from_arrow(self.pubmed_cache.data.table)
            print("Polars Dataframe loaded")

        self.bm25_examples_count_dict = get_count_dict(
            margin_config.bm25_margin_values_tuple
        )
        self.previous_faiss_examples_count_dict = {
            "genes_found": 0,
            "genes_not_found": 0,
        }

        self.non_matching_positive_label = 1  # 1 is positive example, 0 is negative

        self.margin_class = margin_config
        self.margin_value_dict = margin_config.margin_value_dict

        # assert (
        #     self.margin_class.name in cache_file_prefix or not cache
        # ), "Margin class should be included in cache_file_prefix"

        # self.tmp_positive_dict = {}

    def generate_examples(self):
        cache_file_base = self.cache_dir + "/" + self.cache_file_prefix
        train_cache_file = cache_file_base + "_train.pkl"

        if self.cache and Path(train_cache_file).exists():
            # Load using pickle
            self.train = pickle.load(open(cache_file_base + "_train.pkl", "rb"))
            self.dev = pickle.load(open(cache_file_base + "_dev.pkl", "rb"))
            self.test = pickle.load(open(cache_file_base + "_test.pkl", "rb"))

            # Load the count of positive and negative examples using a simple text file
            with open(cache_file_base + "_stats.json", "r") as f:
                self.examples_count_dict = json.load(f)
            try:
                with open(cache_file_base + "_bm25_stats.json", "r") as f:
                    self.bm25_examples_count_dict = json.load(f)
            except FileNotFoundError:
                self.bm25_examples_count_dict = get_count_dict(
                    self.margin_class.bm25_margin_values_tuple
                )
        else:
            self.train = self.sample_generator_for_retriever(
                self.examples.train, self.examples.train_split, "train"  # type: ignore
            )
            self.dev = self.sample_generator_for_retriever(
                self.examples.dev, self.examples.dev_split, "dev"  # type: ignore
            )
            self.test = self.sample_generator_for_retriever(
                self.examples.test, self.examples.test_split, "test"  # type: ignore
            )
            if self.cache:
                # Save using pickle
                pickle.dump(self.train, open(cache_file_base + "_train.pkl", "wb"))
                pickle.dump(self.dev, open(cache_file_base + "_dev.pkl", "wb"))
                pickle.dump(self.test, open(cache_file_base + "_test.pkl", "wb"))

                # Save the count of positive and negative examples using a simple text file
                with open(cache_file_base + "_stats.json", "w") as f:
                    json.dump(self.examples_count_dict, f)
                with open(cache_file_base + "_bm25_stats.json", "w") as f:
                    json.dump(self.bm25_examples_count_dict, f)

    @abstractmethod
    def sample_generator_for_retriever(
        self,
        dataset: Dataset,
        # split: Union[np.ndarray[Any, Any], List[int]],  # Only with Python >= 3.9
        split: Union[np.ndarray, List[int]],
        split_name: str,
    ) -> Tuple[List[InputExample], List[List[InputExample]]]:
        pass

    @abstractmethod
    def add_previous_faiss_examples(self, example_dict: Dict):
        pass

    def add_pubmed_documents(self):
        pubmed_cache = self.load_pubmed_cache()
        self.pl_dataframe = pl.from_arrow(pubmed_cache.data.table)
        print("Polars Dataframe loaded")

        self.all_pmids_texts = self.get_texts_from_pmids(None)

    def load_pubmed_cache(
        self, pubmed_cache_file: str = "edel_repo_cache/datasets/pubmed.dataset"
    ):
        if Path(pubmed_cache_file).exists():
            print("Loading dataset from local cache")
            pubmed_cache = load_from_disk(pubmed_cache_file)
            print("Huggingface Dataset loaded")
        else:
            shards = [
                f"edel_repo_cache/treatment_explorer/pubmed/pubmed23n{i:04d}.xml"
                for i in range(1, 1167)
            ]
            pubmed_cache = Dataset.from_generator(
                process_pubmed_xmls,
                gen_kwargs={"shards": shards},
                num_proc=16,
            )
            pubmed_cache.save_to_disk(pubmed_cache_file)

        return pubmed_cache

    def get_texts_from_pmids(self, pmids: List[str]) -> pl.DataFrame:
        if pmids is not None:
            filtered_df = self.pl_dataframe.filter(self.pl_dataframe["pmid"].is_in(pmids))
        else:
            filtered_df = self.pl_dataframe
        text_column = (
            pl.when((pl.col("title") != "") & (pl.col("abstract") != ""))
            .then(pl.col("title") + ". " + pl.col("abstract"))
            .when(pl.col("title") == "")
            .then(pl.col("abstract"))
            .otherwise(pl.col("title"))
            .alias("text")
        )
        filtered_df = filtered_df.with_columns(text_column)

        return filtered_df

    def get_beir_examples(self) -> List[InputExample]:

        # TODO add other datasets than BioASQ
        bioasq_data = {}
        bioasq_paths = self.beir_dataset_files
        for path in bioasq_paths:
            with open(path, "r") as f:
                data = json.load(f)
                bioasq_data.update(data)

        self.examples_count_dict["count_positives_bioasq"] = 0
        self.examples_count_dict["count_negatives_bioasq"] = 0

        input_examples = []
        bioasq_dict = {}
        all_pmids = set()

        # First pass through all the data
        for question in tqdm(bioasq_data["questions"], desc="Reading BioASQ data"):
            question_text = question["body"]
            for document in question["documents"]:
                bioasq_dict[question_text] = bioasq_dict.get(question_text, [])
                pmid = document.split("/")[-1]
                bioasq_dict[question_text].append(pmid)
                all_pmids.add(pmid)

        all_pmids_texts = self.get_texts_from_pmids(all_pmids)

        # Second pass through all the data
        for question, pmids in tqdm(
            bioasq_dict.items(), desc="Creating examples for BioASQ"
        ):
            for pmid in pmids:
                if all_pmids_texts["pmid"].is_in([pmid]).any():
                    input_examples.append(
                        InputExample(
                            texts=[
                                question,
                                all_pmids_texts.filter(pl.col("pmid") == pmid)[0][
                                    "text"
                                ][0],
                            ],
                            label=1,
                            margin=self.margin_value_dict["positive"],
                            doc_id=int(pmid),
                        )
                    )
                    self.examples_count_dict["count_positives_bioasq"] += 1

            # Sample negative examples from other pmids
            # Make sure that sample is different for each question
            sample_pmids = random.sample(all_pmids_texts["pmid"].to_list(), 10)
            for pmid in sample_pmids:
                if pmid not in pmids and all_pmids_texts["pmid"].is_in([pmid]).any():
                    input_examples.append(
                        InputExample(
                            texts=[
                                question,
                                all_pmids_texts.filter(pl.col("pmid") == pmid)[0][
                                    "text"
                                ][0],
                            ],
                            label=0,
                            margin=self.margin_value_dict[
                                "negative_previous_faiss_examples"
                            ],
                            doc_id=int(pmid),
                        )
                    )
                    self.examples_count_dict["count_negatives_bioasq"] += 1

        return input_examples

    def get_random_negatives(
        self,
        input_examples: list[InputExample],
        margin: float = 0.0,
        max_number: float = np.inf,
        seed: int = 42,
    ) -> list[InputExample]:

        # Create a dict with all query_ids and their corresponding doc_ids
        query_id_dict = {}
        query_text_dict = {}
        doc_text_dict = {}
        doc_ids = set()

        for example in input_examples:
            if example.query_id in query_id_dict:
                query_id_dict[example.query_id].append(example.doc_id)
            else:
                query_id_dict[example.query_id] = [example.doc_id]
            query_text_dict[example.query_id] = example.texts[0]
            doc_text_dict[example.doc_id] = example.texts[1]
            doc_ids.add(example.doc_id)

        random_negatives = []

        for query_id, query_doc_ids in query_id_dict.items():
            current_seed = seed + query_id
            rng = random.Random(current_seed)
            valid_doc_ids = doc_ids - set(query_doc_ids)
            valid_doc_ids = sorted(list(valid_doc_ids))
            sample_size = min(max_number, len(valid_doc_ids))
            doc_ids_sample = rng.sample(valid_doc_ids, sample_size)
            for doc_id in doc_ids_sample:
                random_negatives.append(
                    InputExample(
                        texts=[query_text_dict[query_id], doc_text_dict[doc_id]],
                        label=0,
                        margin=self.margin_class.margin_fn(margin),
                        query_id=query_id,
                        doc_id=doc_id,
                    )
                )
                self.examples_count_dict["count_negative_random"] += 1

        input_examples.extend(random_negatives)
        # Shuffle the input examples
        input_examples = random.Random(seed).sample(input_examples, len(input_examples))

        return input_examples

    @deprecated
    def sample_generator_batch_negatives(
        self,
        dataset: Dataset,
        # split: Union[np.ndarray[Any, Any], List[int]],  # Only with Python >= 3.9
        split: Union[np.ndarray, List[int]],
    ) -> Tuple[List[InputExample]]:
        input_examples = []

        for eg_id in tqdm(split, desc="Generating examples"):
            eg_dataset = dataset.filter(
                lambda x: x["entrez_id"] == eg_id and len(x["drugs"]) > 0
            )
            if len(eg_dataset) > 0 and self.use_supervised_examples:
                positive_examples, _ = self.get_examples(
                    eg_dataset,
                    col_names=[],
                    label=1,
                    margin=self.margin_value_dict["positive"],
                    seed=eg_id,
                )

                input_examples.extend(positive_examples)
                count_pos = len(positive_examples)
                self.examples_count_dict["count_positive"] += count_pos

                # Get negative examples for other genes
                other_eg_dataset = dataset.filter(lambda x: x["entrez_id"] != eg_id)
                batch_negative_examples, _ = self.get_examples(
                    other_eg_dataset,
                    col_names=[],
                    label=0,
                    margin=self.margin_value_dict["negative_other_gene"],
                    seed=eg_id,
                )
                if len(batch_negative_examples) > count_pos * self.max_ratio_negatives:
                    batch_negative_examples = random.sample(
                        batch_negative_examples,
                        count_pos * self.max_ratio_negatives,
                    )
                input_examples.extend(batch_negative_examples)
                self.examples_count_dict["count_negative_other_gene"] += len(
                    batch_negative_examples
                )

        return input_examples
