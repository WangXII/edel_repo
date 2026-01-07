import copy
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import elasticsearch
import numpy as np
import pandas as pd
import polars as pl
from datasets import Dataset
from datasets.utils import disable_progress_bar
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sqlitedict import SqliteDict

from utils.ncithesaurus import NCITheSaurusMapper

disable_progress_bar()


def filter_short_synonyms(synonyms: List[str], threshold: int = 2):
    """
    Filters out synonyms that are too short
    """
    synonyms = [synonym for synonym in synonyms if len(synonym) > threshold]
    if len(synonyms) == 0:
        synonyms = [""]
    return synonyms


def filter_full_name(name: str):
    """
    Filters out synonyms that are too short
    """
    if len(name) == 0:
        name = ""
    return name


class ElasticsearchHelper:
    es_index = "20231127_pubmed"
    es = Elasticsearch(
        ["https://localhost:9200"],
        request_timeout=30,
        max_retries=10,
        retry_on_timeout=True,
        basic_auth=("elastic", ""),
        verify_certs=True,
        ca_certs="edel_repo_cache/elasticsearch-8.11.1/config/certs/http_ca.crt",
    )

    @staticmethod
    def search(query: Dict[str, Any], size: int = 20):
        return ElasticsearchHelper.es.search(
            index=ElasticsearchHelper.es_index, body=query, size=size
        )

    @staticmethod
    def build_keyword_query(
        keyword_lists: List[List[str]],
        not_keyword_lists: List[List[str]],
        query_term: str = "match",
        datefilter: int = -1,
        retrieve_sentences: bool = False,
        retrieve_abstracts_only: bool = True,
    ) -> Dict[str, Any]:
        """Builds Elastic Search Queries from the provided arguments
        Parameters
        ----------
        args : dict of list
            List of list. For instance,
            [["PKB alpha", "RAC", "PKB", "Protein kinase B", "AKT1"]]
        datefilter: bool
            Filter documents of a later publication date than the one provided
        retrieve_sentences: bool
            If True, returns sentences instead of paragraphs
        retrieve_abstracts_only: bool
            If True, returns only abstracts

        Returns
        -------
        dict
            Returns HTTP Request Body for the corresponding ElasticSearch Query
        """

        # TODO: Re-enable if using pubmed2 index
        # if retrieve_sentences:  # Filter for sentences and not paragraphs
        #     predicate = "must_not"
        #     filter = [{"term": {"sentence_id": -1}}]
        # else:
        #     predicate = "filter"
        #     filter = [{"term": {"sentence_id": -1}}]
        # if retrieve_abstracts_only:
        #     filter.append({"term": {"paragraph_id": 0}})

        request_body: dict[str, Any] = {
            "query": {
                "bool": {"must": [], "filter": []}
            },  # TO BE ADDED FROM THE ARGUMENTS
        }

        argument: dict[str, Any] = {
            "bool": {"should": [], "must_not": [], "minimum_should_match": 1}
        }  # TO BE ADDED FROM THE ARGUMENTS

        for list_arg in keyword_lists:
            current_argument = copy.deepcopy(argument)
            for synonym in list_arg:
                # If there is shorter
                current_argument["bool"]["should"].append(
                    # {query_term: {"content": synonym.lower()}}
                    {query_term: {"text": synonym.lower()}}
                )
            request_body["query"]["bool"]["must"].append(current_argument)
        for list_arg in not_keyword_lists:
            current_argument = copy.deepcopy(argument)
            for synonym in list_arg:
                # If there is shorter
                current_argument["bool"]["must_not"].append(
                    # {query_term: {"content": synonym.lower()}}
                    {query_term: {"text": synonym.lower()}}
                )
            request_body["query"]["bool"]["filter"].append(current_argument)

        if datefilter > -1:
            request_body["query"]["bool"]["filter"] = []
            request_body["query"]["bool"]["filter"].append({"range": {}})
            request_body["query"]["bool"]["filter"][-1]["range"]["year"] = {}
            request_body["query"]["bool"]["filter"][-1]["range"]["year"][
                "gt"
            ] = datefilter

        # Get random documents if no keywords are provided
        if len(keyword_lists) == 0:
            function_score_query = {
                "query": {
                    "function_score": {
                        "query": {},
                        "functions": [{"random_score": {}}],
                        "boost_mode": "replace",
                    }
                }
            }
            function_score_query["query"]["function_score"]["query"] = request_body[
                "query"
            ]
            request_body = function_score_query
            # request_body = {"query": {"match_all": {}}}

        return request_body

    @classmethod
    def lexical_query(
        cls,
        query_text: str,
        query_term: str = "match",
        datefilter: int = -1,
        retrieve_sentences: bool = False,
        retrieve_abstracts_only: bool = True,
        number: int = 20,
    ) -> List[Dict[str, Any]]:
        """Queries Elastic Search with the provided arguments
        Parameters
        ----------
        args : dict of list
            List of list. For instance,
            [["PKB alpha", "RAC", "PKB", "Protein kinase B", "AKT1"]]
        datefilter: bool
            Filter documents of a later publication date than the one provided
        retrieve_sentences: bool
            If True, returns sentences instead of paragraphs
        retrieve_abstracts_only: bool
            If True, returns only abstracts
        number: int
            Number of results to return

        Returns
        -------
        list of dict
            Returns a list of dictionaries containing the results of the query
        """
        query = {
            "query": {
                query_term: {
                    "text": query_text.lower(),
                }
            }
        }
        # print(query)
        results = ElasticsearchHelper.search(query=query, size=number)["hits"]["hits"]
        # print(results)
        return results

    @staticmethod
    def explain_lexical_query(
        query_text: str,
        doc_id: int,
        query_term: str = "match",
        datefilter: int = -1,
        retrieve_sentences: bool = False,
        retrieve_abstracts_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Queries Elastic Search with the provided arguments
        Parameters
        ----------
        args : dict of list
            List of list. For instance,
            [["PKB alpha", "RAC", "PKB", "Protein kinase B", "AKT1"]]
        datefilter: bool
            Filter documents of a later publication date than the one provided
        retrieve_sentences: bool
            If True, returns sentences instead of paragraphs
        retrieve_abstracts_only: bool
            If True, returns only abstracts
        number: int
            Number of results to return

        Returns
        -------
        list of dict
            Returns a list of dictionaries containing the results of the query
        """
        query = {
            "query": {
                query_term: {
                    "text": query_text.lower(),
                }
            }
        }
        # print(query)
        try:
            result = ElasticsearchHelper.es.explain(
                index=ElasticsearchHelper.es_index, id=doc_id, body=query
            )["explanation"]["value"]
        except elasticsearch.NotFoundError:
            result = -1
        # print(result)
        return result

    @staticmethod
    def query_keywords(
        keyword_lists: List[List[str]],
        not_keyword_lists: List[List[str]] = [],
        query_term: str = "match",
        datefilter: int = -1,
        retrieve_sentences: bool = False,
        retrieve_abstracts_only: bool = True,
        number: int = 20,
    ) -> List[Dict[str, Any]]:
        """Queries Elastic Search with the provided arguments
        Parameters
        ----------
        args : dict of list
            List of list. For instance,
            [["PKB alpha", "RAC", "PKB", "Protein kinase B", "AKT1"]]
        datefilter: bool
            Filter documents of a later publication date than the one provided
        retrieve_sentences: bool
            If True, returns sentences instead of paragraphs
        retrieve_abstracts_only: bool
            If True, returns only abstracts
        number: int
            Number of results to return

        Returns
        -------
        list of dict
            Returns a list of dictionaries containing the results of the query
        """
        query = ElasticsearchHelper.build_keyword_query(
            keyword_lists,
            not_keyword_lists,
            query_term,
            datefilter,
            retrieve_sentences,
            retrieve_abstracts_only,
        )
        # print(query)
        results = ElasticsearchHelper.es.search(
            index=ElasticsearchHelper.es_index, body=query, size=number
        )["hits"]["hits"]
        # print(results)
        return results

    @staticmethod
    def build_pubmed_id_query(
        pubmed_id_str: str,
        paragraph_id_str: Optional[str] = None,
        sentence_id_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Artifact from our indexing process
        # The sentence index uses int for the numbers, the paragraph index uses str
        # In the current index, we have no sentence_ids so they are all -1
        pubmed_id = int(pubmed_id_str)
        # if sentence_id_str is not None:
        #     sentence_id = int(sentence_id_str)
        # else:
        #     sentence_id = -1
        request_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"pmid": pubmed_id}},
                        # {"term": {"name": pubmed_id}},
                        # {"term": {"sentence_id": sentence_id}},
                    ]
                }
            }
        }
        # if paragraph_id_str is not None:
        #     paragraph_id = int(paragraph_id_str)
        #     request_body["query"]["bool"]["filter"].append(
        #         {"term": {"paragraph_id": paragraph_id}}
        #     )

        return request_body

    @staticmethod
    def build_pubmed_ids_query(
        pubmed_ids_str_list: List[str],
        paragraph_id_str: str,
        sentence_id_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Artifact from our indexing process
        # The sentence index uses int for the numbers, the paragraph index uses str
        pubmed_ids = [int(pubmed_id_str) for pubmed_id_str in pubmed_ids_str_list]
        # paragraph_id = int(paragraph_id_str)
        # if sentence_id_str is not None:
        #     sentence_id = int(sentence_id_str)
        # else:
        #     sentence_id = -1
        request_body = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"pmid": pubmed_ids}},
                        # {"terms": {"name": pubmed_ids}},
                        # {"term": {"paragraph_id": paragraph_id}},
                        # {"term": {"sentence_id": sentence_id}},
                    ]
                }
            }
        }

        return request_body


class DatasetExamples(ABC):
    def __init__(
        self,
        file: str = "01-Nov-2022-ClinicalEvidenceSummaries.tsv",
        mode="raw_full_text",
        filter_pubmed: bool = True,
        bool_group_by_citation_id: bool = False,
        bool_group_by_alteration: bool = False,
        cache_dir_prefix: str = "edel_repo_cache/civic/examples_",
        cache: bool = True,
    ):
        self.file = file
        self.mode = mode

        print("Loading [SEP] token representation...")
        self.sep_token = (
            SentenceTransformer("michiyasunaga/BioLinkBERT-base")
            ._first_module()
            .tokenizer.sep_token
        )

        # Debug Elasticsearch index
        # random_query = self.query_keywords([])

        self.maximum_synonym_length_ratio = 5
        # The following variants are included in the top 20 variants in both CIViC and
        # OncoKB
        self.general_variants_list_civic = [
            "mutation",
            "amplification",
            "overexpression",
            "expression",
            "fusion",
            "loss",
            "underexpression",
            "loss-of-function",
        ]
        self.general_variants_list_onco_kb = [
            "oncogenic mutations",
            "fusions",
            "amplification",
            "deletion",
        ]
        self.general_variants_list = (
            self.general_variants_list_civic + self.general_variants_list_onco_kb
        )

        suffix = ""
        if bool_group_by_alteration:
            suffix += "_group_by_alteration"
        elif bool_group_by_citation_id:
            suffix += "_group_by_citation_id"
        cache_dir = Path(cache_dir_prefix + mode + suffix)
        print(cache_dir)
        if cache and cache_dir.is_dir():
            print(f"Loading from cache file {cache_dir}...")
            self.dataset = Dataset.load_from_disk(cache_dir)
        else:
            self.dataset = self.create_dataset().map(
                self.add_raw_evidence, batched=True
            )

            # Explode the drugs column
            self.dataset = self.dataset.map(self.explode_treatments, batched=True)

            self.dataset = (
                self.dataset.map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "gene"},
                )
                .map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "variant"},
                )
                .map(
                    self.filter_entity_list_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "drugs"},
                )
            )

            # Aggregate drugs from the same citation ID
            if bool_group_by_citation_id:
                self.dataset = self.aggregate_drugs_from_same_citation_id(
                    self.dataset  # type: ignore
                )

            # Add synonyms
            self.nci_thesaurus = NCITheSaurusMapper()
            self.ncbi_gene_db = SqliteDict(
                "edel_repo_cache/gene_names.sqlite",
                autocommit=True,
                tablename="gene_id_to_names",
            )

            self.dataset = (
                self.dataset.map(self.add_gene_synonyms, batched=True)
                .map(self.add_drug_synonyms, batched=True)
                .map(self.add_variant_synonyms, batched=True)
                .map(
                    self.filter_drug_synonyms_in_text,
                    batched=True,
                )
                .map(
                    self.filter_gene_synonyms_in_text,
                    batched=True,
                )
                .map(
                    self.filter_variant_synonyms_in_text,
                    batched=True,
                )
            )
            if cache:
                self.dataset.save_to_disk(cache_dir)

        # Split across Entrez Gene IDs
        eg_ids = set(self.dataset["entrez_id"])  # type: ignore
        np.random.seed(42)
        permutation = np.random.permutation(list(eg_ids))
        self.train_split = permutation[: int(len(permutation) * 0.7)]
        self.dev_split = permutation[
            int(len(permutation) * 0.7) : int(len(permutation) * 0.8)
        ]
        self.test_split = permutation[int(len(permutation) * 0.8) :]

        # Filter dataset
        self.train = self.dataset.filter(lambda x: x["entrez_id"] in self.train_split)
        self.dev = self.dataset.filter(lambda x: x["entrez_id"] in self.dev_split)
        self.test = self.dataset.filter(lambda x: x["entrez_id"] in self.test_split)

    @abstractmethod
    def create_dataset(self) -> Dataset:
        pass

    # Exploding the treatment list
    def explode_treatments(self, example: Dict[str, Any]) -> Dict[str, Any]:
        drug_list = []
        for drugs in example["drugs"]:
            if drugs is not None and type(drugs) == str:
                drug_list.append(tuple(drugs.split(",")))
            elif drugs is not None and type(drugs) == list:
                drug_list.append(tuple(drugs))
            else:
                drug_list.append([])
        example["drugs"] = drug_list
        return example

    # We are adding full texts here as well
    def add_raw_evidence(self, example: Dict[str, Any]) -> Dict[str, Any]:
        queries = [
            ElasticsearchHelper.build_pubmed_id_query(citation_id, "0")
            for citation_id in example["citation_id"]
        ]
        example["evidence_raw_full_text"] = []
        example["evidence_raw_text"] = []
        for query in queries:
            text = ""
            abstract = ""
            # 10000 is the maximum number of results that can be returned
            results = ElasticsearchHelper.es.search(
                index=ElasticsearchHelper.es_index, body=query, size=10000
            )["hits"]["hits"]
            for i, result in enumerate(results):
                # text += result["_source"]["content"] + " "
                text += (
                    result["_source"]["title"]
                    + self.sep_token
                    + result["_source"]["abstract"]
                    + self.sep_token
                )
                if i == 0:
                    # abstract = result["_source"]["content"]
                    # abstract = result["_source"]["text"]
                    abstract = (
                        result["_source"]["title"]
                        + self.sep_token
                        + result["_source"]["abstract"]
                    )
            example["evidence_raw_full_text"].append(text)
            example["evidence_raw_text"].append(abstract)

        return example

    def filter_entity_in_text(
        self, example: Dict[str, Any], entity_type: str
    ) -> Dict[str, Any]:
        text_type = "evidence_" + self.mode
        entity_in_text = entity_type + "_in_" + self.mode
        for i, text in enumerate(example[text_type]):
            if example[entity_type][i] is None or text is None:
                print(entity_type)
                print(example[entity_type][i])
                print(text)
        example[entity_in_text] = [
            [example[entity_type][i].lower() in text.lower()]
            for i, text in enumerate(example[text_type])
        ]

        entity_index_text = entity_type + "_index_" + self.mode
        example[entity_index_text] = []
        for i, text in enumerate(example[text_type]):
            example[entity_index_text].append([])
            # In comparison to drugs, other entities only have one entity per text
            example[entity_index_text][-1].append([])
            sample = example[entity_type][i]
            index = text.lower().find(sample.lower())
            if index != -1:
                # Pyarrow expects same types for all entries, i.e., string
                example[entity_index_text][-1][-1].append(
                    (sample, str(index), str(index + len(sample)))
                )

        return example

    def filter_entity_list_in_text(
        self, example: Dict[str, Any], entity_type: str
    ) -> Dict[str, Any]:
        """
        mode: "raw_text" for abstract or "raw_full_text" for full text
        """
        text_type = "evidence_" + self.mode
        entity_in_text = entity_type + "_in_" + self.mode
        example[entity_in_text] = []
        for i, text in enumerate(example[text_type]):
            example[entity_in_text].append([])
            if type(example[entity_type][i]) == str:
                entity_list = example[entity_type][i].split(",")
            else:
                entity_list = example[entity_type][i]
            for sample in entity_list:
                example[entity_in_text][-1].append(sample.lower() in text.lower())
                # print(sample.lower())
                # print(text.lower())
                # input()

        entity_index_text = entity_type + "_index_" + self.mode
        example[entity_index_text] = []
        for i, text in enumerate(example[text_type]):
            example[entity_index_text].append([])
            if type(example[entity_type][i]) == str:
                entity_list = example[entity_type][i].split(",")
            else:
                entity_list = example[entity_type][i]
            for sample in entity_list:
                example[entity_index_text][-1].append([])
                index = text.lower().find(sample.lower())
                if index != -1:
                    # Pyarrow expects same types for all entries, i.e., string
                    example[entity_index_text][-1][-1].append(
                        (sample, str(index), str(index + len(sample)))
                    )

        return example

    def aggregate_drugs_from_same_citation_id(
        self,
        dataset: Dataset,
    ) -> Dataset:
        columns = [
            "entrez_id",
            "gene",
            "variant",
            "source_type",
            "citation_id",
            "evidence_" + self.mode,
            "drugs",
            "drugs_in_" + self.mode,
            "gene_index_" + self.mode,
            "variant_index_" + self.mode,
            "drugs_index_" + self.mode,
            "gene_in_" + self.mode,
            "variant_in_" + self.mode,
        ]

        columns_dict = {
            "drugs": "sum",
            "drugs_in_" + self.mode: "sum",
            "gene_index_" + self.mode: "sum",
            "variant_index_" + self.mode: "sum",
            "drugs_index_" + self.mode: "sum",
            "gene_in_" + self.mode: "first",
            "variant_in_" + self.mode: "first",
        }

        pd_dataframe: pd.DataFrame = dataset.to_pandas()
        pd_dataframe = pd_dataframe.reset_index()
        pd_dataframe = pd_dataframe[columns]
        # Transform list columns to tuples
        list_columns = [
            "drugs",
            "gene_in_" + self.mode,
            "drugs_in_" + self.mode,
            "variant_in_" + self.mode,
            "gene_index_" + self.mode,
            "variant_index_" + self.mode,
            "drugs_index_" + self.mode,
        ]
        for col in list_columns:
            if col in [
                "drugs",
                "gene_in_" + self.mode,
                "drugs_in_" + self.mode,
                "variant_in_" + self.mode,
            ]:
                pd_dataframe[col] = pd_dataframe[col].apply(lambda x: tuple(x))
            # elif col in ["gene_index_" + self.mode, "variant_index_" + self.mode]:
            #     pd_dataframe[col] = pd_dataframe[col].apply(
            #         lambda x: tuple([tuple(y) for y in x])
            #     )
            else:  # gene_index, drugs_index and variant_index
                pd_dataframe[col] = pd_dataframe[col].apply(
                    lambda x: tuple([tuple(z) for y in x for z in y])
                )

        # pd.set_option("display.max_columns", None)
        # print(pd_dataframe.head(10))

        pd_dataframe = pd_dataframe.drop_duplicates()  # Unique rows

        pd_dataframe_raw_text = pd_dataframe.groupby(
            [
                "entrez_id",
                "gene",
                "variant",
                "source_type",
                "citation_id",
                "evidence_" + self.mode,
            ]
        ).agg(columns_dict)
        agg_dataset = Dataset.from_pandas(pd_dataframe_raw_text)

        return agg_dataset

    def add_gene_synonyms(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example["gene_synonyms"] = [
            filter_short_synonyms(
                [self.ncbi_gene_db[eg_id]["Symbol"]]
                + self.ncbi_gene_db[eg_id]["Synonyms"]
                + [self.ncbi_gene_db[eg_id]["Symbol_from_nomenclature_authority"]]
                + [self.ncbi_gene_db[eg_id]["Full_name_from_nomenclature_authority"]]
            )
            for eg_id in example["entrez_id"]
        ]
        example["gene_full_name"] = [
            filter_full_name(
                self.ncbi_gene_db[eg_id]["Full_name_from_nomenclature_authority"]
            )
            for eg_id in example["entrez_id"]
        ]
        return example

    def add_variant_synonyms(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example["variant_synonyms"] = [[variant] for variant in example["variant"]]
        return example

    def add_drug_synonyms(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example["drugs_synonyms"] = []
        for drugs_list in example["drugs"]:
            synonyms_all_drugs = []
            for drug in drugs_list:
                synonyms = [drug]
                synonym_list = list(
                    self.nci_thesaurus.mapper.loc[
                        self.nci_thesaurus.mapper["display name"] == drug
                    ]["synonyms"]
                )
                if len(synonym_list) > 0:
                    for synonym in synonym_list[0].split("|"):
                        synonyms.append(synonym)
                synonyms_all_drugs.append(synonyms)
            example["drugs_synonyms"].append(synonyms_all_drugs)
        return example

    def count_entity_matches_in_text(
        self,
        example: Dict[str, Any],
        column_names: List[str],
    ) -> Dict[str, Any]:
        # print(example)
        text_type = "evidence_" + self.mode
        for column in column_names:
            synonyms_in_text = "number_" + column
            example[synonyms_in_text] = [
                sum(
                    [
                        int(example[column][i][j] is True)
                        for j in range(len(example[column][i]))
                    ]
                )
                for i, _ in enumerate(example[text_type])
            ]
        example["entity_matches_in_" + self.mode] = [
            np.prod([example["number_" + column][i] for column in column_names])
            for i, _ in enumerate(example[text_type])
        ]
        example["entry_matches_in_" + self.mode] = [
            np.prod([min(1, example["number_" + column][i]) for column in column_names])
            for i, _ in enumerate(example[text_type])
        ]
        # print(example)
        return example

    def is_substring_in_set(self, substring: str, string_set: Set[str]):
        for string in string_set:
            if substring in string:
                return True
        return False

    def get_shortest_synonyms_subset(
        self, entity: str, synonyms: List[str], k: int = 3
    ) -> List[str]:
        synonym_subset = [entity]
        # Additionally, get the k - 1 shortest synonyms which do not share
        # any prefix greater equal three characters
        for synonym in sorted(synonyms, key=len):
            if len(synonym) >= 3 and not any(
                [synonym.startswith(prefix[:3]) for prefix in synonym_subset]
            ):
                synonym_subset.append(synonym)
            if len(synonym_subset) >= k + 2:
                break
            if len(synonym) > 8:  # For long synonyms, we only need one
                break

        # If the k + 1 st or k + 2 nd synonym are shorter than 5 characters, keep them
        if len(synonym_subset) >= k + 1 and len(synonym_subset[k]) > 5:
            synonym_subset = synonym_subset[:k]
        elif len(synonym_subset) >= k + 2 and len(synonym_subset[k + 1]) > 5:
            synonym_subset = synonym_subset[: k + 1]

        return synonym_subset

    def filter_gene_synonyms_in_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        text_type = "evidence_" + self.mode
        gene_synonyms_in_text = "gene_synonyms_in_" + self.mode
        gene_synonyms_index_text = "gene_synonyms_index_" + self.mode
        example[gene_synonyms_in_text] = []
        example[gene_synonyms_index_text] = []
        for i, text in enumerate(example[text_type]):
            example[gene_synonyms_index_text].append([])
            # Gene only has one entity per text so we need to add another list
            example[gene_synonyms_index_text][-1].append([])
            any_synonym_in_text = False
            found_mentions: set[str] = set()
            synonym_subset = self.get_shortest_synonyms_subset(
                example["gene"][i], example["gene_synonyms"][i]
            )
            for synonym in sorted(synonym_subset):
                if synonym.lower() in text.lower():
                    any_synonym_in_text = True
                    # break  # For the indexes, we need to find all mentions
                    start_char = text.lower().find(synonym.lower())
                    end_char = start_char + len(synonym)
                    substring = text[start_char:end_char].lower()
                    if not self.is_substring_in_set(substring, found_mentions):
                        found_mentions.add(substring)
                        # Pyarrow expects same types for all entries, i.e., string
                        example[gene_synonyms_index_text][-1][-1].append(
                            (substring, str(start_char), str(end_char))
                        )
                # Heuristic: Check if all subwords are included in the text
                # TODO: We need to check if multiple variations of the subword are
                # included in the text
                all_subwords_in_text = self.check_all_subwords_in_text(synonym, text)
                if all_subwords_in_text:
                    any_synonym_in_text = True
                    # break  # For the indexes, we need to find all mentions
                    # Find positions of subword occurrences
                    synonym_subwords = re.split(r"\W+", synonym.lower())
                    substring, (start_char, end_char) = self.min_window_with_indices(
                        text.lower(), synonym_subwords
                    )
                    if (
                        not self.is_substring_in_set(substring, found_mentions)
                        and substring != ""
                        # Substring must not be too long (, i.e., span the whole doc)
                        and len(substring)
                        < self.maximum_synonym_length_ratio * len(synonym)
                    ):
                        found_mentions.add(substring)
                        # Pyarrow expects same types for all entries, i.e., string
                        example[gene_synonyms_index_text][-1][-1].append(
                            (substring, str(start_char), str(end_char))
                        )
            example[gene_synonyms_in_text].append([any_synonym_in_text])
        return example

    def check_all_subwords_in_text(self, word: str, text: str) -> bool:
        for subword in re.split(r"\W+", word):
            # print(subword)
            if subword.lower() not in text.lower():
                return False
        return True

    def check_all_subwords_in_text_with_stops(self, word: str, text: str) -> bool:
        for subword in re.split(r"\W+", word):
            # print(subword)
            match_iterator = self.match_string_with_stops(subword, text)
            first_match = next(match_iterator, None)
            if first_match is None:
                return False
        return True

    def match_string_with_stops(self, word, text):
        # Define the pattern with required stop characters around the subword
        # using \b for word boundaries might not be necessary here if using stop characters
        stop_chars = r"[\s,.()]"  # includes space, comma, period, and parentheses
        pattern = rf"{stop_chars}+({re.escape(word)}){stop_chars}+"

        # Special cases for autocatalysis synonyms "auto" and "self" which may be part of other words like "autophosphorylation"
        # There we only need to check for occurrences of "auto" or "self"
        # Check for stop characters around the word only at the beginning
        if "auto" in word:
            pattern = rf"{stop_chars}+(auto)"
        elif "self" in word:
            pattern = rf"{stop_chars}+(self)"

        # Perform case-insensitive search
        return re.finditer(pattern, text, re.IGNORECASE)

    def min_window_with_indices(self, text, pattern):
        # Tokenize the text
        text_list = re.split(r"(?<=\W)", text)

        pattern_freq = Counter(pattern)
        window_counts = Counter()

        left, right = 0, 0
        formed = 0
        required = len(pattern_freq)

        min_length = float("inf")
        min_window = ""

        while right < len(text_list):
            word = re.sub(r"\W+$", "", text_list[right])
            window_counts[word] += 1

            if word in pattern_freq and window_counts[word] == pattern_freq[word]:
                formed += 1

            while left <= right and formed == required:
                word = re.sub(r"\W+$", "", text_list[left])

                if right - left + 1 < min_length:
                    min_length = right - left + 1
                    min_window = "".join(text_list[left : right + 1])

                window_counts[word] -= 1
                if word in pattern_freq and window_counts[word] < pattern_freq[word]:
                    formed -= 1

                left += 1

            right += 1

        # Remove trailing separators from min_window
        min_window = re.sub(r"\W+$", "", min_window)

        # Find the character indices of min_window in the original text
        start_char = text.find(min_window)
        end_char = start_char + len(min_window)

        return min_window, (start_char, end_char)

    def filter_variant_synonyms_in_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        text_type = "evidence_" + self.mode
        variant_synonyms_in_text = "variant_synonyms_in_" + self.mode
        variant_synonyms_index_text = "variant_synonyms_index_" + self.mode
        example[variant_synonyms_in_text] = []
        example[variant_synonyms_index_text] = []
        for i, text in enumerate(example[text_type]):
            example[variant_synonyms_index_text].append([])
            # Variant only has one entity per text so we need to add another list
            example[variant_synonyms_index_text][-1].append([])
            any_synonym_in_text = False
            found_mentions: set[str] = set()
            for synonym in sorted(
                example["variant_synonyms"][i], key=len, reverse=True
            ):
                # Check against the general variants list
                if synonym.lower() in self.general_variants_list:
                    any_synonym_in_text = True
                    # TODO there is no index for this
                    example[variant_synonyms_index_text][-1][-1].append(
                        (synonym.lower(), "-1", "-1")
                    )

                if synonym.lower() in text.lower():
                    any_synonym_in_text = True
                    # break  # For the indexes, we need to find all mentions
                    start_char = text.lower().find(synonym.lower())
                    end_char = start_char + len(synonym)
                    substring = text[start_char:end_char].lower()
                    if not self.is_substring_in_set(substring, found_mentions):
                        found_mentions.add(substring)
                        # Pyarrow expects same types for all entries, i.e., string
                        example[variant_synonyms_index_text][-1][-1].append(
                            (substring, str(start_char), str(end_char))
                        )

                # Heuristic 1: Check if all subwords are included in the text
                all_subwords_in_text = self.check_all_subwords_in_text(synonym, text)
                if all_subwords_in_text:
                    any_synonym_in_text = True
                    # break  # For the indexes, we need to find all mentions
                    # Find positions of subword occurrences
                    synonym_subwords = re.split(r"\W+", synonym.lower())
                    substring, (start_char, end_char) = self.min_window_with_indices(
                        text.lower(), synonym_subwords
                    )
                    if (
                        not self.is_substring_in_set(substring, found_mentions)
                        and substring != ""
                        # Substring must not be too long (, i.e., span the whole doc)
                        and len(substring)
                        < self.maximum_synonym_length_ratio * len(synonym)
                    ):
                        found_mentions.add(substring)
                        # Pyarrow expects same types for all entries, i.e., string
                        example[variant_synonyms_index_text][-1][-1].append(
                            (substring, str(start_char), str(end_char))
                        )

                # Heuristic 2: Split across parentheses and check them separately
                # according to heuristic 1
                # Has actually been only relevant for one case, not really necessary
                # Example C77_N78insL (c.230_231insTCT) -> C77_N78insL and
                # c.230_231insTCT
                # Matches first part
                # match = re.search(r"^([^()]+)", synonym)
                # result = match.group(1).strip() if match else None
                # if result:
                #     all_subwords_in_text = self.check_all_subwords_in_text(
                #       result, text)
                #     if all_subwords_in_text:
                #         print(synonym)
                #         print(f"Found pattern: {result}")
                #         print(f"In text: {text}")
                #         any_synonym_in_text = True
                #         break
                # # Matches second part
                # match = re.search(r"\((.*)\)", synonym)
                # result = match.group(1).strip() if match else None
                # if result:
                #     all_subwords_in_text = self.check_all_subwords_in_text(
                #       result, text)
                #     if all_subwords_in_text:
                #         print(synonym)
                #         print(f"Found pattern: {result}")
                #         print(f"In text: {text}")
                #         any_synonym_in_text = True
                #         break

                # Heuristic 3: Check if any mutation synonym is included in the text
                # and no subword in text
                # pattern_found = has_pattern(text.lower(), mutation_matching_automaton)
                # if pattern_found and not any_subword_in_text:
                #     print(f"Found pattern: {pattern_found}")
                #     print(f"In text: {text}")
                #     print(f"For variant: {synonym}")
                #     print(f"For gene: {example['gene'][i]}")
                #     print(f"For drugs: {example['drugs'][i]}")
                #     any_synonym_in_text = True
                #     break

            example[variant_synonyms_in_text].append([any_synonym_in_text])
        return example

    def filter_drug_synonyms_in_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        text_type = "evidence_" + self.mode
        drug_synonyms_in_text = "drugs_synonyms_in_" + self.mode
        drug_synonyms_index_text = "drugs_synonyms_index_" + self.mode
        example[drug_synonyms_in_text] = []
        example[drug_synonyms_index_text] = []
        for i, text in enumerate(example[text_type]):
            example[drug_synonyms_in_text].append([])
            example[drug_synonyms_index_text].append([])
            for drug in example["drugs_synonyms"][i]:
                found_mentions: set[str] = set()
                example[drug_synonyms_index_text][-1].append([])
                any_synonym_in_text = False
                for synonym in sorted(drug, key=len, reverse=True):
                    if synonym.lower() in text.lower():
                        if not any_synonym_in_text:
                            example[drug_synonyms_in_text][-1].append(True)
                        any_synonym_in_text = True
                        # break  # For the indexes, we need to find all mentions
                        start_char = text.lower().find(synonym.lower())
                        end_char = start_char + len(synonym)
                        substring = text[start_char:end_char].lower()
                        if not self.is_substring_in_set(substring, found_mentions):
                            found_mentions.add(substring)
                            # Pyarrow expects same types for all entries
                            example[drug_synonyms_index_text][-1][-1].append(
                                (substring, str(start_char), str(end_char))
                            )
                    # Heuristic: Check if all subwords are included in the text
                    all_subwords_in_text = self.check_all_subwords_in_text(
                        synonym, text
                    )
                    if all_subwords_in_text:
                        if not any_synonym_in_text:
                            example[drug_synonyms_in_text][-1].append(True)
                        any_synonym_in_text = True
                        # break  # For the indexes, we need to find all mentions
                        # Find positions of subword occurrences
                        synonym_subwords = re.split(r"\W+", synonym.lower())
                        substring, (
                            start_char,
                            end_char,
                        ) = self.min_window_with_indices(text.lower(), synonym_subwords)
                        if (
                            not self.is_substring_in_set(substring, found_mentions)
                            and substring != ""
                            # Substring must not be too long (, i.e., span the whole doc)
                            and len(substring)
                            < self.maximum_synonym_length_ratio * len(synonym)
                        ):
                            found_mentions.add(substring)
                            # Pyarrow expects same types for all entries
                            example[drug_synonyms_index_text][-1][-1].append(
                                (substring, str(start_char), str(end_char))
                            )
                if not any_synonym_in_text:
                    example[drug_synonyms_in_text][-1].append(False)

        return example

    def return_stats(self, column_names: list):
        """Print statistics about the number of examples where the given columns co-occur."""
        self.dataset = self.dataset.map(
            self.count_entity_matches_in_text,
            batched=True,
            fn_kwargs={"column_names": column_names},
            load_from_cache_file=False,
        )
        return f"""Number of examples where {column_names} do co-occur: {sum(self.dataset['entry_matches_in_' + self.mode])}/{sum(self.dataset['entity_matches_in_' + self.mode])}"""

    def detailed_dataset_stats(self):
        examples_with_drugs = self.dataset.filter(
            lambda example: len(example["drugs"]) > 0
        )
        train_with_drugs = self.train.filter(lambda example: len(example["drugs"]) > 0)
        dev_with_drugs = self.dev.filter(lambda example: len(example["drugs"]) > 0)
        test_with_drugs = self.test.filter(lambda example: len(example["drugs"]) > 0)
        # Sum up the examples with drugs
        number_of_examples_with_drugs = sum(
            [len(example["drugs"]) for example in examples_with_drugs]
        )
        number_of_examples_with_variants = len(
            self.dataset.filter(lambda example: len(example["variant"]) > 0)
        )
        number_of_examples_with_genes = len(
            self.dataset.filter(lambda example: len(example["gene"]) > 0)
        )
        examples_with_all_entities = self.dataset.filter(
            lambda example: len(example["gene"]) > 0
            and len(example["variant"]) > 0
            and len(example["drugs"]) > 0
        )
        number_of_entries_with_all_entities = sum(
            [min(1, len(example["drugs"])) for example in examples_with_all_entities]
        )
        number_of_examples_with_all_entities = sum(
            [len(example["drugs"]) for example in examples_with_all_entities]
        )

        number_of_unique_genes_train = len(
            set([example["gene"] for example in self.train])
        )
        number_of_unique_eg_ids_train = len(
            set([example["entrez_id"] for example in self.train])
        )
        number_of_unique_genes_dev = len(set([example["gene"] for example in self.dev]))
        number_of_unique_genes_test = len(
            set([example["gene"] for example in self.test])
        )
        number_of_unique_gene_variants_train = len(
            set([(example["gene"], example["variant"]) for example in self.train])
        )
        number_of_unique_gene_variants_train_with_drugs = len(
            set([(example["gene"], example["variant"]) for example in train_with_drugs])
        )
        number_of_unique_gene_variants_dev = len(
            set([(example["gene"], example["variant"]) for example in self.dev])
        )
        number_of_unique_gene_variants_dev_with_drugs = len(
            set([(example["gene"], example["variant"]) for example in dev_with_drugs])
        )
        number_of_unique_gene_variants_test = len(
            set([(example["gene"], example["variant"]) for example in self.test])
        )
        number_of_unique_gene_variants_test_with_drugs = len(
            set([(example["gene"], example["variant"]) for example in test_with_drugs])
        )

        adjusted_train = train_with_drugs.filter(
            lambda example: len(example["evidence_raw_text"]) > 3
        )
        adjusted_dev = dev_with_drugs.filter(lambda example: len(example["evidence_raw_text"]) > 3)
        adjusted_test = test_with_drugs.filter(lambda example: len(example["evidence_raw_text"]) > 3)

        # Add one separate record for each drug in the list
        unique_triples_train = set(
                [
                    (example["gene"], example["variant"], drug)
                    for example in adjusted_train
                    for drug in example["drugs"]
                ]
            )
        number_of_unique_triples_train = len(
            set(
                [
                    (example["gene"], example["variant"], drug)
                    for example in adjusted_train
                    for drug in example["drugs"]
                ]
            )
        )
        number_of_unique_triples_dev = len(
            set(
                [
                    (example["gene"], example["variant"], drug)
                    for example in adjusted_dev
                    for drug in example["drugs"]
                ]
            )
        )
        number_of_unique_triples_test = len(
            set(
                [
                    (example["gene"], example["variant"], drug)
                    for example in adjusted_test
                    for drug in example["drugs"]
                ]
            )
        )

        unique_gene_variants_pmids_train = list(  # set(
            [
                (example["gene"], example["variant"], example["citation_id"])
                for example in adjusted_train
            ]
        )
        unique_gene_variants_pmids_dev = list(  # set(
            [
                (example["gene"], example["variant"], example["citation_id"])
                for example in adjusted_dev
            ]
        )
        unique_gene_variants_pmids_test = list(  # set(
            [
                (example["gene"], example["variant"], example["citation_id"])
                for example in adjusted_test
            ]
        )
        # self.unique_gene_variants_pmids = unique_gene_variants_pmids_train | unique_gene_variants_pmids_dev | unique_gene_variants_pmids_test
        self.unique_gene_variants_pmids = unique_gene_variants_pmids_train + unique_gene_variants_pmids_dev + unique_gene_variants_pmids_test
        number_of_unique_gene_variant_pmid_train = len(
            unique_gene_variants_pmids_train
        )
        number_of_unique_gene_variant_pmid_dev = len(unique_gene_variants_pmids_dev)
        number_of_unique_gene_variant_pmid_test = len(unique_gene_variants_pmids_test)

        number_of_unique_pmids_train = len(set([example["citation_id"] for example in self.train]))
        number_of_unique_pmids_dev = len(set([example["citation_id"] for example in self.dev]))
        number_of_unique_pmids_test = len(set([example["citation_id"] for example in self.test]))

        # Group by each citation_id and count the number of unique genes and unique gene-variant pairs
        tmp_parquet_path = "po_datasets/tmp.parquet"
        self.dataset.to_parquet(tmp_parquet_path)
        polars_dataset = pl.read_parquet(tmp_parquet_path)
        # agg_gene = (
        #     polars_dataset.groupby("citation_id")
        #     .agg(
        #         ["gene", "citation_id"]).count()
            
        # )
        print(polars_dataset)
        agg_freq_count = (
            polars_dataset.group_by(["citation_id", "gene", "variant"])
            .len()
            .group_by("citation_id")
            .len(name="frequency_count")
        )
        agg_gene_variant = (
            agg_freq_count.group_by("frequency_count")
            .len(name="gene_variant_pairs")
        )
        agg_gene_variant = (
            agg_gene_variant
            .with_columns(pl.lit(agg_gene_variant["gene_variant_pairs"].sum()).alias("total_gene_variant_pairs"))
            .with_columns((pl.col("gene_variant_pairs") / pl.col("total_gene_variant_pairs") * 100).alias("percentage"))
            .sort("frequency_count", descending=False)
        )
        print("Median frequency count of gene-variant pairs per citation_id")
        print(agg_freq_count.median())
        print("Average frequency count of gene-variant pairs per citation_id")
        print(agg_freq_count.mean())
        print(agg_gene_variant)

        variant_dict = {}
        for example in self.dataset:
            if len(example["variant"]) > 0:
                variant_dict.setdefault(example["variant"], 0)
                variant_dict[example["variant"]] += 1
        sorted_variant_dict = sorted(
            variant_dict.items(), key=lambda x: x[1], reverse=True
        )
        # Get top 20 most frequent variants and their counts and add them as string to details
        top_20_variants = "\n".join(
            [f"{variant[0]}: {variant[1]}" for variant in sorted_variant_dict[:20]]
        )

        details = [
            "Number of unique entries/Number of entries counting all drugs separately",
            f"Total number of examples: {len(self.dataset)}",
            f"Train examples: {len(self.train)}",
            f"Dev examples: {len(self.dev)}",
            f"Test examples: {len(self.test)}",
            f"Number of unique genes in train: {number_of_unique_genes_train}",
            f"Number of unique entrez ids in train: {number_of_unique_eg_ids_train}",
            f"Number of unique genes in dev: {number_of_unique_genes_dev}",
            f"Number of unique genes in test: {number_of_unique_genes_test}",
            f"Number of unique gene-variant pairs in train: {number_of_unique_gene_variants_train}",
            f"  - with drugs: {number_of_unique_gene_variants_train_with_drugs}",
            f"Number of unique gene-variant pairs in dev: {number_of_unique_gene_variants_dev}",
            f"  - with drugs: {number_of_unique_gene_variants_dev_with_drugs}",
            f"Number of unique gene-variant pairs in test: {number_of_unique_gene_variants_test}",
            f"  - with drugs: {number_of_unique_gene_variants_test_with_drugs}",
            "  ",
            f"First hundred train triples: {list(unique_triples_train)[:100]}",
            f"Number of unique triples in train: {number_of_unique_triples_train}",
            f"Number of unique triples in dev: {number_of_unique_triples_dev}",
            f"Number of unique triples in test: {number_of_unique_triples_test}",
            "  ",
            f"First hundred train gene-variant-pmid triples: {sorted(unique_gene_variants_pmids_train, key=lambda x: (x[0], x[1]))[:100]}",
            f"Number of unique gene-variant-pmid triples in train: {number_of_unique_gene_variant_pmid_train}",
            f"Number of unique gene-variant-pmid triples in dev: {number_of_unique_gene_variant_pmid_dev}",
            f"Number of unique gene-variant-pmid triples in test: {number_of_unique_gene_variant_pmid_test}",
            "  ",
            f"Number of unique pmids in train: {number_of_unique_pmids_train}",
            f"Number of unique pmids in dev: {number_of_unique_pmids_dev}",
            f"Number of unique pmids in test: {number_of_unique_pmids_test}",
            f"Number of examples with a gene entity: {number_of_examples_with_genes}",
            self.return_stats(
                ["gene_synonyms_in_" + self.mode],
            ),
            self.return_stats(
                ["gene_in_" + self.mode],
            ),
            f"Number of examples with a variant entity: {number_of_examples_with_variants}",
            self.return_stats(
                ["variant_synonyms_in_" + self.mode],
            ),
            self.return_stats(
                ["variant_in_" + self.mode],
            ),
            f"Number of examples with at least one drug: {number_of_examples_with_drugs}",
            self.return_stats(
                ["drugs_synonyms_in_" + self.mode],
            ),
            self.return_stats(
                ["drugs_in_" + self.mode],
            ),
            f"Number of examples with all entities: {number_of_entries_with_all_entities}/{number_of_examples_with_all_entities}",
            self.return_stats(
                [
                    "gene_synonyms_in_" + self.mode,
                    "variant_synonyms_in_" + self.mode,
                    "drugs_synonyms_in_" + self.mode,
                ],
            ),
            self.return_stats(
                [
                    "gene_synonyms_in_" + self.mode,
                    "variant_in_" + self.mode,
                    "drugs_synonyms_in_" + self.mode,
                ],
            ),
            self.return_stats(
                [
                    "gene_in_" + self.mode,
                    "variant_synonyms_in_" + self.mode,
                    "drugs_synonyms_in_" + self.mode,
                ],
            ),
            self.return_stats(
                [
                    "gene_in_" + self.mode,
                    "variant_in_" + self.mode,
                    "drugs_in_" + self.mode,
                ],
            ),
            f"Top 20 most frequent variants: {top_20_variants}",
            # self.dataset[:1],
        ]

        return "\n".join(details)
