import argparse
import datetime
import json
import re
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import polars as pl
import pytrec_eval
from datasets import load_from_disk
from tqdm import tqdm

from evaluation.data import DRUG_KEYWORDS
from evaluation.evaluation_retriever import BiEncoderRetrieverEvaluator
from models.uniprot_retriever import get_uniprot_retriever_query
from po_datasets.dataset import ElasticsearchHelper
from utils.utils import (
    get_dataset_dict,
    get_dataset_dict_from_csv,
    get_dataset_dict_uniprot,
    get_retriever_query,
)


def beir_evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    log_file: TextIOWrapper,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        log_file.write("\n")
        for k in eval.keys():
            log_file.write("{}: {:.4f}\n".format(k, eval[k]))

    return ndcg, _map, recall, precision


class DatasetEvaluator:
    def __init__(
        self,
        retriever: Union[BiEncoderRetrieverEvaluator],
        dataset_name: str,
        drug_triggers,
        dataset_dictionary,
        file_prefix="data/retriever_results/",
        results_file="retriever_results_2.log",
        bool_bi_encoder_results=False,
        synonyms_in_query=False,
        full_name_in_query=False,
        data_split="test",
        compare_models=["faiss", "bm25"],
        model_name="",
        model_abbreviation="",
        top_k_hits=[10, 25, 50],
        graded_ndcg=False,
    ):
        self.retriever = retriever
        self.dataset_name = dataset_name
        if self.dataset_name == "uniprot":
            self.subject_name = "substrate"
            self.object_name = "catalysts"
        else:
            self.subject_name = "gene"
            self.object_name = "drugs"

        self.total_bm25_docs = 0
        self.total_faiss_docs = 0
        self.total_faiss_es_docs = 0
        self.total_bm25_faiss_docs = 0
        self.total_objects_count = []
        self.total_objects_found = []
        self.total_objects_synonyms_found = []
        self.total_subject_objects_synonyms_found = []
        self.total_pmids_count = []
        self.total_pmids_found = []
        self.total_pmids_with_entities_count = []
        self.total_pmids_with_entities_found = []
        self.total_pmids_with_entities_synonyms_count = []
        self.total_pmids_with_entities_synonyms_found = []
        self.all_documents_found = []
        self.all_documents_with_synonyms_found = []
        self.examples_info = []
        self.total_objects_pmids_count = []
        self.total_objects_pmids_found = []
        self.total_objects_pmids_synonyms_found = []
        self.total_subjects_found = []
        self.total_subject_synonyms_found = []
        self.faiss_example_subjects_found_pmid = {"mapping": {}, "pmid_texts": {}}

        self.graded_ndcg = graded_ndcg
        self.qrels = {}
        self.results_bm25 = {}
        self.results_faiss = {}

        self.model_name = model_name
        self.model_abbreviation = model_abbreviation
        self.top_k_hits = top_k_hits

        assert len(compare_models) == 2
        assert compare_models[0] in ["faiss", "bm25+faiss", "bm25"]
        assert compare_models[1] in ["faiss", "bm25+faiss", "bm25"]
        assert compare_models[0] != compare_models[1]

        self.model_one_string = compare_models[0]
        self.model_two_string = compare_models[1]

        self.scores_discrepancies_faiss = []
        self.scores_discrepancies_bm25 = []
        self.scores_discrepancies_bm25_faiss = []

        self.drug_triggers = drug_triggers
        self.file_prefix = file_prefix
        self.result_file = results_file

        self.dataset_dictionary = dataset_dictionary
        self.synonyms_in_query = synonyms_in_query
        self.full_name_in_query = full_name_in_query
        self.data_split = data_split
        self.bool_bi_encoder_results = bool_bi_encoder_results

        dataset_file = "edel_repo_cache/datasets/pubmed.dataset"
        if Path(dataset_file).exists():
            print("Loading dataset from local cache")
            self.dataset = load_from_disk(dataset_file)
            self.pl_dataframe = pl.from_arrow(self.dataset.data.table)
            print("Dataset loaded")

        self.evaluate()
        self.print_results()

    def match_string_with_stops(self, word, text):
        # Define the pattern with required stop characters around the subword
        # using \b for word boundaries might not be necessary here if using stop characters
        stop_chars = (
            r"[\s,.()]"  # includes any whitespace, comma, period, and parentheses
            # r"[\s,.\-()]"
        )
        pattern = rf"{stop_chars}+({re.escape(word)}){stop_chars}+"

        # Special cases for autocatalysis synonyms "auto" and "self" which may be part of other words like "autophosphorylation"
        # There we only need to check for occurrences of "auto" or "self"
        # Check for stop characters around the word only at the beginning
        if "auto" in word:
            pattern = rf"{stop_chars}+(auto)"
        elif "self" in word:
            pattern = rf"{stop_chars}+(self)"

        # Perform case-insensitive search
        return re.search(pattern, text, re.IGNORECASE)

    def evaluate(self):
        def merge(a: dict, b: dict, path=[]):
            for key in b:
                if key in a:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        merge(a[key], b[key], path + [str(key)])
                    elif a[key] != b[key]:
                        raise Exception("Conflict at " + ".".join(path + [str(key)]))
                else:
                    a[key] = b[key]
            return a

        for i, example in tqdm(
            enumerate(self.dataset_dictionary.values()),
            total=len(self.dataset_dictionary.values()),
        ):
            # print(example)
            self.faiss_examples_string = []
            self.bm25_examples_string = []
            counts = self.get_number_found_objects_and_pmids(example, i)
            self.total_objects_count.append(counts[0])
            self.total_objects_found.append(counts[1])
            self.total_objects_synonyms_found.append(counts[2])
            self.total_pmids_count.append(counts[3])
            self.total_pmids_found.append(counts[4])
            self.total_pmids_with_entities_count.append(counts[5])
            self.total_pmids_with_entities_found.append(counts[6])
            self.total_pmids_with_entities_synonyms_count.append(counts[7])
            self.total_pmids_with_entities_synonyms_found.append(counts[8])
            self.total_faiss_docs += counts[9]
            self.total_bm25_faiss_docs += counts[10]
            self.total_bm25_docs += counts[11]
            self.total_faiss_es_docs += counts[12]
            self.all_documents_found.append((i, counts[13]))
            self.all_documents_with_synonyms_found.append((i, counts[14]))
            self.examples_info.append(counts[15])
            self.scores_discrepancies_faiss.extend(counts[16])
            self.scores_discrepancies_bm25.extend(counts[17])
            self.scores_discrepancies_bm25_faiss.extend(counts[18])
            self.total_objects_pmids_count.append(counts[19])
            self.total_objects_pmids_found.append(counts[20])
            self.total_objects_pmids_synonyms_found.append(counts[21])
            self.total_subjects_found.append(counts[22])
            self.total_subject_synonyms_found.append(counts[23])
            self.total_subject_objects_synonyms_found.append(counts[24])
            self.qrels = merge(self.qrels, counts[25])
            self.results_bm25 = merge(self.results_bm25, counts[26])
            self.results_faiss = merge(self.results_faiss, counts[27])
            for count in counts[1]:
                assert count <= counts[0]
                # number of found catalysts <= number of gold catalysts

            # Debugging BEIR evaluate dicts
            # print(f"Iteration {i}")
            # print("Qrels:")
            # print(self.qrels["pmids"].keys())
            # print(self.qrels["pmids"][str(i)])
            # print(self.qrels["pmids_with_entities"].keys())
            # print(self.qrels["pmids_with_entities"][str(i)])
            # print(self.qrels["pmids_with_entities_synonyms"].keys())
            # print(self.qrels["pmids_with_entities_synonyms"][str(i)])
            # print("Results BM25:")
            # print(self.results_bm25[str(i)])
            # input("Press any key to Continue")
            # print(self.qrels)
            # input("Press any key to Continue")
            # print(self.results_bm25)
            # continue_input = input("Continue? (y/n)")
            # if continue_input == "n":
            #     break

        self.number_gold_objects = sum(self.total_objects_count)
        self.number_found_objects = np.sum(np.array(self.total_objects_found), axis=0)
        self.number_found_objects_synonyms = np.sum(
            np.array(self.total_objects_synonyms_found), axis=0
        )
        self.number_found_subject_objects_synonyms = np.sum(
            np.array(self.total_subject_objects_synonyms_found), axis=0
        )

        self.number_gold_objects_pmids = sum(self.total_objects_pmids_count)
        self.number_found_objects_pmids = np.sum(
            np.array(self.total_objects_pmids_found), axis=0
        )
        self.number_found_objects_pmids_synonyms = np.sum(
            np.array(self.total_objects_pmids_synonyms_found), axis=0
        )

        self.number_gold_pmids = sum(self.total_pmids_count)
        self.number_found_pmids = np.sum(np.array(self.total_pmids_found), axis=0)

        self.number_gold_pmids_with_entities = sum(self.total_pmids_with_entities_count)
        self.number_found_pmids_with_entities = np.sum(
            np.array(self.total_pmids_with_entities_found), axis=0
        )

        self.number_gold_pmids_with_entities_synonyms = sum(
            self.total_pmids_with_entities_synonyms_count
        )
        self.number_found_pmids_with_entities_synonyms = np.sum(
            np.array(self.total_pmids_with_entities_synonyms_found), axis=0
        )

        self.average_number_found_subjects = np.mean(
            np.array(self.total_subjects_found), axis=0
        )

        self.average_number_found_subject_synonyms = np.mean(
            np.array(self.total_subject_synonyms_found), axis=0
        )

        # Get Gold PMIDs overlap
        # 1) FAISS and BM25 found gold PMIDs
        # 2) FAISS only found gold PMIDs
        # 3) BM25 only found gold PMIDs
        # 4) FAISS and BM25 did not find gold PMIDs
        self.intersection_dict = {
            "both_found_pmids": [],
            "both_found_pmids_count": 0,
            "both_found_pmids_set": set(),
            f"{self.model_one_string}_only_pmids": [],
            f"{self.model_one_string}_only_pmids_count": 0,
            f"{self.model_one_string}_only_pmids_set": set(),
            f"{self.model_two_string}_only_pmids": [],
            f"{self.model_two_string}_only_pmids_count": 0,
            f"{self.model_two_string}_only_pmids_set": set(),
            "not_found_pmids": [],
            "not_found_pmids_count": 0,
            "not_found_pmids_set": set(),
            "gold_pmids_count": 0,
            "gold_pmids_set": set(),
        }
        for i, example_info in enumerate(self.examples_info):
            if self.bool_bi_encoder_results:
                (
                    gold_pmids,
                    bm25_faiss_pmids,
                    faiss_pmids,
                    bm25_pmids,
                ) = example_info[3]
            else:
                (
                    gold_pmids,
                    faiss_pmids,
                    bm25_pmids,
                ) = example_info[3]

            if self.model_one_string == "faiss":
                m1_pmids = faiss_pmids
            elif self.model_one_string == "bm25":
                m1_pmids = bm25_pmids
            elif self.model_one_string == "bm25+faiss":
                m1_pmids = bm25_faiss_pmids

            if self.model_two_string == "faiss":
                m2_pmids = faiss_pmids
            elif self.model_two_string == "bm25":
                m2_pmids = bm25_pmids
            elif self.model_two_string == "bm25+faiss":
                m2_pmids = bm25_faiss_pmids

            both_found_pmids = [
                pmid for pmid in gold_pmids if pmid in m1_pmids and pmid in m2_pmids
            ]
            if len(both_found_pmids) > 0:
                self.intersection_dict["both_found_pmids"].append(
                    (example_info[:3], both_found_pmids)
                )
            self.intersection_dict["both_found_pmids_set"].update(set(both_found_pmids))
            faiss_only_found_pmids = [
                pmid for pmid in gold_pmids if pmid in m1_pmids and pmid not in m2_pmids
            ]
            if len(faiss_only_found_pmids) > 0:
                self.intersection_dict[f"{self.model_one_string}_only_pmids"].append(
                    (example_info[:3], faiss_only_found_pmids)
                )
            self.intersection_dict[f"{self.model_one_string}_only_pmids_set"].update(
                set(faiss_only_found_pmids)
            )
            bm25_only_found_pmids = [
                pmid for pmid in gold_pmids if pmid not in m1_pmids and pmid in m2_pmids
            ]
            if len(bm25_only_found_pmids) > 0:
                self.intersection_dict[f"{self.model_two_string}_only_pmids"].append(
                    (example_info[:3], bm25_only_found_pmids)
                )
            self.intersection_dict[f"{self.model_two_string}_only_pmids_set"].update(
                set(bm25_only_found_pmids)
            )
            not_found_pmids = [
                pmid
                for pmid in gold_pmids
                if pmid not in m1_pmids and pmid not in m2_pmids
            ]
            if len(not_found_pmids) > 0:
                self.intersection_dict["not_found_pmids"].append(
                    (example_info[:3], not_found_pmids)
                )
            self.intersection_dict["not_found_pmids_set"].update(set(not_found_pmids))

            self.intersection_dict["gold_pmids_set"].update(set(gold_pmids))

        self.intersection_dict["both_found_pmids_count"] = len(
            self.intersection_dict["both_found_pmids_set"]
        )
        self.intersection_dict[f"{self.model_one_string}_only_pmids_count"] = len(
            self.intersection_dict[f"{self.model_one_string}_only_pmids_set"]
        )
        self.intersection_dict[f"{self.model_two_string}_only_pmids_count"] = len(
            self.intersection_dict[f"{self.model_two_string}_only_pmids_set"]
        )
        self.intersection_dict["not_found_pmids_count"] = len(
            self.intersection_dict["not_found_pmids_set"]
        )
        self.intersection_dict["gold_pmids_count"] = len(
            self.intersection_dict["gold_pmids_set"]
        )

    def get_gold_pmids_with_entity_mentions(
        self, gold_pmids, gold_texts, gold_subject, gold_objects, gold_objects_synonyms
    ):
        pmids_with_entity_mentions = []
        pmid_with_entity_synonym_mentions = []
        for pmid, text in zip(gold_pmids, gold_texts):
            found_subject = False
            found_any_object = False
            found_any_object_synoynm = False
            if self.dataset_name == "uniprot":
                found_subject = True
            elif self.match_string_with_stops(gold_subject, text):
                found_subject = True
            for object in gold_objects:
                if self.match_string_with_stops(gold_subject, text):
                    found_any_object = True
                    break
            for object_synonyms in gold_objects_synonyms:
                for object in object_synonyms:
                    if object.lower() in text.lower():
                        found_any_object_synoynm = True
                        break
            if found_subject and found_any_object:
                pmids_with_entity_mentions.append(pmid)
            if found_subject and found_any_object_synoynm:
                pmid_with_entity_synonym_mentions.append(pmid)
        return pmids_with_entity_mentions, pmid_with_entity_synonym_mentions

    def get_text_from_pmids(self, pmids):
        # TODO: Enable this again
        es_query = ElasticsearchHelper.build_pubmed_ids_query(pmids, "0")
        retrieval_result = ElasticsearchHelper.search(query=es_query, size=len(pmids))[
            "hits"
        ]["hits"]
        doc_texts = ["" for i in range(len(pmids))]
        doc_pmids_to_index = {int(pmid): i for i, pmid in enumerate(pmids)}
        # print(doc_pmids_to_index)
        # print(retrieval_result)
        for hit in retrieval_result:
            # print(type(hit["_source"]["name"]))
            # print(type(pmids[0]))
            # index_pos = doc_pmids_to_index[hit["_source"]["name"]]
            # retrieved_text = hit["_source"]["content"]
            index_pos = doc_pmids_to_index[int(hit["_source"]["pmid"])]
            retrieved_text = hit["_source"]["title"] + "[SEP]" + hit["_source"]["abstract"]
            doc_texts[index_pos] += retrieved_text

        return doc_texts

    def update_faiss_example_subjects_found_pmid(
        self, pmids, doc_texts, subject, subject_synonyms, variant
    ):
        for pmid, doc in zip(pmids, doc_texts):
            if pmid not in self.faiss_example_subjects_found_pmid["pmid_texts"]:
                self.faiss_example_subjects_found_pmid["pmid_texts"][pmid] = doc
            if subject not in self.faiss_example_subjects_found_pmid["mapping"]:
                self.faiss_example_subjects_found_pmid["mapping"][subject] = {}

            subject_synonym_found = False
            for synonym in subject_synonyms:
                if self.match_string_with_stops(synonym, doc):
                    if (
                        variant
                        not in self.faiss_example_subjects_found_pmid["mapping"][
                            subject
                        ]
                    ):
                        self.faiss_example_subjects_found_pmid["mapping"][subject][
                            variant
                        ] = [
                            [pmid],
                            [],
                        ]
                    else:
                        self.faiss_example_subjects_found_pmid["mapping"][subject][
                            variant
                        ][0].append(pmid)
                    subject_synonym_found = True
                    break

            if not subject_synonym_found:
                if (
                    variant
                    not in self.faiss_example_subjects_found_pmid["mapping"][subject]
                ):
                    self.faiss_example_subjects_found_pmid["mapping"][subject][
                        variant
                    ] = [
                        [],
                        [pmid],
                    ]
                else:
                    self.faiss_example_subjects_found_pmid["mapping"][subject][variant][
                        1
                    ].append(pmid)

    def check_object_mention_in_retrieved_texts(
        self,
        doc_texts,
        aggregated_objects_list,
        aggregated_objects_synonyms_list,
        model_index,
        pmids,
        subject,
        subject_synonyms,
    ):
        found_documents_objects = []
        objects_count_found = 0
        found_documents_objects_synonyms = []
        objects_synonyms_count_found = 0
        found_documents_subject_objects_synonyms = []
        subject_objects_synonyms_count_found = 0

        count_subject = 0
        count_subject_synonyms = 0

        # print(subject)
        # print(subject_synonyms)
        for k, object in enumerate(aggregated_objects_list):
            found = False
            for i, doc in enumerate(doc_texts):
                # subject counting
                # print(doc)
                # print(count_subject)
                # print(count_subject_synonyms)
                # continue_input = input("Continue? (y/n)")
                if k == 0 and self.match_string_with_stops(subject, doc):
                    # Only count subjects once per document
                    count_subject += 1
                if k == 0:
                    for synonym in subject_synonyms:
                        if self.match_string_with_stops(synonym, doc):
                            count_subject_synonyms += 1
                            break
                # object counting
                if self.match_string_with_stops(object, doc):
                    if not found:
                        objects_count_found += 1
                        found = True
                    found_documents_objects.append((pmids[i], object, doc, model_index))
                    # break

        for object in aggregated_objects_synonyms_list:
            object_found = False
            subject_object_found = False
            for i, doc in enumerate(doc_texts):
                for object_synonym in object:
                    # Do not allow substrings to be found
                    # if synonym.lower() in doc.lower():
                    if self.match_string_with_stops(object_synonym, doc):
                        if not object_found:
                            objects_synonyms_count_found += 1
                            object_found = True
                        found_documents_objects_synonyms.append(
                            (pmids[i], object, doc, model_index)
                        )
                        if not subject_object_found:
                            for subject_synonym in subject_synonyms:
                                if self.match_string_with_stops(subject_synonym, doc):
                                    subject_object_found = True
                                    subject_objects_synonyms_count_found += 1
                                    found_documents_subject_objects_synonyms.append(
                                        (pmids[i], subject, object, doc, model_index)
                                    )
                                    break

        return (
            objects_count_found,
            found_documents_objects,
            objects_synonyms_count_found,
            found_documents_objects_synonyms,
            subject_objects_synonyms_count_found,
            found_documents_subject_objects_synonyms,
            count_subject,
            count_subject_synonyms,
        )

    def get_gold_pmids_in_retrieved_pmids(self, gold_pmids, retrieved_pmids):
        pmids_found = set()
        for pmid in retrieved_pmids:
            if pmid in gold_pmids:
                pmids_found.add(pmid)
        return pmids_found

    def match_entities_in_text(self, example, col_names, index_number):
        entity_type_matching_dict = {}
        if self.dataset_name == "uniprot":
            entity_types = ["substrate", "ptm_type", "residue", "position", "catalysts"]
        else:
            entity_types = ["gene", "variant", "drugs"]
        pattern = "|".join(re.escape(entity_type) for entity_type in entity_types)
        for col_name in col_names.keys():
            match = re.search(pattern, col_name)
            if match:
                entity_type = match.group()
                if col_names[col_name]:
                    entity_type_matching_dict[entity_type] = any(example[col_name][index_number])
                else:
                    entity_type_matching_dict[entity_type] = not any(example[col_name][index_number])
                assert isinstance(example[col_name][index_number][0], bool)
            else:
                raise ValueError("This should not happen")
        return entity_type_matching_dict

    def get_qrels_pmids_with_relevance(self, example, index_number):
        qrels = {str(index_number): {}}
        if self.dataset_name == "uniprot":
            column_dict = {
                "substrate_synonyms": True,
                "ptm_type": True,
                "residue": True,
                "position": True,
                "catalysts_synonyms": True,
            }
            col_names_check = {
                f"{col_key}_in_raw_text": col_value
                for col_key, col_value in column_dict.items()
            }
            seen_pmids = set()
            for i, pmid in enumerate(example["citation_id"]):
                if pmid in seen_pmids:
                    continue
                matching_dict = self.match_entities_in_text(example, col_names_check, i)

                # Aggregate residue and position into one entry
                if "residue" in matching_dict and "position" in matching_dict:
                    matching_dict["respos"] = (
                        matching_dict["residue"] and matching_dict["position"]
                    )
                    del matching_dict["residue"]
                    del matching_dict["position"]
                elif "residue" in matching_dict:
                    matching_dict["respos"] = matching_dict["residue"]
                    del matching_dict["residue"]
                elif "position" in matching_dict:
                    matching_dict["respos"] = matching_dict["position"]
                    del matching_dict["position"]

                all_entities_in_text = sum(matching_dict.values()) == len(column_dict)
                three_entities_in_text = (sum(matching_dict.values()) == 3) and matching_dict["catalysts"]
                if all_entities_in_text:
                    qrels[str(index_number)][str(pmid)] = 3
                elif three_entities_in_text:
                    qrels[str(index_number)][str(pmid)] = 2
                else:
                    qrels[str(index_number)][str(pmid)] = 1
                seen_pmids.add(pmid)
        else:
            column_dict = {
                "gene_synonyms": True,
                "variant": True,
                "drugs_synonyms": True,
            }
            col_names_check = {
                f"{col_key}_in_raw_text": col_value
                for col_key, col_value in column_dict.items()
            }
            seen_pmids = set()
            for i, pmid in enumerate(example["citation_id"]):
                if pmid in seen_pmids:
                    continue
                matching_dict = self.match_entities_in_text(example, col_names_check, i)
                all_entities_in_text = sum(matching_dict.values()) == len(column_dict)
                two_entities_in_text = (sum(matching_dict.values()) == 2) and matching_dict["drugs"]
                if all_entities_in_text:
                    qrels[str(index_number)][str(pmid)] = 3
                elif two_entities_in_text:
                    qrels[str(index_number)][str(pmid)] = 2
                else:
                    qrels[str(index_number)][str(pmid)] = 1
                seen_pmids.add(pmid)

        return qrels


    def get_number_found_objects_and_pmids(self, example, index_number):
        if self.dataset_name == "uniprot":
            query = get_uniprot_retriever_query(
                example["substrate"],
                example["substrate_full_name"],
                example["substrate_synonyms"],
                example["ptm_type"],
                example["residue"],
                example["position"],
                synonyms_in_query=self.synonyms_in_query,
                full_name_in_query=self.full_name_in_query,
            )
            modifier = (example["ptm_type"], example["residue"], example["position"])
        else:
            query = get_retriever_query(
                example["gene"],
                example["gene_synonyms"],
                example["gene_full_name"],
                example["variant"],
                synonyms_in_query=self.synonyms_in_query,
            )
            modifier = example["variant"]

        faiss_scores, faiss_pm_ids = self.retriever.get_faiss_results(query)
        count_faiss_results = len(faiss_pm_ids)
        faiss_texts = self.get_text_from_pmids(faiss_pm_ids)
        count_faiss_es_results = len(faiss_texts)

        if self.bool_bi_encoder_results:
            bi_encoder_scores, bi_encoder_pm_ids = (
                self.retriever.get_bi_encoder_results(query, faiss_texts, faiss_pm_ids)
            )

            bi_encoder_texts = self.get_text_from_pmids(bi_encoder_pm_ids)
            (
                bm25_faiss_scores,
                bm25_faiss_pm_ids,
                bm25_faiss_texts,
            ) = self.retriever.get_bm25_faiss_results(query)

            count_bm25_faiss_results = len(bm25_faiss_pm_ids)
        else:
            count_bm25_faiss_results = 0

        bm25_scores, bm25_pm_ids, bm25_texts = self.retriever.get_bm25_results(query)
        count_bm25_results = len(bm25_pm_ids)

        gold_objects = sorted(set(example[self.object_name]))
        gold_objects_synonyms = sorted(
            set(
                tuple(set(sorted(synonyms)))
                for synonyms in example[f"{self.object_name}_synonyms"]
            )
        )

        if len(gold_objects) != len(gold_objects_synonyms):
            print("Gold objects and gold objects synonyms are not the same length")
            print(gold_objects)
            print(gold_objects_synonyms)
            print(example[self.object_name])
            print(example[f"{self.object_name}_synonyms"])
            print(example)
            print(index_number)

        assert len(gold_objects) == len(gold_objects_synonyms)

        gold_pmids = sorted(set(example["citation_id"]))
        gold_doc_texts = self.get_text_from_pmids(gold_pmids)

        # Get gold objects, gold pmid tuples
        gold_object_pmid_tuples = []
        assert len(example["citation_id"]) == len(example[f"{self.object_name}_list"])
        for pmid, object_list in zip(
            example["citation_id"], example[f"{self.object_name}_list"]
        ):
            for object in object_list:
                gold_object_pmid_tuples.append((object, pmid))
        gold_object_pmid_tuples = sorted(set(gold_object_pmid_tuples))
        total_objects_pmids_count = len(gold_object_pmid_tuples)

        (
            gold_pmids_with_entity_mentions,
            gold_pmids_with_entity_synonym_mentions,
        ) = self.get_gold_pmids_with_entity_mentions(
            gold_pmids,
            gold_doc_texts,
            example[self.subject_name],
            gold_objects,
            gold_objects_synonyms,
        )

        if self.bool_bi_encoder_results:
            # print(query)
            # print(len(gold_doc_texts))
            # print(len(faiss_texts))
            # print(len(bm25_faiss_texts))
            # print(len(bm25_texts))
            (
                gold_text_cosine_scores,
                faiss_text_cosine_scores,
                bm25_faiss_text_cosine_scores,
                bm25_text_cosine_scores,
            ) = self.retriever.check_similarity_scores(
                query,
                gold_doc_texts,
                faiss_texts,
                bm25_faiss_texts,
                bm25_texts,
            )

            (
                gold_faiss_scores,
                gold_bm25_scores,
                faiss_filtered_distances,
            ) = self.retriever.get_gold_pmids_scores(query, gold_pmids, faiss_pm_ids)

            retrieved_texts_all = [
                faiss_texts,
                bm25_faiss_texts,
                bm25_texts,
                bi_encoder_texts,
            ]
            retrieved_pmids_all = [
                faiss_pm_ids,
                bm25_faiss_pm_ids,
                bm25_pm_ids,
                bi_encoder_pm_ids,
            ]
        else:
            retrieved_texts_all = [
                faiss_texts,
                bm25_texts,
            ]
            retrieved_pmids_all = [
                faiss_pm_ids,
                bm25_pm_ids,
            ]

        # Get the top k = 10, 25, 50 results for each model
        retrieved_texts = []
        retrieved_pmids = []
        for texts, pmids in zip(retrieved_texts_all, retrieved_pmids_all):
            for top_k_value in self.top_k_hits:
                retrieved_texts.append(texts[:top_k_value])
                retrieved_pmids.append(pmids[:top_k_value])

        faiss_scores_fmt = [f"{x:.{2}f}" for x in faiss_scores[: max(self.top_k_hits)]]
        bm25_scores_fmt = [f"{x:.{2}f}" for x in bm25_scores[: max(self.top_k_hits)]]

        if self.bool_bi_encoder_results:
            bm25_faiss_scores_fmt = [
                f"{x:.{2}f}" for x in bm25_faiss_scores[: max(self.top_k_hits)]
            ]
            bi_encoder_scores_fmt = [
                f"{x:.{2}f}" for x in bi_encoder_scores[: max(self.top_k_hits)]
            ]

            # Plot discrepancy between top retrieved BM25/FAISS score and that of the gold PMIDs
            # Also add whether the gold PMID was found by BM25/FAISS
            scores_discrepancies_faiss = [
                (
                    index_number,
                    gold_pmids[i],
                    gold_faiss_scores[i],
                    faiss_scores[0],
                    faiss_scores[0] - gold_faiss_scores[i],
                    gold_pmids[i] in faiss_pm_ids,
                )
                for i in range(len(gold_pmids))
            ]
            scores_discrepancies_bm25 = [
                (
                    index_number,
                    gold_pmids[i],
                    gold_bm25_scores[i],
                    bm25_scores[0],
                    bm25_scores[0] - gold_bm25_scores[i],
                    gold_pmids[i] in bm25_pm_ids,
                )
                for i in range(len(gold_pmids))
            ]
            scores_discrepancies_bm25_faiss = [
                (
                    index_number,
                    gold_pmids[i],
                    gold_faiss_scores[i],
                    bm25_faiss_scores[0],
                    bm25_faiss_scores[0] - gold_faiss_scores[i],
                    gold_pmids[i] in bm25_faiss_pm_ids,
                )
                for i in range(len(gold_pmids))
            ]
        else:
            scores_discrepancies_faiss = []
            scores_discrepancies_bm25 = []
            scores_discrepancies_bm25_faiss = []

        total_objects_count = len(gold_objects)
        all_found_documents_objects = []
        total_objects_found = []
        total_objects_pmids_found = []
        all_found_documents_objects_synonyms = []
        all_found_documents_subject_objects_synonyms = []
        total_subject_objects_synonyms_found = []
        total_objects_synonyms_found = []
        total_objects_synonyms_pmids_found = []
        count_subjects_found = []
        count_subject_synonyms_found = []
        for i, texts in enumerate(retrieved_texts):
            (
                objects_found,
                found_documents_objects,
                objects_synonyms_found,
                found_documents_objects_synonyms,
                subject_objects_synonyms_found,
                found_documents_subject_objects_synonyms,
                subjects_found,
                subject_synonyms_found,
            ) = self.check_object_mention_in_retrieved_texts(
                texts,
                gold_objects,
                gold_objects_synonyms,
                i,
                retrieved_pmids[i],
                example[self.subject_name],
                example[f"{self.subject_name}_synonyms"],
            )
            total_objects_found.append(objects_found)
            all_found_documents_objects.append(found_documents_objects)
            total_objects_synonyms_found.append(objects_synonyms_found)
            all_found_documents_objects_synonyms.append(
                found_documents_objects_synonyms
            )
            total_subject_objects_synonyms_found.append(subject_objects_synonyms_found)
            all_found_documents_subject_objects_synonyms.append(
                found_documents_subject_objects_synonyms
            )

            count_subjects_found.append(subjects_found)
            count_subject_synonyms_found.append(subject_synonyms_found)

            if i == 2:  # Corresponds to FAISS top-k hits
                self.update_faiss_example_subjects_found_pmid(
                    retrieved_pmids[i],
                    texts,
                    example[self.subject_name],
                    example[f"{self.subject_name}_synonyms"],
                    modifier,
                )

            assert objects_found <= len(gold_objects)
            assert objects_synonyms_found <= len(gold_objects_synonyms)

        total_pmids_count = len(gold_pmids)
        pmids_found = [
            self.get_gold_pmids_in_retrieved_pmids(gold_pmids, pmids)
            for pmids in retrieved_pmids
        ]
        total_pmids_found = [len(pmids) for pmids in pmids_found]
        total_pmids_with_entities_count = len(gold_pmids_with_entity_mentions)
        total_pmids_with_entities_found = [
            len(
                self.get_gold_pmids_in_retrieved_pmids(
                    gold_pmids_with_entity_mentions, pmids
                )
            )
            for pmids in retrieved_pmids
        ]
        total_pmids_with_entities_synonyms_count = len(
            gold_pmids_with_entity_synonym_mentions
        )
        total_pmids_with_entities_synonyms_found = [
            len(
                self.get_gold_pmids_in_retrieved_pmids(
                    gold_pmids_with_entity_synonym_mentions, pmids
                )
            )
            for pmids in retrieved_pmids
        ]

        # Check if the objects retrieved in the documents belong to gold PMIDs
        # If so, add them to the total count
        for i, documents in enumerate(all_found_documents_objects):
            objects_pmids_count = 0
            for document in documents:
                if document[0] in gold_pmids:
                    objects_pmids_count += 1
            total_objects_pmids_found.append(objects_pmids_count)
        for i, documents in enumerate(all_found_documents_objects_synonyms):
            objects_synonyms_pmids_count = 0
            for document in documents:
                if document[0] in gold_pmids:
                    objects_synonyms_pmids_count += 1
            total_objects_synonyms_pmids_found.append(objects_synonyms_pmids_count)

        if self.bool_bi_encoder_results:
            pmids_found_gold_faiss_bm25 = (
                set(gold_pmids),
                pmids_found[1],
                pmids_found[1 + len(self.top_k_hits)],
                pmids_found[1 + 2 * len(self.top_k_hits)],
            )
        else:
            pmids_found_gold_faiss_bm25 = (
                set(gold_pmids),
                pmids_found[1],
                pmids_found[1 + len(self.top_k_hits)],
            )
        full_example_info = (
            example[self.subject_name],
            modifier,
            example[self.object_name],
            pmids_found_gold_faiss_bm25,
        )

        qrels = {}
        if self.graded_ndcg:
            qrels["pmids"] = self.get_qrels_pmids_with_relevance(
                example, index_number
            )
        else:
            qrels["pmids"] = {str(index_number): {str(pmid): 1 for pmid in gold_pmids}}
        qrels["pmids_with_entities"] = {
            str(index_number): {
                str(pmid): 1 for pmid in gold_pmids_with_entity_mentions
            }
        }
        if len(qrels["pmids_with_entities"][str(index_number)]) == 0:
            qrels["pmids_with_entities"] = {}
        qrels["pmids_with_entities_synonyms"] = {
            str(index_number): {
                str(pmid): 1 for pmid in gold_pmids_with_entity_synonym_mentions
            }
        }
        if len(qrels["pmids_with_entities_synonyms"][str(index_number)]) == 0:
            qrels["pmids_with_entities_synonyms"] = {}
        results_bm25 = {
            str(index_number): {
                str(pmid): float(score) for pmid, score in zip(bm25_pm_ids, bm25_scores)
            }
        }
        results_faiss = {
            str(index_number): {
                str(pmid): float(score)
                for pmid, score in zip(faiss_pm_ids, faiss_scores)
            }
        }

        # Write results scores and pmids to file,
        # Log them with date of execution
        # Append to log file
        max_length = min(50, max(self.top_k_hits))
        with open(self.result_file, "a") as log_file:
            # Current date and time
            current_time = datetime.datetime.now()
            date_time = current_time.strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"{date_time}\n")
            log_file.write(f"Model: {self.model_name}\n")
            log_file.write(f"Index: {index_number}\n")
            log_file.write(f"Query: {query}\n")
            log_file.write(f"Examples (subject): {example[self.subject_name]}\n")
            log_file.write(
                f"Examples (subject synonyms): {example[f'{self.subject_name}_synonyms']}\n"
            )
            log_file.write(f"Examples (modifier): {modifier}\n")
            log_file.write(f"Examples (objects): {example[self.object_name]}\n")
            log_file.write(f"Examples (gold pmids): {gold_pmids}\n")
            if self.bool_bi_encoder_results:
                log_file.write(
                    f"Examples (gold pmids FAISS scores): {gold_faiss_scores[:max_length]}\n\n"
                )
                log_file.write(
                    f"Examples (gold pmids cosine similarity): {gold_text_cosine_scores[:max_length]}\n"
                )
                log_file.write(
                    f"Examples (gold pmids BM25 scores): {gold_bm25_scores[:max_length]}\n\n"
                )
            log_file.write(
                f"Number of subjects found [FAISS, FAISS+BM25, BM25, Bi-Encoder]: {count_subjects_found[:max_length]}\n"
            )
            log_file.write(
                f"Number of subject synonyms found [FAISS, FAISS+BM25, BM25, Bi-Encoder]: {count_subject_synonyms_found[:max_length]}\n\n"
            )
            log_file.write(
                f"FAISS PubMed IDs k={max_length}: {faiss_pm_ids[:max_length]}\n"
            )
            if self.bool_bi_encoder_results:
                log_file.write(
                    f"FAISS scores k={max_length}: {faiss_scores_fmt[:max_length]}\n"
                )
                log_file.write(
                    f"Sanity check FAISS scores k={max_length}: {faiss_filtered_distances[0][:max_length]}\n"
                )
                log_file.write(
                    f"FAISS Cosine Similarity k={max_length}: {faiss_text_cosine_scores[:max_length]}\n\n"
                )
                log_file.write(
                    f"BM25 plus FAISS PubMed IDs k={max_length}: {bm25_faiss_pm_ids[:max_length]}\n\n"
                )
                log_file.write(
                    f"BM25 plus FAISS scores k={max_length}: {bm25_faiss_scores_fmt[:max_length]}\n"
                )
                log_file.write(
                    f"BM25 plus FAISS Cosine Similarity k={max_length}: {bm25_faiss_text_cosine_scores[:max_length]}\n\n"
                )
            log_file.write(
                f"BM25 PubMed IDs k={max_length}: {bm25_pm_ids[:max_length]}\n"
            )
            if self.bool_bi_encoder_results:
                log_file.write(
                    f"BM25 scores k={max_length}: {bm25_scores_fmt[:max_length]}\n"
                )
                log_file.write(
                    f"BM25 Cosine Similarity k={max_length}: {bm25_text_cosine_scores[:max_length]}\n\n"
                )
                log_file.write(
                    f"Bi-encoder PubMed IDs k={max_length}: {bi_encoder_pm_ids[:max_length]}\n"
                )
                log_file.write(
                    f"Bi-encoder scores k={max_length}: {bi_encoder_scores_fmt[:max_length]}\n"
                )

        return (
            total_objects_count,
            total_objects_found,
            total_objects_synonyms_found,
            total_pmids_count,
            total_pmids_found,
            total_pmids_with_entities_count,
            total_pmids_with_entities_found,
            total_pmids_with_entities_synonyms_count,
            total_pmids_with_entities_synonyms_found,
            count_faiss_results,
            count_bm25_faiss_results,
            count_bm25_results,
            count_faiss_es_results,
            all_found_documents_objects,
            all_found_documents_objects_synonyms,
            full_example_info,
            scores_discrepancies_faiss,
            scores_discrepancies_bm25,
            scores_discrepancies_bm25_faiss,
            total_objects_pmids_count,
            total_objects_pmids_found,
            total_objects_synonyms_pmids_found,
            count_subjects_found,
            count_subject_synonyms_found,
            total_subject_objects_synonyms_found,
            qrels,
            results_bm25,
            results_faiss,
        )

    def print_results(self):
        def convert_keys_to_strings(d):
            if isinstance(d, dict):
                return {
                    str(key): convert_keys_to_strings(value) for key, value in d.items()
                }
            else:
                return d

        def pf(percentages):
            # Pretty format
            return [f"{x*100:.{2}f}" for x in percentages]

        if self.bool_bi_encoder_results:
            model_strings = [f"Faiss k={k}" for k in self.top_k_hits]
            model_strings.extend([f"BM25 plus Faiss k={k}" for k in self.top_k_hits])
            model_strings.extend([f"BM25 k={k}" for k in self.top_k_hits])
            model_strings.extend([f"Bi-encoder k={k}" for k in self.top_k_hits])

        else:
            model_strings = [f"Faiss k={k}" for k in self.top_k_hits]
            model_strings.extend([f"BM25 k={k}" for k in self.top_k_hits])

        # The following three prints are just for debugging purposes
        print(
            f"Average number of retrieved docs (FAISS): {self.total_faiss_docs / len(self.dataset_dictionary)}"
        )
        print(
            f"Average number of retrieved docs (FAISS with corresponding abstract): {self.total_faiss_es_docs / len(self.dataset_dictionary)}"
        )
        print(
            f"Average number of retrieved docs (BM25 plus FAISS): {self.total_bm25_faiss_docs / len(self.dataset_dictionary)}"
        )
        print(
            f"Average number of retrieved docs (BM25): {self.total_bm25_docs / len(self.dataset_dictionary)}"
        )

        print(model_strings)

        log_format_strings = [
            f"\n Dataset: {self.dataset_name}",
            f"\n Data split: {self.data_split}",
            f"\nSynonyms in query: {self.synonyms_in_query}",
            f"\nFull name in query: {self.full_name_in_query}",
            f"\nNDCG with graded relevance: {self.graded_ndcg}",
            # f"Use dot product: {self.retriever.use_dot_product}",
            f"\nModels: {model_strings}",
            f"Number of gold objects: {self.number_gold_objects}",
            f"Number of found objects: {self.number_found_objects}",
            f"Recall objects: {pf(self.number_found_objects / self.number_gold_objects)}",
            f"Number of found objects with synonyms: {self.number_found_objects_synonyms}",
            f"Recall objects with synonyms: {pf(self.number_found_objects_synonyms / self.number_gold_objects)}",
            f"Number of found subject-objects combinations with synonyms: {self.number_found_subject_objects_synonyms}",
            f"Recall subject-objects combinations with synonyms: {pf(self.number_found_subject_objects_synonyms / self.number_gold_objects)}",
            f"Number of gold (objects, PMID) pairs: {self.number_gold_objects_pmids}",
            f"Number of found (objects, PMID) pairs: {self.number_found_objects_pmids}",
            f"Recall (objects, PMID) pairs: {pf(self.number_found_objects_pmids / self.number_gold_objects_pmids)}",
            f"Number of found (objects, PMID) pairs with synonyms: {self.number_found_objects_pmids_synonyms}",
            f"Recall (objects, PMID) pairs with synonyms: {pf(self.number_found_objects_pmids_synonyms / self.number_gold_objects_pmids)}",
            f"Number of gold pmids: {self.number_gold_pmids}",
            f"Number of found pmids: {self.number_found_pmids}",
            f"Recall pmids: {pf(self.number_found_pmids / self.number_gold_pmids)}",
            # f"Number of gold pmids with entities: {self.number_gold_pmids_with_entities}",
            # f"Number of found pmids with entities: {self.number_found_pmids_with_entities}",
            # f"Recall pmids with entities: {self.number_found_pmids_with_entities / self.number_gold_pmids_with_entities}",
            # f"Number of gold pmids with entities (+ object synonyms): {self.number_gold_pmids_with_entities_synonyms}",
            # f"Number of found pmids with entities (+ object synonyms): {self.number_found_pmids_with_entities_synonyms}",
            # f"Recall pmids with entities (+ object synonyms): {self.number_found_pmids_with_entities_synonyms / self.number_gold_pmids_with_entities_synonyms}",
            f"Number of gold pmids with objects: {self.number_gold_pmids_with_entities}",
            f"Number of found pmids with objects: {self.number_found_pmids_with_entities}",
            f"Recall pmids with objects: {pf(self.number_found_pmids_with_entities / self.number_gold_pmids_with_entities)}",
            f"Number of gold pmids with object synonyms: {self.number_gold_pmids_with_entities_synonyms}",
            f"Number of found pmids with object synonyms: {self.number_found_pmids_with_entities_synonyms}",
            f"Recall pmids with entities object synonyms: {pf(self.number_found_pmids_with_entities_synonyms / self.number_gold_pmids_with_entities_synonyms)}",
        ]
        # For debugging purposes, print out PMIDs found by both models, only FAISS, only BM25, and none
        # Add to log_format_strings
        # Sort discrepancies by highest first
        self.scores_discrepancies_faiss.sort(key=lambda x: x[4], reverse=True)
        self.scores_discrepancies_bm25.sort(key=lambda x: x[4], reverse=True)
        self.scores_discrepancies_bm25_faiss.sort(key=lambda x: x[4], reverse=True)
        log_format_strings.extend(
            [
                "\nDebugging found PMIDs",
                f"Both found PMIDs: {self.intersection_dict['both_found_pmids_count']}",
                f"{self.model_one_string} only found PMIDs: {self.intersection_dict[f'{self.model_one_string}_only_pmids_count']}",
                f"{self.model_two_string} only found PMIDs: {self.intersection_dict[f'{self.model_two_string}_only_pmids_count']}",
                f"None found PMIDs: {self.intersection_dict['not_found_pmids_count']}\n",
                f"Gold number of PMIDs (unique): {self.intersection_dict['gold_pmids_count']}",
                f"Both found PMIDs (unique): {len(self.intersection_dict['both_found_pmids_set'])}",
                f"{self.model_one_string} only found PMIDs (unique): {len(self.intersection_dict[f'{self.model_one_string}_only_pmids_set'])}",
                f"{self.model_two_string} only found PMIDs (unique): {len(self.intersection_dict[f'{self.model_two_string}_only_pmids_set'])}",
                f"None found PMIDs (unique): {len(self.intersection_dict['not_found_pmids_set'])}\n",
                f"Both found PMIDs: {self.intersection_dict['both_found_pmids']}\n",
                f"{self.model_one_string} only found PMIDs: {self.intersection_dict[f'{self.model_one_string}_only_pmids']}\n",
                f"{self.model_two_string} only found PMIDs: {self.intersection_dict[f'{self.model_two_string}_only_pmids']}\n",
                f"None found PMIDs: {self.intersection_dict['not_found_pmids']}\n",
                f"Score discrepancies sorted by highest first (index_number, pmid, gold_score, model_score, score_difference, found_by_model): \n"
                f"FAISS discrepancies: {self.scores_discrepancies_faiss}\n",
                f"BM25 discrepancies: {self.scores_discrepancies_bm25}\n",
                f"BM25+FAISS discrepancies: {self.scores_discrepancies_bm25_faiss}\n",
            ]
        )
        # Debug found objects split as follows:
        # Found by FAISS only, BM25 only and both
        bm_25_found_objects = defaultdict(list)
        faiss_found_objects = defaultdict(list)
        comparison_index = 1  # Hard-coded, TODO: Change this
        for example_index, found_documents in self.all_documents_found:
            for found_model_documents in found_documents:
                for found_document in found_model_documents:
                    if found_document[3] == comparison_index:
                        faiss_found_objects[(example_index, found_document[1])].append(
                            found_document[0]
                        )
                    elif found_document[3] == comparison_index + 2 * len(
                        self.top_k_hits
                    ):
                        bm_25_found_objects[(example_index, found_document[1])].append(
                            found_document[0]
                        )

        faiss_found_objects_set = set(faiss_found_objects.keys())
        bm_25_found_objects_set = set(bm_25_found_objects.keys())
        faiss_only_found_objects_set = faiss_found_objects_set - bm_25_found_objects_set
        bm25_only_found_objects_set = bm_25_found_objects_set - faiss_found_objects_set
        both_found_objects_set = faiss_found_objects_set.intersection(
            bm_25_found_objects_set
        )

        faiss_only_found_objects = {
            key: faiss_found_objects[key] for key in faiss_only_found_objects_set
        }
        bm25_only_found_objects = {
            key: bm_25_found_objects[key] for key in bm25_only_found_objects_set
        }
        both_found_objects = {
            key: faiss_found_objects[key] for key in both_found_objects_set
        }

        log_format_strings.extend(
            [
                "\nDebugging found objects",
                f"Both found objects: {len(both_found_objects)}",
                f"FAISS only found objects: {len(faiss_only_found_objects)}",
                f"BM25 only found objects: {len(bm25_only_found_objects)}\n",
                f"Both found objects: {both_found_objects}\n",
                f"FAISS only found objects: {faiss_only_found_objects}\n",
                f"BM25 only found objects: {bm25_only_found_objects}\n",
            ]
        )

        # Debug found subject mentions to evaluate model performance
        # Add to log_format_strings
        log_format_strings.extend(
            [
                "\nDebugging found subject mentions",
                f"Number of subjects found: {self.average_number_found_subjects}",
                f"Number of subject synonyms found: {self.average_number_found_subject_synonyms}\n",
            ]
        )

        for format_string in log_format_strings:
            print(format_string)

        with open(self.result_file, "a") as log_file:
            for format_string in log_format_strings:
                log_file.write(format_string + "\n")

            # Further metrics for PMID extraction only
            # For drugs/catalyst extraction, only the first document found is considered for the recall
            log_file.write("\nFurther metrics for PMID extraction only")
            log_file.write("\nGold PMIDs")
            log_file.write("\n  BM25")
            beir_evaluate(
                self.qrels["pmids"], self.results_bm25, self.top_k_hits, log_file
            )
            log_file.write("\n  FAISS")
            beir_evaluate(
                self.qrels["pmids"], self.results_faiss, self.top_k_hits, log_file
            )
            log_file.write("\nGold PMIDs with entities")
            log_file.write("\n  BM25")
            beir_evaluate(
                self.qrels["pmids_with_entities"],
                self.results_bm25,
                self.top_k_hits,
                log_file,
            )
            log_file.write("\n  FAISS")
            beir_evaluate(
                self.qrels["pmids_with_entities"],
                self.results_faiss,
                self.top_k_hits,
                log_file,
            )
            log_file.write("\nGold PMIDs with entities and synonyms")
            log_file.write("\n  BM25")
            beir_evaluate(
                self.qrels["pmids_with_entities_synonyms"],
                self.results_bm25,
                self.top_k_hits,
                log_file,
            )
            log_file.write("\n  FAISS")
            beir_evaluate(
                self.qrels["pmids_with_entities_synonyms"],
                self.results_faiss,
                self.top_k_hits,
                log_file,
            )

        # Save faiss_example_subjects_found_pmid to .json file
        json_results = convert_keys_to_strings(self.faiss_example_subjects_found_pmid)
        # Change tuple keys to strings

        with open(
            f"{self.file_prefix}faiss_example_subjects_found_pmid_{self.model_abbreviation}_{self.data_split}.json",
            "w",
        ) as json_file:
            json.dump(json_results, json_file)


if __name__ == "__main__":
    from datasets.utils import logging as dataset_logging

    from evaluation.model_list import MODEL_LIST

    # from models.retriever import Retriever
    from po_datasets.civic import CiVICExamples
    from po_datasets.concat_dataset import ConcatExamples
    from po_datasets.onco_kb import OncoKBExamples
    from po_datasets.uniprot_ptms import UniProtPTMExamples

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=11)
    parser.add_argument("--mode", type=str, default="raw_text")
    parser.add_argument(
        "--use_dot_product",
        action="store_true",
        help="Use dot product instead of cosine for retrieval",
    )
    parser.add_argument(
        "--only_use_flat_index",
        action="store_true",
        help="Only use flat index for retrieval",
        default=True,
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        help="Pooling strategy for sentence transformer",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        help="Data split to use for evaluation",
    )
    parser.add_argument("--data_set", type=str, default="civic")
    parser.add_argument(
        "--results_file",
        type=str,
        default="retriever_results.log",
        help="File to write results to",
    )
    parser.add_argument(
        "--top_k_hits",
        type=list,
        default=[10, 50, 200, 1000],
        help="Number of hits to retrieve from FAISS",
    )
    parser.add_argument(
        "--synonyms_in_query",
        action="store_true",
        help="Use synonyms in query",
    )
    parser.add_argument(
        "--full_name_in_query",
        action="store_true",
        help="Use full name in query",
    )
    parser.add_argument(
        "--use_asym_bi_encoder",
        action="store_true",
        help="Use asymmetric bi-encoder",
    )
    parser.add_argument(
        "--graded_ndcg",
        action="store_true",
        help="Use graded NDCG",
    )
    args = parser.parse_args()

    model_index = MODEL_LIST[args.model_index]

    retriever = BiEncoderRetrieverEvaluator(
        bi_encoder_articles=model_index[0],
        bi_encoder_queries=model_index[5] if len(model_index) == 6 else model_index[0],
        document_index=model_index[1],
        flat_index_file=model_index[2],
        flat_index_mapping_file=model_index[3],
        pooling="cls",
        use_asym_bi_encoder=args.use_asym_bi_encoder,
        use_dot_product=args.use_dot_product,
        only_use_flat_index=True,
        top_k_hits=args.top_k_hits,
        es_index="20231127_pubmed",
    )

    civic_dataset_samples = CiVICExamples(mode=args.mode)
    oncokb_dataset_samples = OncoKBExamples(mode=args.mode)
    samples_split = ConcatExamples(
        [
            civic_dataset_samples,
            oncokb_dataset_samples,
        ],
        mode=args.mode,
    )

    if args.data_set == "civic":
        dataset_dict_train, _, _ = get_dataset_dict(
            civic_dataset_samples, datasplit=args.data_split, split_examples=samples_split
        )
    elif args.data_set == "oncokb":
        dataset_dict_train, _, _ = get_dataset_dict(
            oncokb_dataset_samples, datasplit=args.data_split, split_examples=samples_split
        )
    elif args.data_set == "civic_oncokb":
        dataset_dict_train, _, _ = get_dataset_dict(
            samples_split, datasplit=args.data_split, split_examples=samples_split
        )
    elif args.data_set == "onkopedia":
        dataset_dict_train = get_dataset_dict_from_csv("onkopedia_guidelines.csv")
    elif args.data_set == "uniprot":
        uniprot_samples = UniProtPTMExamples(mode=args.mode)
        dataset_dict_train, _, _ = get_dataset_dict_uniprot(
            uniprot_samples, datasplit=args.data_split
        )
    else:
        raise ValueError("Invalid dataset")

    # dataset_logging.disable_progress_bar()

    dataset_evaluator = DatasetEvaluator(
        retriever,
        args.data_set,
        DRUG_KEYWORDS,
        dataset_dict_train,
        file_prefix="data/retriever_results/",
        results_file=args.results_file,
        bool_bi_encoder_results=True,
        synonyms_in_query=args.synonyms_in_query,
        full_name_in_query=args.full_name_in_query,
        data_split=args.data_split,
        compare_models=["faiss", "bm25"],
        model_name=model_index[0],
        model_abbreviation=model_index[4],
        top_k_hits=args.top_k_hits,
        graded_ndcg=args.graded_ndcg,
    )
