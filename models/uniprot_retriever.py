import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from datasets import Dataset
from tqdm import tqdm

import models.margin_config_uniprot as margin_config
from models.retriever import Retriever
from models.transformers import InputExample
from po_datasets.dataset import ElasticsearchHelper
from po_datasets.uniprot_dictionaries import ALL_PTM_SYNONYMS

random.seed(42)


class UniprotRetriever(Retriever):
    def __init__(
        self,
        substrate_synonyms: bool = False,
        catalysts_synonyms: bool = False,
        full_name_in_query: bool = False,
        filter_no_catalysts: bool = True,
        filter_seen_pmids: bool = True,
        *args,
        **kwargs,
    ):
        # Create input examples
        # TODO: These are not really batch negatives, but rather random negatives
        self.use_substrate_synonyms = "_synonyms" if substrate_synonyms else ""
        self.use_catalysts_synonyms = "_synonyms" if catalysts_synonyms else ""
        self.full_name_in_query = full_name_in_query
        self.filter_no_catalysts = filter_no_catalysts

        self.examples_count_dict = {
            "count_positive": 0,
            "count_positive_three_entities_catalyst_match": 0,
            "count_positive_three_entities_no_catalyst_match": 0,
            "count_positive_ptm_catalyst_match": 0,
            "count_positive_other_two_entities_match": 0,
            "count_positive_substrate_match": 0,
            "count_positive_other_one_entity_match": 0,
            "count_positive_no_entities_match": 0,
            # Any catalyst matches
            "count_negative_same_substrate_ptm_res_no_other_pos_any_catalyst_match": 0,
            "count_negative_same_substrate_ptm_res_other_pos_any_catalyst_match": 0,
            "count_negative_same_substrate_ptm_no_other_respos_any_catalyst_match": 0,
            "count_negative_same_substrate_ptm_other_res_no_other_pos_any_catalyst_match": 0,
            "count_negative_same_substrate_ptm_other_respos_any_catalyst_match": 0,
            "count_negative_same_substrate_no_other_ptm_respos_any_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_no_other_respos_any_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_res_no_other_pos_any_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_respos_any_catalyst_match": 0,
            "count_negative_same_no_substrate_any_catalyst_match": 0,
            # No catalyst matches
            "count_negative_same_substrate_ptm_res_no_other_pos_no_catalyst_match": 0,
            "count_negative_same_substrate_ptm_res_other_pos_no_catalyst_match": 0,
            "count_negative_same_substrate_ptm_no_other_respos_no_catalyst_match": 0,
            "count_negative_same_substrate_ptm_other_res_no_other_pos_no_catalyst_match": 0,
            "count_negative_same_substrate_ptm_other_respos_no_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_no_other_respos_no_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_res_no_other_pos_no_catalyst_match": 0,
            "count_negative_same_substrate_other_ptm_respos_no_catalyst_match": 0,
            "count_negative_same_no_substrate_no_catalyst_match": 0,
            # Other substrate matches
            "count_negative_other_substrate_any_catalyst_match": 0,
            "count_negative_other_substrate_no_catalyst_match": 0,
            # BM25 negatives
            "count_negative_same_substrate_not_uniprot_bm25": 0,
            # PubMed negatives
            "count_negative_pubmed": 0,
        }

        self.non_matching_positive_label = 1  # 1 is positive example, 0 is negative
        self.check_for_seen_pmids = filter_seen_pmids
        self.currently_seen_pmids = set()
        self.seen_pmid_duplicates = 0

        super().__init__(*args, **kwargs)

        if "negative_pubmed" in self.margin_value_dict:
            self.add_pubmed_documents()

        self.bm25_query_cache_file = "edel_repo_cache/treatment_explorer_bm25_query_cache.json"
        if Path(self.bm25_query_cache_file).exists():
            with open(self.bm25_query_cache_file, "r") as f:
                self.bm25_query_cache = json.load(f)
        else:
            self.bm25_query_cache = {}

        self.generate_examples()

        with open(self.bm25_query_cache_file, "w") as f:
            json.dump(self.bm25_query_cache, f)

        # Load self.seen_pmid_duplicates from file if cache and file exists
        misc_cache_path = (
            self.cache_dir + "/" + self.cache_file_prefix + "_seen_pmids.txt"
        )
        if self.cache and Path(misc_cache_path).exists():
            with open(
                misc_cache_path,
                "r",
            ) as f:
                self.seen_pmid_duplicates = int(f.read())
        elif self.cache:
            # Dump the seen pmids to a file
            with open(misc_cache_path, "w") as f:
                f.write(str(self.seen_pmid_duplicates))

    def add_previous_faiss_examples(self, example_dict: dict):
        # Just do a no-op
        pass

    def get_negative_examples_other_substrate(
        self,
        dataset: Dataset,
        substrate_name: str,
        substrate_full_name: str,
        substrate_synonyms: List[str],
        ptm_residue_combinations: List[Tuple[str, str, str]],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
        seed: int = 42,
    ) -> List[InputExample]:
        input_examples = []
        shuffled_dataset = dataset.shuffle(seed=seed)
        text_type = "evidence_" + self.mode
        # Since cosine function is not linear, we need to adjust the margin

        for example in shuffled_dataset:
            if len(input_examples) > max_number:
                break

            if len(example[text_type]) > 3:
                for ptm, residue, position in ptm_residue_combinations:
                    if (substrate_name, ptm, residue, position) in self.query_id_dict:
                        query_id = self.query_id_dict[
                            (substrate_name, ptm, residue, position)
                        ]
                    else:
                        self.current_query_id += 1
                        query_id = self.current_query_id
                        self.query_id_dict[(substrate_name, ptm, residue, position)] = (
                            query_id
                        )
                    query = get_uniprot_retriever_query(
                        substrate_name,
                        substrate_full_name,
                        substrate_synonyms,
                        ptm,
                        residue,
                        position,
                        self.synonyms_in_query,
                        self.full_name_in_query,
                    )
                    input_examples.append(
                        InputExample(
                            texts=[query, example[text_type]],
                            label=label,
                            margin=self.margin_class.margin_fn(margin),
                            query_id=query_id,
                            doc_id=int(example["citation_id"]),
                        )
                    )

        if len(input_examples) > max_number:
            input_examples = random.sample(
                input_examples,
                max_number,
            )

        return input_examples

    def get_negative_examples_bm25(
        self,
        substrate_name: str,
        substrate_full_name: str,
        substrate_synonyms: List[str],
        ptm_residue_combinations: List[Tuple[str, str, str]],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
    ) -> List[InputExample]:
        input_examples = []
        all_seen_pmids = dict()

        query_key = "_".join(
            ["uniprot", substrate_name, substrate_full_name]
        )
        if query_key in self.bm25_query_cache:
            query_pmids = self.bm25_query_cache[query_key]
            for pmid in query_pmids:
                all_seen_pmids[pmid] = self.bm25_query_cache["pmid_" + pmid]
        else:
            results = ElasticsearchHelper.query_keywords(
                keyword_lists=[[substrate_name, substrate_full_name]],
                not_keyword_lists=[ALL_PTM_SYNONYMS],
                query_term="term",
                number=max_number,
            )
            for result in results[:max_number]:
                example_text = result["_source"]["text"]
                pmid = result["_source"]["pmid"]
                if pmid not in all_seen_pmids:
                    all_seen_pmids[pmid] = example_text
                self.bm25_query_cache["pmid_" + pmid] = example_text
            self.bm25_query_cache[query_key] = list(all_seen_pmids.keys())

        for ptm, residue, position in ptm_residue_combinations:
            if (substrate_name, ptm, residue, position) in self.query_id_dict:
                query_id = self.query_id_dict[(substrate_name, ptm, residue, position)]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[(substrate_name, ptm, residue, position)] = query_id
            query = get_uniprot_retriever_query(
                substrate_name,
                substrate_full_name,
                substrate_synonyms,
                ptm,
                residue,
                position,
                self.synonyms_in_query,
                self.full_name_in_query,
            )
            for pmid, example_text in all_seen_pmids.items():
                input_examples.append(
                    InputExample(
                        texts=[query, example_text],
                        label=label,
                        margin=self.margin_class.margin_fn(margin),
                        query_id=query_id,
                        doc_id=int(pmid),
                    )
                )

        if len(input_examples) > max_number:
            input_examples = random.sample(
                input_examples,
                max_number,
            )

        return input_examples

    def get_negative_examples_pubmed(
        self,
        substrate_name: str,
        substrate_full_name: str,
        substrate_synonyms: List[str],
        ptm_residue_combinations: List[Tuple[str, str, str]],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
        seed: int = 42,
    ) -> List[InputExample]:
        input_examples = []

        for ptm, residue, position in ptm_residue_combinations:
            if (substrate_name, ptm, residue, position) in self.query_id_dict:
                query_id = self.query_id_dict[(substrate_name, ptm, residue, position)]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[(substrate_name, ptm, residue, position)] = query_id
            query = get_uniprot_retriever_query(
                substrate_name,
                substrate_full_name,
                substrate_synonyms,
                ptm,
                residue,
                position,
                self.synonyms_in_query,
                self.full_name_in_query,
            )
            # TODO Check that entity does not occur in the selected PubMed text
            # Draw random sample of PMIDs in all PubMed texts
            current_pmids = self.all_pmids_texts.sample(n=max_number, seed=seed)
            for i, example_text in enumerate(current_pmids["text"]):
                input_examples.append(
                    InputExample(
                        texts=[query, example_text],
                        label=label,
                        margin=self.margin_class.margin_fn(margin),
                        query_id=query_id,
                        doc_id=int(current_pmids["pmid"][i]),
                    )
                )

        if len(input_examples) > max_number:
            input_examples = random.sample(
                input_examples,
                max_number,
            )

        return input_examples

    def match_entities_in_text(self, example: Dict, col_names: Dict[str, bool]) -> Dict:
        entity_type_matching_dict = {}
        entity_types = ["substrate", "ptm_type", "residue", "position", "catalysts"]
        pattern = "|".join(re.escape(entity_type) for entity_type in entity_types)
        for col_name in col_names.keys():
            match = re.search(pattern, col_name)
            if match:
                entity_type = match.group()
                if col_names[col_name]:
                    entity_type_matching_dict[entity_type] = any(example[col_name])
                else:
                    entity_type_matching_dict[entity_type] = not any(example[col_name])
                assert isinstance(example[col_name][0], bool)
            else:
                raise ValueError("This should not happen")
        return entity_type_matching_dict

    def get_examples(
        self,
        dataset: Dataset,
        col_names: Dict[str, bool],
        query: str,
        label: int,
        margin: float = 0.0,
        process_non_matching_texts: bool = True,
        max_number: float = np.inf,
        seed: int = 42,
        query_id: int = -1,
    ) -> Tuple[List[InputExample], List[InputExample]]:
        input_examples = []
        non_matching_examples = {}
        shuffled_dataset = dataset.shuffle(seed=seed)
        text_type = "evidence_" + self.mode
        # Since cosine function is not linear, we need to adjust the margin
        for i, example in enumerate(shuffled_dataset):
            if i >= max_number:
                break
            col_names_check = {
                f"{col_key}_in_{self.mode}": col_value
                for col_key, col_value in col_names.items()
            }
            matching_dict = self.match_entities_in_text(example, col_names_check)

            entities_in_text = sum(matching_dict.values()) == len(col_names)

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

            if (
                entities_in_text
                and len(example[text_type]) > 3
                and (example["citation_id"] not in self.currently_seen_pmids or not self.check_for_seen_pmids)
            ):
                input_examples.append(
                    InputExample(
                        texts=[query, example[text_type]],
                        label=label,
                        margin=self.margin_class.margin_fn(margin),
                        query_id=query_id,
                        doc_id=int(example["citation_id"]),
                    )
                )
                self.currently_seen_pmids.add(example["citation_id"])
            elif (
                self.use_non_matching_texts
                and process_non_matching_texts
                and not entities_in_text
                and len(example[text_type]) > 3
                and (example["citation_id"] not in self.currently_seen_pmids or not self.check_for_seen_pmids)
            ):
                margin_class_string = ""
                if (sum(matching_dict.values()) == 3) and (matching_dict["catalysts"]):
                    margin_class_string = "positive_three_entities_catalyst_match"
                elif (sum(matching_dict.values()) == 3) and (
                    not matching_dict["catalysts"]
                ):
                    margin_class_string = "positive_three_entities_no_catalyst_match"
                elif (
                    matching_dict["substrate"]
                    and matching_dict["catalysts"]
                    and sum(matching_dict.values()) == 2
                ):
                    margin_class_string = "positive_substrate_catalyst_match"
                elif (
                    matching_dict["ptm_type"]
                    and matching_dict["catalysts"]
                    and sum(matching_dict.values()) == 2
                ):
                    margin_class_string = "positive_ptm_catalyst_match"
                elif sum(matching_dict.values()) == 2:
                    margin_class_string = "positive_other_two_entities_match"
                elif matching_dict["substrate"]:
                    margin_class_string = "positive_substrate_match"
                elif sum(matching_dict.values()) == 1:
                    margin_class_string = "positive_other_one_entity_match"
                elif sum(matching_dict.values()) == 0:
                    margin_class_string = "positive_no_entities_match"
                else:
                    raise ValueError("This should not happen")

                # New with uniprot_v13: Make queries more substrate centric
                if (
                    "positive_three_entities_substrate_match"
                    in self.margin_class.margin_value_dict
                ):
                    if (sum(matching_dict.values()) == 3) and (
                        matching_dict["substrate"]
                    ):
                        margin_class_string = "positive_three_entities_substrate_match"
                    elif (
                        (sum(matching_dict.values()) == 2)
                        and (matching_dict["substrate"])
                        and (matching_dict["respos"])
                    ):
                        margin_class_string = "positive_substrate_respos_match"
                    elif (
                        (sum(matching_dict.values()) == 2)
                        and (matching_dict["substrate"])
                        and (matching_dict["ptm_type"])
                    ):
                        margin_class_string = "positive_substrate_ptm_match"
                    elif (sum(matching_dict.values()) == 3) and (
                        not matching_dict["substrate"]
                    ):
                        margin_class_string = (
                            "positive_three_entities_no_substrate_match"
                        )
                    elif (sum(matching_dict.values()) == 2) and (
                        not matching_dict["substrate"]
                    ):
                        margin_class_string = "positive_two_entities_no_substrate_match"

                # New with uniprot_v32: Make queries more substrate centric
                if (
                    "positive_three_entities_substrate_catalyst_match"
                    in self.margin_class.margin_value_dict
                ):
                    if (
                        (sum(matching_dict.values()) == 3)
                        and matching_dict["substrate"]
                        and matching_dict["catalysts"]
                    ):
                        margin_class_string = "positive_three_entities_substrate_catalyst_match"
                    elif (
                        (sum(matching_dict.values()) == 3)
                        and matching_dict["substrate"]
                    ):
                        margin_class_string = "positive_three_entities_substrate_no_catalyst_match"
                    elif (sum(matching_dict.values()) == 3) and (
                        not matching_dict["substrate"]
                    ):
                        margin_class_string = (
                            "positive_three_entities_no_substrate_match"
                        )
                    elif (
                        (sum(matching_dict.values()) == 2)
                        and (matching_dict["substrate"])
                        and not (matching_dict["catalysts"])
                    ):
                        margin_class_string = "positive_two_entities_substrate_no_catalyst_match"
                    elif (sum(matching_dict.values()) == 2) and (
                        not matching_dict["substrate"]
                    ):
                        margin_class_string = "positive_two_entities_no_substrate_match"

                if margin_class_string not in self.margin_class.margin_value_dict:
                    continue

                non_matching_examples.setdefault(margin_class_string, [])
                non_matching_examples[margin_class_string].append(
                    InputExample(
                        texts=[query, example[text_type]],
                        label=self.non_matching_positive_label,
                        margin=self.margin_class.margin_fn(
                            self.margin_value_dict[margin_class_string]
                        ),
                        query_id=query_id,
                        doc_id=int(example["citation_id"]),
                    )
                )
                self.currently_seen_pmids.add(example["citation_id"])
            elif example["citation_id"] in self.currently_seen_pmids:
                self.seen_pmid_duplicates += 1
        return input_examples, non_matching_examples

    def get_negative_examples_with_margin_class_string(
        self,
        substrate_dataset: Dataset,
        col_names: Dict[str, bool],
        margin_class_string: str,
        query: str,
        current_ptm_type: str,
        current_residue: str,
        current_position: str,
        bool_ptm_type: bool = False,
        bool_residue: bool = False,
        bool_position: bool = False,
        bool_catalysts: bool = False,
        num_positives: int = 0,
        seed: int = 42,
        query_id: int = -1,
    ) -> List[InputExample]:

        if margin_class_string not in self.margin_class.margin_value_dict:
            return []

        if "no_substrate" in margin_class_string:
            negative_dataset = substrate_dataset.filter(
                lambda x: (len(x["catalysts"]) > 0) is bool_catalysts
            )
        elif "ptm_any_catalyst" in margin_class_string or "ptm_no_catalyst" in margin_class_string:
            negative_dataset = substrate_dataset.filter(
                lambda x: (x["ptm_type"] == current_ptm_type) is bool_ptm_type
                and (len(x["catalysts"]) > 0) is bool_catalysts
            )
        else:
            negative_dataset = substrate_dataset.filter(
                lambda x: (x["ptm_type"] == current_ptm_type) is bool_ptm_type
                and (x["residue"] == current_residue) is bool_residue
                and (x["position"] == current_position) is bool_position
                and (len(x["catalysts"]) > 0) is bool_catalysts
            )
        negative_examples, _ = self.get_examples(
            negative_dataset,
            col_names=col_names,
            query=query,
            label=0,
            margin=self.margin_value_dict[margin_class_string],
            process_non_matching_texts=False,
            seed=seed,
            query_id=query_id,
        )
        if self.filter_no_catalysts and len(negative_examples) > min(
            num_positives * self.max_ratio_negatives, self.max_negatives
        ):
            negative_examples = random.sample(
                negative_examples,
                min(num_positives * self.max_ratio_negatives, self.max_negatives),
            )
        elif (
            not self.filter_no_catalysts
            and len(negative_examples) > int(self.max_negatives / 2)
            # and margin_class_string
            # not in [
            #     "negative_same_no_substrate_any_catalyst_match",
            #     "negative_same_no_substrate_no_catalyst_match",
            # ]
        ):
            negative_examples = random.sample(
                negative_examples, int(self.max_negatives / 2)
            )
        # elif (
        #     not self.filter_no_catalysts
        #     and len(negative_examples) > int(self.max_negatives / 2)
        #     and margin_class_string
        #     in [
        #         "negative_same_no_substrate_any_catalyst_match",
        #         "negative_same_no_substrate_no_catalyst_match",
        #     ]
        # ):
        #     negative_examples = random.sample(
        #         negative_examples, int(self.max_negatives / 2)
        #     )
        self.examples_count_dict[f"count_{margin_class_string}"] += len(
            negative_examples
        )
        return negative_examples

    def get_positive_and_hard_negative_examples(
        self, dataset: Dataset
    ) -> Tuple[List[InputExample], int]:
        input_examples = []
        substrate_col = "substrate" + self.use_substrate_synonyms
        catalysts_col = "catalysts" + self.use_catalysts_synonyms
        uniprot_count_pos = 0
        uniprot_id = dataset[0]["primary_accession"]
        # Extract numbers from the uniprot_id
        seed = int(re.findall(r"\d+", uniprot_id)[0])

        # Get unique ptm_type, residue, position combinations for the uniprot_id
        if self.filter_no_catalysts:
            ptm_residue_position_combinations = set(
                [
                    (x["ptm_type"], x["residue"], x["position"])
                    for x in dataset
                    if (len(x["catalysts"]) > 0)
                ]
            )
        else:
            ptm_residue_position_combinations = set(
                [(x["ptm_type"], x["residue"], x["position"]) for x in dataset]
            )

        for (
            current_ptm_type,
            current_residue,
            current_position,
        ) in ptm_residue_position_combinations:

            # Reset the currently seen pmids
            self.currently_seen_pmids = set()

            substrate_name = dataset[0]["substrate"]
            if (
                substrate_name,
                current_ptm_type,
                current_residue,
                current_position,
            ) in self.query_id_dict:
                query_id = self.query_id_dict[
                    (
                        substrate_name,
                        current_ptm_type,
                        current_residue,
                        current_position,
                    )
                ]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[
                    (
                        substrate_name,
                        current_ptm_type,
                        current_residue,
                        current_position,
                    )
                ] = query_id
            query = get_uniprot_retriever_query(
                dataset[0]["substrate"],
                dataset[0]["substrate_full_name"],
                dataset[0]["substrate_synonyms"],
                current_ptm_type,
                current_residue,
                current_position,
                self.synonyms_in_query,
                self.full_name_in_query,
            )

            positive_dataset = dataset.filter(
                lambda x: x["ptm_type"] == current_ptm_type
                and x["residue"] == current_residue
                and x["position"] == current_position
                and len(x["catalysts"]) > 0
            )

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
                catalysts_col: True,
            }
            # Get positive examples for same gene and same variant
            positive_examples, non_matching_examples_dict = self.get_examples(
                positive_dataset,
                column_dict,
                query=query,
                label=1,
                margin=self.margin_value_dict["positive"],
                seed=seed,
                query_id=query_id,
            )
            uniprot_count_pos += len(positive_examples)
            input_examples.extend(positive_examples)
            self.examples_count_dict["count_positive"] += len(positive_examples)

            number_all_positive_examples = len(positive_examples)
            for (
                _,
                non_matching_examples,
            ) in non_matching_examples_dict.items():
                number_all_positive_examples += len(non_matching_examples)
                # Newly added 12/11/24
                if self.all_negatives:
                    uniprot_count_pos += len(non_matching_examples)

            # These are negative examples with entities and their synonyms
            # not matching any text
            # TODO: Exclude examples where a synonym of an entity is actually in the text
            # TODO: These examples may be noisy. Check the performance and
            # remove them, if necessary.
            if (
                self.use_non_matching_texts
                #     and self.use_substrate_synonyms
                #     and self.use_variant_synonyms
                #     and self.use_catalysts_synonyms
            ):
                for (
                    margin_class_string,
                    non_matching_examples,
                ) in non_matching_examples_dict.items():
                    input_examples.extend(non_matching_examples)
                    self.examples_count_dict[f"count_{margin_class_string}"] += len(
                        non_matching_examples
                    )

            if self.use_batch_negatives or self.use_random_negatives:
                continue

            column_dict = {
                substrate_col: False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_no_substrate_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_no_substrate_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            ## Added with uniprot_margin_classes_v23

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=True,
                bool_ptm_type=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                # "ptm_type": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=True,
                bool_ptm_type=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=False,
                bool_ptm_type=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                # "ptm_type": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_catalysts=False,
                bool_ptm_type=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            ##########

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": False,
                "position": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_no_other_respos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_res_no_other_pos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_res_no_other_pos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=True,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_res_other_pos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=True,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_other_res_no_other_pos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_other_res_no_other_pos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_res_no_other_pos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=True,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_res_other_pos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=True,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": False,
                "position": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_no_other_respos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": True,
                "position": True,
                "ptm_type": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_respos_no_other_ptm_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=True,
                bool_position=True,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_other_respos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": True,
                "position": True,
                "ptm_type": True,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_respos_other_ptm_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=True,
                bool_position=True,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": False,
                "position": False,
                "ptm_type": False,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_no_other_ptm_respos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": True,
                "position": True,
                "ptm_type": True,
                # catalysts_col: True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_respos_any_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=True,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_respos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=True,
                bool_position=True,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": True,
                "position": True,
                "ptm_type": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_respos_no_other_ptm_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=True,
                bool_position=True,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": False,
                "position": False,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_no_other_respos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "residue": True,
                "position": True,
                "ptm_type": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_respos_other_ptm_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=True,
                bool_position=True,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_ptm_other_respos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=True,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

            column_dict = {
                substrate_col: True,
                "ptm_type": True,
                "residue": True,
                "position": True,
            }
            negative_examples = self.get_negative_examples_with_margin_class_string(
                dataset,
                column_dict,
                "negative_same_substrate_other_ptm_respos_no_catalyst_match",
                query,
                current_ptm_type,
                current_residue,
                current_position,
                bool_ptm_type=False,
                bool_residue=False,
                bool_position=False,
                bool_catalysts=False,
                num_positives=number_all_positive_examples,
                seed=seed,
                query_id=query_id,
            )
            input_examples.extend(negative_examples)

        return (input_examples, uniprot_count_pos, ptm_residue_position_combinations)

    def sample_generator_for_retriever(
        self,
        dataset: Dataset,
        # split: Union[np.ndarray[Any, Any], List[int]],  # Requires Python >= 3.9
        split: Union[np.ndarray, List[int]],
        split_name: str,
    ) -> Tuple[List[InputExample], List[List[InputExample]]]:

        input_examples = []

        # tmp_counter = 0

        for uniprot_id in tqdm(split, desc="Generating examples"):
            # tmp_counter += 1
            # if tmp_counter % 20 == 0:
            #     break

            uniprot_dataset = dataset.filter(
                lambda x: x["primary_accession"] == uniprot_id
            )

            # Iterate over the uniprot_dataset and check if any of the
            # entries contain a catalyst
            # If not, continue
            catalysts_found = False
            for entry in uniprot_dataset:
                if len(entry["catalysts"]) > 0:
                    catalysts_found = True
                    break
            if not self.filter_no_catalysts:
                catalysts_found = True
            if not catalysts_found:
                continue

            (
                examples,
                count_pos,
                ptm_residue_position_combinations,
            ) = self.get_positive_and_hard_negative_examples(uniprot_dataset)

            uniprot_id = uniprot_dataset[0]["primary_accession"]
            seed = int(re.findall(r"\d+", uniprot_id)[0])

            if len(examples) > 0 and self.use_supervised_examples:
                input_examples.extend(examples)

                if self.use_batch_negatives or self.use_random_negatives:
                    continue

                if self.filter_no_catalysts:
                    # if self.all_negatives:
                    #     max_number = count_pos * int(self.max_ratio_negatives / 4)
                    # else:
                    max_number = min(
                        count_pos * self.max_ratio_negatives, self.max_negatives
                    )
                    # max_number = count_pos * int(self.max_ratio_negatives / 2)
                else:
                    max_number = int(self.max_negatives / 3)

                # Get negative examples for other genes and no treatment
                other_uniprot_dataset_no_treatment = dataset.filter(
                    lambda x: x["primary_accession"] != uniprot_id
                    and len(x["catalysts"]) == 0
                )

                if (
                    "negative_other_substrate_no_catalyst_match"
                    in self.margin_value_dict
                ):
                    samples = self.get_negative_examples_other_substrate(
                        other_uniprot_dataset_no_treatment,
                        uniprot_dataset[0]["substrate"],
                        uniprot_dataset[0]["substrate_full_name"],
                        uniprot_dataset[0]["substrate_synonyms"],
                        ptm_residue_position_combinations,
                        label=0,
                        margin=self.margin_value_dict[
                            "negative_other_substrate_no_catalyst_match"
                        ],
                        max_number=max_number,
                        seed=seed,
                    )
                    input_examples.extend(samples)
                    self.examples_count_dict[
                        "count_negative_other_substrate_no_catalyst_match"
                    ] += len(samples)

                if (
                    "negative_other_substrate_any_catalyst_match"
                    in self.margin_value_dict
                ):
                    # Get negative examples for other genes and any treatment
                    other_uniprot_dataset_any_treatment = dataset.filter(
                        lambda x: x["primary_accession"] != uniprot_id
                        and len(x["catalysts"]) > 0
                    )

                    samples = self.get_negative_examples_other_substrate(
                        other_uniprot_dataset_any_treatment,
                        uniprot_dataset[0]["substrate"],
                        uniprot_dataset[0]["substrate_full_name"],
                        uniprot_dataset[0]["substrate_synonyms"],
                        ptm_residue_position_combinations,
                        label=0,
                        margin=self.margin_value_dict[
                            "negative_other_substrate_any_catalyst_match"
                        ],
                        max_number=max_number,
                        seed=seed,
                    )

                    input_examples.extend(samples)
                    self.examples_count_dict[
                        "count_negative_other_substrate_any_catalyst_match"
                    ] += len(samples)

                # Get negative examples same substrate not Uniprot from BM25
                if "negative_same_substrate_not_uniprot_bm25" in self.margin_value_dict:
                    samples = self.get_negative_examples_bm25(
                        uniprot_dataset[0]["substrate"],
                        uniprot_dataset[0]["substrate_full_name"],
                        uniprot_dataset[0]["substrate_synonyms"],
                        ptm_residue_position_combinations,
                        label=0,
                        margin=self.margin_value_dict[
                            "negative_same_substrate_not_uniprot_bm25"
                        ],
                        max_number=max_number,
                    )

                    input_examples.extend(samples)
                    self.examples_count_dict[
                        "count_negative_same_substrate_not_uniprot_bm25"
                    ] += len(samples)

                # print(self.margin_value_dict)
                # exit()

                # Get negative examples from PubMed
                if "negative_pubmed" in self.margin_value_dict:
                    samples = self.get_negative_examples_pubmed(
                        uniprot_dataset[0]["substrate"],
                        uniprot_dataset[0]["substrate_full_name"],
                        uniprot_dataset[0]["substrate_synonyms"],
                        ptm_residue_position_combinations,
                        label=0,
                        margin=self.margin_value_dict["negative_pubmed"],
                        max_number=max_number,
                        seed=seed,
                    )

                    input_examples.extend(samples)
                    self.examples_count_dict["count_negative_pubmed"] += len(samples)

                    # print(self.margin_value_dict)
                    # print(self.examples_count_dict["count_negative_pubmed"])

                # exit()
        # Get random negatives for the whole dataset
        if self.use_random_negatives:
            input_examples = self.get_random_negatives(input_examples, margin=self.margin_value_dict["negative_other_substrate_any_catalyst_match"], max_number=64, seed=seed)

        return input_examples


def get_uniprot_retriever_query(
    substrate_name: str,
    substrate_full_name: str,
    substrate_synonyms: List[str],
    ptm_type: str,
    residue: str,
    position: str,
    synonyms_in_query: bool = False,
    full_name_in_query: bool = False,
) -> str:
    if synonyms_in_query:
        substrates = [substrate_name]
        k = 4
        # Additionally, get the two shortest synonyms which do not share
        # any prefix greater equal three characters
        for substrate in sorted(substrate_synonyms, key=len):
            if len(substrate) >= 3 and not any(
                [substrate.startswith(prefix[:3]) for prefix in substrates]
            ):
                substrates.append(substrate)
            if len(substrates) >= k:
                break
            if len(substrate) > 8:  # For long synonyms, we only need one
                break

        # If the k + 1 st or k + 2 nd synonym are shorter than 5 characters, keep them
        if len(substrates) >= k + 1 and len(substrates[k]) > 5:
            substrates = substrates[:k]
        elif len(substrates) >= k + 2 and len(substrates[k + 1]) > 5:
            substrates = substrates[: k + 1]

        # Join the genes with a comma and "and" for the last one
        if len(substrates) == 2:
            substrate_synonyms_str = " (also known as {})".format(substrates[1])
        elif len(substrates) > 2:
            substrate_synonyms_str = " (also known as {} and {})".format(
                ", ".join(substrates[1:-1]), substrates[-1]
            )
        else:
            substrate_synonyms_str = ""

        query = "Catalyst for the {} of {}{} at {} position {}.".format(
            ptm_type.lower(),
            substrates[0],
            substrate_synonyms_str,
            residue.lower(),
            position,
        )
    elif full_name_in_query:
        query = "Catalyst for the {} of {} ({}) at {} position {}.".format(
            ptm_type.lower(),
            substrate_full_name,
            substrate_name,
            residue.lower(),
            position,
        )
    else:
        query = "Catalyst for the {} of {} at {} position {}.".format(
            ptm_type.lower(), substrate_name, residue.lower(), position
        )
    return query


if __name__ == "__main__":
    from datasets import disable_caching

    from po_datasets.uniprot_ptms import UniProtPTMExamples

    disable_caching()
    examples = UniProtPTMExamples(mode="raw_text")

    retriever = UniprotRetriever(
        substrate_synonyms=True,
        catalysts_synonyms=True,
        full_name_in_query=True,
        filter_no_catalysts=True,
        examples=examples,
        synonyms_in_query=False,
        cache_file_prefix="uniprot_margin_classes_v40_all_neg",
        use_supervised_examples=True,
        use_distant_bm_25_examples=False,
        bm25_k=5,
        bm25_repeat_seen_pmids=False,
        margin_config=margin_config.margin_classes_uniprot_v40,
        use_batch_negatives=False,
        use_random_negatives=False,
        cache=True,
        max_ratio_negatives=20,
        max_negatives=50,
        all_negatives=True,
    )
    print(f"Number of substrates in train: {len(examples.train_split)}")
    print(f"Number of substrates in dev: {len(examples.dev_split)}")
    print(f"Number of substrates in test: {len(examples.test_split)}")

    print(f"Number of retriever train examples: {len(retriever.train)}")
    print(f"Number of retriever dev examples: {len(retriever.dev)}")
    print(f"Number of retriever test examples: {len(retriever.test)}")

    print("Margin values dict:")
    print(retriever.margin_value_dict)

    margin_values = set()
    for example in retriever.train:
        # print(example)
        margin_values.add(example.margin)
        # continue_sample = input("Continue? (y/n)")
        # if continue_sample == "n":
        # break
    print("Margin values:")
    print(margin_values)

    print("InputExample:")
    print(retriever.train[0])
    print(retriever.dev[0])
    print(retriever.test[0])
    print(f"Number of duplicates skipped: {retriever.seen_pmid_duplicates}")

    # Supervised examples, ignore all counts with 0 examples
    print("Breakdown of example types supervised:")
    # Skip examples with 0 counts
    for key, value in retriever.examples_count_dict.items():
        if value > 0:
            print(f"{key}: {value}")
    # print(retriever.examples_count_dict)
    # BM25 examples
    # print("Breakdown of example types BM25:")
    # print(retriever.bm25_examples_count_dict)

    # for sample in retriever.train:
    #     print(sample)
    #     continue_sample = input("Continue? (y/n)")
    #     if continue_sample == "n":
    #         break
