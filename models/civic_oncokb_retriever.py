import itertools
import json
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from datasets import Dataset
from deprecated import deprecated
from tqdm import tqdm

import models.margin_config as margin_class
from models.retriever import Retriever
from models.transformers import InputExample
from po_datasets.create_bm25_examples import BM25Examples
from po_datasets.dataset import ElasticsearchHelper
from utils.utils import get_retriever_query

random.seed(42)

drug_keywords = [
    "clinic",
    "clinical",
    "trial",
    "trials",
    "patient",
    "patients",
    "in vivo",
    "in vitro",
    "in silico",
    "treatment",
    "treated",
    "drug",
    "drugs",
    "therapy",
    "therapies",
    "therapeutic",
    "therapeutically",
    "therapeutical",
    "antitumor",
    "chemotherapy",
    "immunotherapy",
]


class CiVICOncoKBRetriever(Retriever):
    def __init__(
        self,
        gene_synonyms: bool = False,
        variant_synonyms: bool = False,
        drugs_synonyms: bool = False,
        check_for_seen_pmids: bool = False,
        *args,
        **kwargs,
    ):
        # Create input examples
        # TODO: These are not really batch negatives, but rather random negatives
        self.use_gene_synonyms = "_synonyms" if gene_synonyms else ""
        self.use_variant_synonyms = "_synonyms" if variant_synonyms else ""
        self.use_drugs_synonyms = "_synonyms" if drugs_synonyms else ""

        self.examples_count_dict = {
            "count_positive": 0,
            "count_positive_no_entities_match": 0,
            "count_positive_one_entity_match": 0,
            "count_positive_gene_variant_match": 0,
            "count_positive_two_entities_treatment_match": 0,
            "count_negative_same_variant": 0,
            "count_negative_other_variant_no_treatment": 0,
            "count_negative_not_other_variant_any_treatment": 0,  # Only used for margin_classes_v11
            "count_negative_other_variant_any_treatment": 0,
            "count_positive_other_variant_any_treatment": 0,
            "count_negative_other_gene_no_treatment": 0,
            "count_negative_other_gene_any_treatment": 0,
            "count_negative_random": 0,  # Only used for batch negatives
            # Only used for previous faiss examples
            "count_positive_previous_faiss_examples": 0,
            "count_negative_previous_faiss_examples": 0,
            "count_negative_same_substrate_bm25": 0,
            "count_negative_bioasq": 0,
        }
        self.previous_faiss_examples_count_dict = {
            "genes_found": 0,
            "genes_not_found": 0,
        }

        self.negative_sample_overlap_dict = {
            "variant_negative": 0,
            "variant_overlap": 0,
            "gene_negative": 0,
            "gene_overlap": 0,
            "gene_synonym_overlap": 0,
        }

        self.example_overlap_list = []

        self.non_matching_positive_label = 1  # 1 is positive example, 0 is negative

        self.check_for_seen_pmids = check_for_seen_pmids
        # self.seen_pmids = set()
        self.seen_examples = []

        super().__init__(*args, **kwargs)

        if "negative_bioasq" in self.margin_value_dict:
            self.add_bioasq_documents()

        if "margin_classes_v1" in self.cache_file_prefix:
            self.bool_examples_other_variant_any_treatment = False
        elif "margin_classes_v2" in self.cache_file_prefix:
            self.bool_examples_other_variant_any_treatment = False
        elif "margin_classes_v3" in self.cache_file_prefix:
            self.bool_examples_other_variant_any_treatment = True
        elif "margin_classes_v5" in self.cache_file_prefix:
            self.bool_examples_other_variant_any_treatment = False
        else:
            self.bool_examples_other_variant_any_treatment = False

        self.bm25_query_cache_file = "edel_repo_cache/tmp/treatment_explorer_bm25_query_cache.json"
        if Path(self.bm25_query_cache_file).exists():
            with open(self.bm25_query_cache_file, "r") as f:
                self.bm25_query_cache = json.load(f)
        else:
            self.bm25_query_cache = {}

        self.generate_examples()

        with open(self.bm25_query_cache_file, "w") as f:
            json.dump(self.bm25_query_cache, f)

    def add_previous_faiss_examples(self, example_dict: dict):
        # mapping => gene => variant => [pmids_found, pmids_not_found]
        # pmid_texts => pmid => text
        # Do not overwrite existing keys but extend the lists
        for key, value in example_dict.items():
            if key == "mapping":
                for gene, variants in value.items():
                    for variant, pmids in variants.items():
                        if gene in self.previous_faiss_examples["mapping"]:
                            if variant in self.previous_faiss_examples["mapping"][gene]:
                                self.previous_faiss_examples["mapping"][gene][variant][
                                    0
                                ].extend(pmids[0])
                                self.previous_faiss_examples["mapping"][gene][variant][
                                    1
                                ].extend(pmids[1])
                            else:
                                self.previous_faiss_examples["mapping"][gene][
                                    variant
                                ] = pmids
                        else:
                            self.previous_faiss_examples["mapping"][gene] = {
                                variant: pmids
                            }
            elif key == "pmid_texts":
                self.previous_faiss_examples["pmid_texts"].update(value)
            else:
                raise ValueError(f"Unknown key {key} in previous_faiss_examples")

    def get_examples_from_previous_faiss(
        self,
        query: str,
        pubmed_ids: list[int],
        label: int,
        margin_value: float = 0.8,
        query_id: int = -1,
        max_number: Optional[int] = None,
    ):
        input_examples = []

        if max_number is not None and len(pubmed_ids) > max_number:
            pubmed_ids = random.sample(pubmed_ids, max_number)

        for pubmed_id in pubmed_ids:
            # TODO: Add full text support
            pubmed_id_text = self.previous_faiss_examples["pmid_texts"][str(pubmed_id)]
            if isinstance(pubmed_id_text, list):
                pubmed_id_text = pubmed_id_text[0]
            input_examples.append(
                InputExample(
                    texts=[query, pubmed_id_text],
                    label=label,
                    margin=self.margin_class.margin_fn(margin_value),
                    noisy_bool=False,
                    # noisy_bool=True,  # Testing this
                    query_id=query_id,
                    doc_id=pubmed_id,
                )
            )

        return input_examples

    def get_negative_examples_other_gene(
        self,
        dataset: Dataset,
        gene_name: str,
        gene_synonyms: list[str],
        gene_full_name: str,
        variants: list[str],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
        seed: int = 42,
    ) -> list[InputExample]:
        input_examples = []
        shuffled_dataset = dataset.shuffle(seed=seed)
        text_type = "evidence_" + self.mode
        # Since cosine function is not linear, we need to adjust the margin

        for example in shuffled_dataset:
            # if example["citation_id"] in self.seen_pmids and self.check_for_seen_pmids:
            #     continue
            # if len(input_examples) > max_number:
            #     break

            if len(example[text_type]) > 3:
                for variant in variants:
                    if (gene_name, variant) in self.query_id_dict:
                        query_id = self.query_id_dict[(gene_name, variant)]
                    else:
                        self.current_query_id += 1
                        query_id = self.current_query_id
                        self.query_id_dict[(gene_name, variant)] = query_id
                    query = get_retriever_query(
                        gene_name,
                        gene_synonyms,
                        gene_full_name,
                        variant,
                        self.synonyms_in_query,
                    )

                    sample = InputExample(
                            texts=[query, example[text_type]],
                            label=label,
                            margin=self.margin_class.margin_fn(margin),
                            query_id=query_id,
                            doc_id=int(example["citation_id"]),
                        )
                    if sample in self.seen_examples and self.check_for_seen_pmids:
                        continue
                    input_examples.append(
                        sample
                    )
                    self.seen_examples.append(sample)
                    # self.seen_pmids.add(example["citation_id"])

                    # This is for an ablation study
                    # Counting the overlap of the variant of the positive sample in the negative sample
                    if label == 0:
                        self.negative_sample_overlap_dict["gene_negative"] += 1
                        direct_match = 0
                        if gene_name.lower() in example[text_type].lower() or gene_full_name.lower() in example[text_type].lower():
                            self.negative_sample_overlap_dict["gene_overlap"] += 1
                            direct_match = 1
                        if any([synonym.lower() in example[text_type].lower() for synonym in gene_synonyms]):
                            self.negative_sample_overlap_dict["gene_synonym_overlap"] += 1
                        self.example_overlap_list.append(
                            [gene_name, example[text_type], query, label, margin, direct_match]
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
        substrate_synonyms: list[str],
        substrate_full_name: str,
        variants: list[str],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
    ) -> list[InputExample]:
        input_examples = []
        all_seen_pmids = dict()

        query_key = "_".join(
            ["civic_oncokb", substrate_name, substrate_full_name]
        )
        if query_key in self.bm25_query_cache:
            query_pmids = self.bm25_query_cache[query_key]
            for pmid in query_pmids:
                all_seen_pmids[pmid] = self.bm25_query_cache["pmid_" + pmid]
        else:
            results = ElasticsearchHelper.query_keywords(
                keyword_lists=[[substrate_name, substrate_full_name]],
                not_keyword_lists=[drug_keywords],
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

        for variant in variants:
            if (substrate_name, variant) in self.query_id_dict:
                query_id = self.query_id_dict[(substrate_name, variant)]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[(substrate_name, variant)] = query_id
            query = get_retriever_query(
                substrate_name,
                substrate_synonyms,
                substrate_full_name,
                variant,
                self.synonyms_in_query,
            )
            for pmid, example_text in all_seen_pmids.items():
                pmid_int = int(pmid)
                sample = InputExample(
                        texts=[query, example_text],
                        label=label,
                        margin=self.margin_class.margin_fn(margin),
                        query_id=query_id,
                        doc_id=int(pmid_int),
                    )
                # if pmid_int in self.seen_pmids and self.check_for_seen_pmids:
                #     continue
                if sample in self.seen_examples and self.check_for_seen_pmids:
                    continue
                input_examples.append(
                    sample
                )
                self.seen_examples.append(sample)
                # self.seen_pmids.add(pmid_int)

        if len(input_examples) > max_number:
            input_examples = random.sample(
                input_examples,
                max_number,
            )

        return input_examples

    def get_negative_examples_bioasq(
        self,
        substrate_name: str,
        substrate_synonyms: list[str],
        substrate_full_name: str,
        variants: list[str],
        label: int,
        margin: float = 0.0,
        max_number: float = np.inf,
        seed: int = 42,
    ) -> list[InputExample]:
        input_examples = []

        for variant in variants:
            if (substrate_name, variant) in self.query_id_dict:
                query_id = self.query_id_dict[(substrate_name, variant)]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[(substrate_name, variant)] = query_id
            query = get_retriever_query(
                substrate_name,
                substrate_synonyms,
                substrate_full_name,
                variant,
            )

            # TODO Check that entity does not occur in the selected BioASQ text
            # Draw random sample of PMIDs in all BioASQ texts
            current_pmids = self.all_pmids_texts.sample(n=max_number, seed=seed)
            for i, example_text in enumerate(current_pmids["text"]):
                current_pmid = int(current_pmids["pmid"][i])
                sample = InputExample(
                    texts=[query, example_text],
                    label=label,
                    margin=self.margin_class.margin_fn(margin),
                    query_id=query_id,
                    doc_id=current_pmid,
                )
                # if current_pmid in self.seen_pmids and self.check_for_seen_pmids:
                #     continue
                if sample in self.seen_examples and self.check_for_seen_pmids:
                    continue
                input_examples.append(
                    sample
                )

        if len(input_examples) > max_number:
            input_examples = random.sample(
                input_examples,
                max_number,
            )

        return input_examples

    def match_entities_in_text(self, example: dict, col_names: dict) -> dict:
        entity_type_matching_dict = {}
        for col_name in col_names.keys():
            if "gene" in col_name:
                assert isinstance(example[col_name][0], bool)
                if col_names[col_name]:
                    entity_type_matching_dict["gene"] = any(example[col_name])
                else:
                    entity_type_matching_dict["gene"] = not any(example[col_name])
            elif "variant" in col_name:
                assert isinstance(example[col_name][0], bool)
                if col_names[col_name]:
                    entity_type_matching_dict["variant"] = any(example[col_name])
                else:
                    entity_type_matching_dict["variant"] = not any(example[col_name])
            elif "drugs" in col_name:
                assert isinstance(example[col_name][0], bool)
                if col_names[col_name]:
                    entity_type_matching_dict["drugs"] = any(example[col_name])
                else:
                    entity_type_matching_dict["drugs"] = not any(example[col_name])
            else:
                raise ValueError("This should not happen")
        return entity_type_matching_dict

    def get_examples(
        self,
        dataset: Dataset,
        col_names: dict[str],
        query: str,
        label: int,
        margin: float = 0.0,
        process_non_matching_texts: bool = True,
        max_number: float = np.inf,
        seed: int = 42,
        query_id: int = -1,
        variant: str = "UNDEFINED",
    ) -> tuple[list[InputExample], list[InputExample]]:
        input_examples = []
        non_matching_examples = {}
        shuffled_dataset = dataset.shuffle(seed=seed)
        text_type = "evidence_" + self.mode
        # Since cosine function is not linear, we need to adjust the margin
        for i, example in enumerate(shuffled_dataset):
            if i >= max_number:
                break
            citation_id = int(example["citation_id"])
            # if citation_id in self.seen_pmids and self.check_for_seen_pmids and label == 0:
            # if citation_id in self.seen_pmids and self.check_for_seen_pmids:
                # print(citation_id)
                # print(query)
                # print(label)
                # print(margin)
                # print(example)
                # input_string = input("Press enter to continue")
                # continue
            # entities_in_text = all(
            #     # drugs_in_text is nested list with multiple values
            #     # whereas the others are just a simple bool
            #     [
            #         all(
            #             list(
            #                 itertools.chain.from_iterable(
            #                     [example[f"{col_name}_in_{self.mode}"]]
            #                 )
            #             )
            #         )
            #         for col_name in col_names
            #     ]
            # )
            col_names_check = {
                f"{col_name}_in_{self.mode}": col_bool
                for col_name, col_bool in col_names.items()
            }
            matching_dict = self.match_entities_in_text(example, col_names_check)
            entities_in_text = all(matching_dict.values())
            if entities_in_text and len(example[text_type]) > 3:
                sample = InputExample(
                    texts=[query, example[text_type]],
                    label=label,
                    margin=self.margin_class.margin_fn(margin),
                    query_id=query_id,
                    doc_id=citation_id,
                )
                if sample in self.seen_examples and self.check_for_seen_pmids:
                    continue
                input_examples.append(
                    sample
                )
                self.seen_examples.append(sample)
                # self.seen_pmids.add(citation_id)
                # This is for an ablation study
                # Counting the overlap of the variant of the positive sample in the negative sample
                if label == 0:
                    self.negative_sample_overlap_dict["variant_negative"] += 1
                    if variant.lower() in example[text_type].lower():
                        self.negative_sample_overlap_dict["variant_overlap"] += 1
                    self.example_overlap_list.append(
                        [variant, example[text_type], query, label, margin, 1]
                    )
            elif (
                self.use_non_matching_texts
                and process_non_matching_texts
                and not entities_in_text
                and len(example[text_type]) > 3
            ):
                # Check out which type on positive non-matching example
                if (
                    # not any(example[f"gene_in_{self.mode}"])
                    # and not any(example[f"variant_in_{self.mode}"])
                    # and not any(example[f"drugs_in_{self.mode}"])
                    sum(matching_dict.values())
                    == 0
                ):
                    margin_class_string = "positive_no_entities_match"
                elif (
                    # any(example[f"gene_in_{self.mode}"])
                    # + any(example[f"variant_in_{self.mode}"])
                    # + any(example[f"drugs_in_{self.mode}"])
                    # == 1
                    sum(matching_dict.values())
                    == 1
                ):
                    margin_class_string = "positive_one_entity_match"
                elif (
                    #     any(example[f"gene_in_{self.mode}"])
                    #     + any(example[f"variant_in_{self.mode}"])
                    #     + any(example[f"drugs_in_{self.mode}"])
                    #     == 2
                    # ) and any(example[f"drugs_in_{self.mode}"]):
                    sum(matching_dict.values()) == 2
                    and matching_dict["drugs"]
                ):
                    margin_class_string = "positive_two_entities_treatment_match"
                # elif any(example[f"gene_in_{self.mode}"]) and any(
                #     example[f"variant_in_{self.mode}"]
                elif (
                    sum(matching_dict.values()) == 2
                    and matching_dict["gene"]
                    and matching_dict["variant"]
                ):
                    margin_class_string = "positive_gene_variant_match"
                else:
                    raise ValueError("This should not happen")

                # Only for margin_classes_v7
                if margin_class_string not in self.margin_value_dict:
                    continue

                non_matching_examples.setdefault(margin_class_string, [])
                sample = InputExample(
                    texts=[query, example[text_type]],
                    label=self.non_matching_positive_label,
                    margin=self.margin_class.margin_fn(
                        self.margin_value_dict[margin_class_string]
                    ),
                    query_id=query_id,
                    doc_id=citation_id,
                )
                non_matching_examples[margin_class_string].append(
                    sample
                )
                self.seen_examples.append(sample)
                # self.seen_pmids.add(citation_id)
        return input_examples, non_matching_examples

    def get_bm25_examples(self, dataset: Dataset) -> list[InputExample]:
        gene_col = "gene" + self.use_gene_synonyms
        drugs_col = "drugs" + self.use_drugs_synonyms

        bm25exmples = BM25Examples(
            dataset,
            synonyms_in_query=self.synonyms_in_query,
            k=self.bm25_k,
            margin_class=self.margin_class,
            use_gene_synonyms=gene_col,
            use_drug_synonyms=drugs_col,
            repeat_seen_pmids=self.bm25_repeat_seen_pmids,
        )
        # Sum up values from frequency dict
        for key, value in bm25exmples.frequency_dict.items():
            self.bm25_examples_count_dict[key] += value

        return bm25exmples.input_examples

    def get_positive_and_hard_negative_examples(
        self, dataset: Dataset
    ) -> tuple[list[InputExample], int]:
        input_examples = []
        gene_col = "gene" + self.use_gene_synonyms
        variant_col = "variant"  # + self.use_variant_synonyms
        # Always use exact variants for now
        drugs_col = "drugs" + self.use_drugs_synonyms
        variants = set(dataset[variant_col])
        eg_count_pos = 0
        eg_id = dataset[0]["entrez_id"]
        query = ""

        for variant in variants:
            variant_treatment_dataset = dataset.filter(
                lambda x: x[variant_col] == variant and len(x[drugs_col]) > 0
            )
            variant_no_treatment_dataset = dataset.filter(
                lambda x: x[variant_col] == variant and len(x[drugs_col]) == 0
            )
            other_variant_dataset_no_treatment = dataset.filter(
                lambda x: x[variant_col] != variant and len(x[drugs_col]) == 0
            )
            other_variant_dataset_any_treatment = dataset.filter(
                lambda x: x[variant_col] != variant and len(x[drugs_col]) > 0
            )

            gene_name = dataset[0]["gene"]
            if (gene_name, variant) in self.query_id_dict:
                query_id = self.query_id_dict[(gene_name, variant)]
            else:
                self.current_query_id += 1
                query_id = self.current_query_id
                self.query_id_dict[(gene_name, variant)] = query_id
            query = get_retriever_query(
                dataset[0]["gene"],
                dataset[0]["gene_synonyms"],
                dataset[0]["gene_full_name"],
                variant,
                self.synonyms_in_query,
            )
            # self.seen_pmids = set()

            col_names = {
                gene_col: True,
                variant_col: True,
                drugs_col: True,
            }

            # Get positive examples for same gene and same variant
            positive_examples, non_matching_examples_dict = self.get_examples(
                variant_treatment_dataset,
                col_names=col_names,
                query=query,
                label=1,
                margin=self.margin_value_dict["positive"],
                seed=eg_id,
                query_id=query_id,
                variant=variant,
            )

            input_examples.extend(positive_examples)
            self.examples_count_dict["count_positive"] += len(positive_examples)

            number_all_positive_examples = len(positive_examples)
            for (
                _,
                non_matching_examples,
            ) in non_matching_examples_dict.items():
                number_all_positive_examples += len(non_matching_examples)
            if self.all_negatives:
                count_positive_examples = number_all_positive_examples
            else:
                count_positive_examples = len(positive_examples)
            eg_count_pos += count_positive_examples

            # These are negative examples with entities and their synonyms
            # not matching any text
            # TODO: Exclude examples where a synonym of an entity is actually in the text
            # TODO: These examples may be noisy. Check the performance and
            # remove them, if necessary.
            if (
                self.use_non_matching_texts
                #     and self.use_gene_synonyms
                #     and self.use_variant_synonyms
                #     and self.use_drugs_synonyms
            ):
                for (
                    margin_class_string,
                    non_matching_examples,
                ) in non_matching_examples_dict.items():
                    input_examples.extend(non_matching_examples)
                    self.examples_count_dict[f"count_{margin_class_string}"] += len(
                        non_matching_examples
                    )

            # Filter general alteration specifiers, e.g. "mutation" in CIViC
            # and "oncogenic mutations" in OncoKB
            # Also add expression and overexpression to the list
            if variant.lower() in [
                "mutation",
                "oncogenic mutations",
                "expression",
                "overexpression",
            ]:
                continue

            if self.use_batch_negatives or self.use_random_negatives:
                continue

            # Special case for negative_samee_gene_other_variant_any_treatment_match
            # If other variant not matching in the abstract text, treat it as more likely to be positive
            # Only used in margin_classes_v11 for now
            if (
                "negative_same_gene_not_other_variant_any_treatment_match"
                in self.margin_value_dict
            ):
                col_names = {
                    gene_col: True,
                    variant_col: False,
                }
                negative_examples_not_other_variant_any_treatment, _ = (
                    self.get_examples(
                        other_variant_dataset_any_treatment,
                        col_names=col_names,
                        query=query,
                        label=0,
                        margin=self.margin_value_dict[
                            "negative_same_gene_not_other_variant_any_treatment_match"
                        ],
                        process_non_matching_texts=False,
                        seed=eg_id,
                        query_id=query_id,
                        variant=variant,
                    )
                )
                if (
                    len(negative_examples_not_other_variant_any_treatment)
                    > count_positive_examples * self.max_ratio_negatives
                ):
                    negative_examples_not_other_variant_any_treatment = random.sample(
                        negative_examples_not_other_variant_any_treatment,
                        count_positive_examples * self.max_ratio_negatives,
                    )
                # Printing some examples for debugging
                # if len(negative_examples_not_other_variant_any_treatment) > 0:
                #     print(
                #         "First example for margin class negative_same_gene_not_other_variant_any_treatment_match"
                #     )
                #     print(variant)
                #     print(eg_id)
                #     print(negative_examples_not_other_variant_any_treatment[0])
                #     input_string = input("Press enter to continue")
                input_examples.extend(negative_examples_not_other_variant_any_treatment)
                self.examples_count_dict[
                    "count_negative_not_other_variant_any_treatment"
                ] += len(negative_examples_not_other_variant_any_treatment)

            if "negative_same_gene_variant_no_treatment_match" in self.margin_value_dict:
                # Get negative examples for same gene and same variant
                # print(self.margin_value_dict)
                col_names = {
                    gene_col: True,
                    variant_col: True,
                }
                negative_examples_same_variant, _ = self.get_examples(
                    variant_no_treatment_dataset,
                    col_names=col_names,
                    query=query,
                    label=0,
                    margin=self.margin_value_dict[
                        "negative_same_gene_variant_no_treatment_match"
                    ],
                    process_non_matching_texts=False,
                    seed=eg_id,
                    query_id=query_id,
                    variant=variant,
                )
                if (
                    len(negative_examples_same_variant)
                    > count_positive_examples * self.max_ratio_negatives
                ):
                    negative_examples_same_variant = random.sample(
                        negative_examples_same_variant,
                        count_positive_examples * self.max_ratio_negatives,
                    )
                input_examples.extend(negative_examples_same_variant)
                self.examples_count_dict["count_negative_same_variant"] += len(
                    negative_examples_same_variant
                )

            if self.bool_examples_other_variant_any_treatment and (
                "positive_same_gene_other_variant_any_treatment_match"
                in self.margin_value_dict
            ):
                # Get additional positive examples for same gene and other variant and any treatment
                col_names = {
                    gene_col: True,
                    variant_col: True,
                }
                positive_examples_other_variant_any_treatment, _ = self.get_examples(
                    other_variant_dataset_any_treatment,
                    col_names=col_names,
                    query=query,
                    label=1,
                    # Labels are more or less arbitrary for now. They only decide either min or max value for margin
                    margin=self.margin_value_dict[
                        "positive_same_gene_other_variant_any_treatment_match"
                    ],
                    process_non_matching_texts=False,
                    seed=eg_id,
                    query_id=query_id,
                    variant=variant,
                )
                if len(positive_examples_other_variant_any_treatment) > len(
                    count_positive_examples * self.max_ratio_negatives
                ):  # * self.max_ratio_negatives, make these examples less frequent
                    positive_examples_other_variant_any_treatment = random.sample(
                        positive_examples_other_variant_any_treatment,
                        count_positive_examples * self.max_ratio_negatives,
                    )
                input_examples.extend(positive_examples_other_variant_any_treatment)
                self.examples_count_dict[
                    "count_positive_other_variant_any_treatment"
                ] += len(positive_examples_other_variant_any_treatment)
                # TODO: Add batch negatives for other genes
                self.examples_count_dict["count_positive"] += len(
                    positive_examples_other_variant_any_treatment
                )
            elif "negative_same_gene_other_variant_any_treatment_match" in self.margin_value_dict:
                col_names = {
                    gene_col: True,
                    variant_col: True,
                }
                negative_examples_not_other_variant_any_treatment, _ = (
                    self.get_examples(
                        other_variant_dataset_any_treatment,
                        col_names=col_names,
                        query=query,
                        label=0,
                        margin=self.margin_value_dict[
                            "negative_same_gene_other_variant_any_treatment_match"
                        ],
                        process_non_matching_texts=False,
                        seed=eg_id,
                        query_id=query_id,
                        variant=variant,
                    )
                )
                if (
                    len(negative_examples_not_other_variant_any_treatment)
                    > count_positive_examples * self.max_ratio_negatives
                ):
                    negative_examples_not_other_variant_any_treatment = random.sample(
                        negative_examples_not_other_variant_any_treatment,
                        count_positive_examples * self.max_ratio_negatives,
                    )
                input_examples.extend(negative_examples_not_other_variant_any_treatment)
                self.examples_count_dict[
                    "count_negative_other_variant_any_treatment"
                ] += len(negative_examples_not_other_variant_any_treatment)

            if "negative_same_gene_other_variant_no_treatment_match" in self.margin_value_dict:
                # Get negative examples for same gene and other variant and no treatment
                col_names = {
                    gene_col: True,
                    variant_col: True,
                }
                negative_examples_other_variant_no_treatment, _ = self.get_examples(
                    other_variant_dataset_no_treatment,
                    col_names=col_names,
                    query=query,
                    label=0,
                    margin=self.margin_value_dict[
                        "negative_same_gene_other_variant_no_treatment_match"
                    ],
                    process_non_matching_texts=False,
                    seed=eg_id,
                    query_id=query_id,
                    variant=variant,
                )
                if (
                    len(negative_examples_other_variant_no_treatment)
                    > count_positive_examples * self.max_ratio_negatives
                ):
                    negative_examples_other_variant_no_treatment = random.sample(
                        negative_examples_other_variant_no_treatment,
                        count_positive_examples * self.max_ratio_negatives,
                    )
                # Debugging retriever queries and input examples
                # if len(negative_examples_other_variant_no_treatment) > 0:
                #     print(
                #         "First example for margin class negative_same_gene_other_variant_no_treatment_match"
                #     )
                #     print(variant)
                #     print(eg_id)
                #     print(negative_examples_other_variant_no_treatment[0])
                #     input_string = input("Press enter to continue")
                input_examples.extend(negative_examples_other_variant_no_treatment)
                self.examples_count_dict[
                    "count_negative_other_variant_no_treatment"
                ] += len(negative_examples_other_variant_no_treatment)

        return (input_examples, eg_count_pos)

    def sample_generator_for_retriever(
        self,
        dataset: Dataset,
        split: Union[np.ndarray[Any, Any], list[int]],
        split_name: str,
    ) -> tuple[list[InputExample], list[list[InputExample]]]:
        input_examples = []

        if self.add_beir_datasets and split_name == "train":
            beir_examples = self.get_beir_examples()
            input_examples.extend(beir_examples)
        if self.only_beir_datasets:
            return input_examples

        for eg_id in tqdm(split, desc="Generating examples"):
            eg_dataset = dataset.filter(lambda x: x["entrez_id"] == eg_id)
            (
                examples,
                count_pos,
            ) = self.get_positive_and_hard_negative_examples(eg_dataset)
            variants = set(eg_dataset["variant"])
            seed = eg_dataset[0]["entrez_id"]
            max_number = count_pos * self.max_ratio_negatives
            # max_number = int(count_pos * self.max_ratio_negatives / 4)
            if len(examples) > 0 and self.use_supervised_examples:
                input_examples.extend(examples)

                if self.use_batch_negatives or self.use_random_negatives:
                    continue

                # Get negative examples for other genes and no treatment
                other_eg_dataset_no_treatment = dataset.filter(
                    lambda x: x["entrez_id"] != eg_id and len(x["drugs"]) == 0
                )

                negative_examples = self.get_negative_examples_other_gene(
                    other_eg_dataset_no_treatment,
                    eg_dataset[0]["gene"],
                    eg_dataset[0]["gene_synonyms"],
                    eg_dataset[0]["gene_full_name"],
                    variants,
                    label=0,
                    margin=self.margin_value_dict[
                        "negative_other_gene_no_treatment_match"
                    ],
                    max_number=max_number,
                    seed=seed,
                )
                input_examples.extend(negative_examples)
                self.examples_count_dict[
                    "count_negative_other_gene_no_treatment"
                ] += len(negative_examples)

                # Get negative examples for other genes and any treatment
                other_eg_dataset_any_treatment = dataset.filter(
                    lambda x: x["entrez_id"] != eg_id and len(x["drugs"]) > 0
                )

                negative_examples = self.get_negative_examples_other_gene(
                    other_eg_dataset_any_treatment,
                    eg_dataset[0]["gene"],
                    eg_dataset[0]["gene_synonyms"],
                    eg_dataset[0]["gene_full_name"],
                    variants,
                    label=0,
                    margin=self.margin_value_dict[
                        "negative_other_gene_any_treatment_match"
                    ],
                    max_number=max_number,
                    seed=seed,
                )

                input_examples.extend(negative_examples)
                self.examples_count_dict[
                    "count_negative_other_gene_any_treatment"
                ] += len(negative_examples)

            # Get negative examples same substrate not Uniprot from BM25
            if "negative_same_substrate_bm25" in self.margin_value_dict:
                samples = self.get_negative_examples_bm25(
                    eg_dataset[0]["gene"],
                    eg_dataset[0]["gene_synonyms"],
                    eg_dataset[0]["gene_full_name"],
                    variants,
                    label=0,
                    margin=self.margin_value_dict[
                        "negative_same_substrate_bm25"
                    ],
                    max_number=max_number,
                )

                input_examples.extend(samples)
                self.examples_count_dict[
                    "count_negative_same_substrate_bm25"
                ] += len(samples)

            # Get negative examples from BioASQ
            if "negative_bioasq" in self.margin_value_dict:
                samples = self.get_negative_examples_bioasq(
                    eg_dataset[0]["gene"],
                    eg_dataset[0]["gene_synonyms"],
                    eg_dataset[0]["gene_full_name"],
                    variants,
                    label=0,
                    margin=self.margin_value_dict["negative_bioasq"],
                    max_number=max_number,
                    seed=seed,
                )

                input_examples.extend(samples)
                self.examples_count_dict["count_negative_bioasq"] += len(samples)

            if self.use_distant_bm_25_examples:
                bm25_examples = self.get_bm25_examples(eg_dataset)
                input_examples.extend(bm25_examples)

            if self.use_previous_faiss_examples:
                gene_name = eg_dataset[0]["gene"]
                variants = set(eg_dataset["variant"])

                for variant_name in variants:
                    if (
                        gene_name in self.previous_faiss_examples["mapping"]
                        and variant_name
                        in self.previous_faiss_examples["mapping"][gene_name]
                    ):
                        if (gene_name, variant_name) in self.query_id_dict:
                            query_id = self.query_id_dict[(gene_name, variant_name)]
                        else:
                            self.current_query_id += 1
                            query_id = self.current_query_id
                            self.query_id_dict[(gene_name, variant_name)] = query_id
                        query = get_retriever_query(
                            gene_name,
                            eg_dataset[0]["gene_synonyms"],
                            eg_dataset[0]["gene_full_name"],
                            variant_name,
                            self.synonyms_in_query,
                        )

                        if self.include_positive_faiss_examples:
                            positive_faiss_examples = (
                                self.get_examples_from_previous_faiss(
                                    query,
                                    self.previous_faiss_examples["mapping"][gene_name][
                                        variant_name
                                    ][0],
                                    label=1,
                                    margin_value=self.margin_value_dict[
                                        "positive_previous_faiss_examples"
                                    ],
                                    query_id=query_id,
                                    max_number=10,
                                )
                            )
                            input_examples.extend(positive_faiss_examples)
                            self.examples_count_dict[
                                "count_positive_previous_faiss_examples"
                            ] += len(positive_faiss_examples)

                        negative_faiss_examples = self.get_examples_from_previous_faiss(
                            query,
                            self.previous_faiss_examples["mapping"][gene_name][
                                variant_name
                            ][1],
                            label=0,
                            margin_value=self.margin_value_dict[
                                "negative_previous_faiss_examples"
                            ],
                            query_id=query_id,
                            max_number=50,
                        )
                        input_examples.extend(negative_faiss_examples)
                        self.examples_count_dict[
                            "count_negative_previous_faiss_examples"
                        ] += len(negative_faiss_examples)

                        # Debugging created examples
                        print(f"Gene: {gene_name}, Variant: {variant_name}")
                        print(f"Synonyms: {eg_dataset[0]['gene_synonyms']}")
                        for sample in positive_faiss_examples:
                            print(sample)
                            input_string = input("Press enter to continue")
                            if input_string == "q":
                                break
                        for sample in negative_faiss_examples:
                            print(sample)
                            input_string = input("Press enter to continue")
                            if input_string == "q":
                                break

        # Get random negatives for the whole dataset
        if self.use_random_negatives:
            input_examples = self.get_random_negatives(input_examples, margin=self.margin_value_dict["negative_other_gene_any_treatment_match"], max_number=64, seed=seed)

        return input_examples


if __name__ == "__main__":
    from datasets import disable_caching

    from po_datasets.civic import CiVICExamples
    from po_datasets.concat_dataset import ConcatExamples
    from po_datasets.onco_kb import OncoKBExamples

    disable_caching()
    examples = ConcatExamples(
        [CiVICExamples(mode="raw_text"), OncoKBExamples(mode="raw_text")],
        mode="raw_text",
    )
    examples.detailed_dataset_stats()
    # examples = OncoKBExamples(mode="raw_text")
    # examples = CiVICExamples(mode="raw_text")

    retriever = CiVICOncoKBRetriever(
        gene_synonyms=True,
        variant_synonyms=True,
        drugs_synonyms=True,
        check_for_seen_pmids=True,
        all_negatives=True,
        examples=examples,
        synonyms_in_query=False,
        cache_file_prefix="civic_onco_kb_abstracts_margin_classes_v13_debug",
        use_supervised_examples=True,
        bm25_repeat_seen_pmids=False,
        margin_config=margin_class.margin_classes_v13,
        use_batch_negatives=True,
        use_random_negatives=False,
        cache=False,
        # add_beir_datasets=False,
        max_ratio_negatives=20,
    )
    print(f"Number of genes in train: {len(examples.train_split)}")
    print(f"Number of genes in dev: {len(examples.dev_split)}")
    print(f"Number of genes in test: {len(examples.test_split)}")

    print(f"Number of retriever train examples: {len(retriever.train)}")
    print(f"Number of retriever dev examples: {len(retriever.dev)}")
    print(f"Number of retriever test examples: {len(retriever.test)}")

    print("Margin classes:")
    print(retriever.margin_value_dict)

    print("InputExample:")
    print(retriever.train[0])
    print(retriever.dev[0])
    print(retriever.test[0])

    # Supervised examples
    print("Breakdown of example types supervised:")
    # Skip examples with 0 counts
    for key, value in retriever.examples_count_dict.items():
        if value > 0:
            print(f"{key}: {value}")
    # BM25 examples
    print("Breakdown of example types BM25:")
    print(retriever.bm25_examples_count_dict)

    # Print out a nice example for the overview figure in the paper
    # Iterate through the train set to get a sorted list of all margin values
    margin_values = set()
    for example in retriever.train:
        margin_values.add(example.margin)
    margin_values = sorted(list(margin_values))
    print(margin_values)

    # Group train examples by query
    # Add counts for each label/margin combination
    query_dict = {}
    for example in retriever.train:
        current_query = example.texts[0]
        if current_query not in query_dict:
            query_dict[current_query] = {
                "positive": {margin: 0 for margin in margin_values},
                "negative": {margin: 0 for margin in margin_values},
            }
        if example.label == 1:
            query_dict[current_query]["positive"][example.margin] += 1
        else:
            query_dict[current_query]["negative"][example.margin] += 1

    # Filter for queries with at least two positive examples, one of them with margin 0
    # and at least three negative examples with different margins
    # Return one unique example for each label/margin combination for the first relevant query
    relevant_queries = []
    for query, values in query_dict.items():
        unique_negative_margins = 0
        if values["positive"][0] > 0 and sum(values["positive"].values()) >= 2:
            for margin in margin_values:
                if values["negative"][margin] > 0:
                    unique_negative_margins += 1

            if query == "Treatment for gene MGMT and variant Promoter Methylation.":
                continue

            if unique_negative_margins >= 3:
                relevant_queries.append(query)
                print(query)
                for margin in margin_values:
                    if values["positive"][margin] > 0:
                        print(f"Positive {margin}: {values['positive'][margin]}")
                    if values["negative"][margin] > 0:
                        print(f"Negative {margin}: {values['negative'][margin]}")
                print()
                break

    # Print out the first unique example for each label/margin combination for the first relevant query
    processed_label_margin_combinations = set()
    for example in retriever.train:
        current_margin = example.margin
        current_label = example.label
        current_query = example.texts[0]
        if current_query in relevant_queries and (current_label, current_margin) not in processed_label_margin_combinations:
            processed_label_margin_combinations.add((current_label, current_margin))
            print(example)

    # for sample in retriever.train:
    #     print(sample)
    #     continue_sample = input("Continue? (y/n)")
    #     if continue_sample == "n":
    #         break

    # Print results for ablation study
    print("Negative sample overlap:")
    print(retriever.negative_sample_overlap_dict)
    print("Example overlap list:")
    # Gather random sample of 20 examples
    random.shuffle(retriever.example_overlap_list)
    print(retriever.example_overlap_list[:20])

    # Find out where the differences between positive samples come from in the retriever and the dataset
    dataset_dict = {}
    retriever_dict = {}
    for entry in examples.unique_gene_variants_pmids:
        dataset_dict.setdefault(entry[2], []).append(entry)
    retriever_examples = retriever.train + retriever.dev + retriever.test
    for example in retriever_examples:
        if example.label == 1:
            retriever_dict.setdefault(example.doc_id, []).append(example)
    # Compare number of positive examples for each gene/variant pair
    for key, value in dataset_dict.items():
        if key in retriever_dict:
            # if len(value) != len(retriever_dict[key]):
            if len(value) != len(retriever_dict[key]):
                print(key)
                print(sorted(value, key=lambda x: x[1]))
                print(f"Number of examples in dataset: {len(value)}")
                queries = []
                for example in retriever_dict[key]:
                    queries.append(example.texts[0])
                print(sorted(queries))
                print(f"Number of examples in retriever: {len(retriever_dict[key])}")
                print()
                input("Press enter to continue")
    print("Comparison done")
