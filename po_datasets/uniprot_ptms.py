import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sqlitedict import SqliteDict
from tqdm import tqdm

from utils.ncithesaurus import NCITheSaurusMapper

from .dataset import DatasetExamples, filter_short_synonyms
from .uniprot_dictionaries import (
    AMINO_ACIDS,
    GENE_SYMBOL_TO_NCBI,
    MOD_RES_MAPPING,
    NON_CATALYSTS,
    PTM_MOD_RES_MAPPING_REVERSE,
    PTM_SYNONYMS,
    SPECIAL_ENTITY_DICT,
)


class UniProtPTMExamples(DatasetExamples):
    def __init__(
        self,
        file: str = "edel_repo_cache/uniprot_sprot.dat",
        mode="raw_full_text",
        filter_pubmed: bool = True,
        bool_group_by_citation_id: bool = False,
        bool_group_by_alteration: bool = False,
        cache_dir_prefix: str = "edel_repo_cache/uniprot_ptms/examples_",
        cache: bool = True,
        es_index: str = "20231127_pubmed",
    ):
        self.es_index = es_index
        self.file = file
        self.mode = mode

        self.maximum_synonym_length_ratio = 5

        self.not_found_symbols = set()

        self.sep_token = (
            SentenceTransformer("michiyasunaga/BioLinkBERT-base")
            ._first_module()
            .tokenizer.sep_token
        )

        # Debug Elasticsearch index
        # random_query = self.query_keywords([])

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
            self.dataset = (
                self.create_dataset()
                .map(self.extract_pmids, batched=True)
                .map(self.split_notes, batched=True)
            )  # .map(
            # self.add_raw_evidence, batched=True
            # )

            print(f"Loaded {len(self.dataset)} entries")
            print(self.dataset)
            print(self.dataset[3])

            # For debugging processes, restrict to a subset
            # self.dataset = self.dataset.select(range(1000))

            # Post-processing the raw UniProt entries
            # Filter for human genes was already done in the dataset creation
            self.dataset = self.dataset.filter(
                lambda x: x["gene_names"] != ["UNDEFINED"]
            )
            print(
                f"Filtered for genes with DEFINED names, {len(self.dataset)} entries left"
            )

            # Get set of all unique mod_res_product
            unique_mod_res_products = set(self.dataset["mod_res_product"])
            print(f"Unique mod_res_products: {len(unique_mod_res_products)}")
            print(unique_mod_res_products)

            # Count number of entries with PMIDs
            entries_with_pmids = self.dataset.filter(
                lambda x: len(x["mod_res_pmids"]) > 0
            )
            print(f"Entries with PMIDs: {len(entries_with_pmids)}")

            # Split mod_resproduct into mod_res and entity
            self.dataset = self.dataset.map(self.split_mod_res_product, batched=True)

            # Get unique mod_res and entities and their frequencies
            unique_mod_res_dict = {}
            unique_entity_dict = {}
            for entry in tqdm(self.dataset):
                mod_res = entry["mod_res"]
                entity = entry["entity"]
                if mod_res not in unique_mod_res_dict:
                    unique_mod_res_dict[mod_res] = 0
                unique_mod_res_dict[mod_res] += 1
                if entity not in unique_entity_dict:
                    unique_entity_dict[entity] = 0
                unique_entity_dict[entity] += 1
            # Sort the dictionaries
            unique_mod_res_dict = dict(
                sorted(
                    unique_mod_res_dict.items(), key=lambda item: item[1], reverse=True
                )
            )
            unique_entity_dict = dict(
                sorted(
                    unique_entity_dict.items(), key=lambda item: item[1], reverse=True
                )
            )
            print(f"Unique mod_res: {len(unique_mod_res_dict)}")
            print(unique_mod_res_dict)
            print(f"Unique entities: {len(unique_entity_dict)}")
            print(unique_entity_dict)

            # Count entries with catalysts
            dataset_with_catalysts = self.dataset.filter(
                lambda x: x["mod_res_catalyst"] != ""
                and " by " in x["mod_res_catalyst"]
            )
            print(f"Entries with catalysts: {len(dataset_with_catalysts)}")
            # Split exemplary " by MAPKAPK2, MAPKAPK3 and MAPKAPK5" into individual catalysts
            self.dataset = self.dataset.map(self.split_catalysts, batched=True)

            # Get unique catalysts and their frequencies
            unique_catalyst_dict = {}
            for entry in tqdm(self.dataset):
                catalysts = entry["mod_res_catalyst_list"]
                for catalyst in catalysts:
                    if catalyst == "":
                        continue
                    if catalyst not in unique_catalyst_dict:
                        unique_catalyst_dict[catalyst] = 0
                    unique_catalyst_dict[catalyst] += 1
            # Sort the dictionary
            unique_catalyst_dict = dict(
                sorted(
                    unique_catalyst_dict.items(), key=lambda item: item[1], reverse=True
                )
            )
            print(f"Unique catalysts: {len(unique_catalyst_dict)}")
            print(unique_catalyst_dict)

            # Print all unique mod_res_pmids
            unique_pmids = set()
            for entry in self.dataset:
                unique_pmids.update(entry["mod_res_pmids"])
            print(f"Unique PMIDs: {len(unique_pmids)}")
            # print(unique_pmids)
            # print(self.dataset["mod_res_pmids"])

            # Filter out all entries without well-defined residue entities
            print(
                f"Number of entries before filtering for well-formed residues: {len(self.dataset)}"
            )
            self.dataset = self.dataset.filter(lambda x: x["entity"] != "UNKNOWN")
            print(
                f"Number of entries after filtering for well-formed residues: {len(self.dataset)}"
            )

            print("Processing the dataset...")
            # Split PMIDs into separate entries
            self.dataset = (
                self.dataset.map(self.add_dummy_pmids, batched=True)
                .map(
                    self.manual_explode_pmids,
                    batched=True,
                    remove_columns=["mod_res_pmids"],
                )
                .rename_column("mod_res_pmids", "citation_id")
            )

            # print(self.dataset["citation_id"])

            self.dataset = self.dataset.filter(lambda x: x["citation_id"] != -1)

            # print(self.dataset["citation_id"])

            # Add abstract evidences
            self.dataset = self.dataset.map(self.add_raw_evidence, batched=True)

            # Add re-named columns
            self.dataset = (
                self.dataset.map(self.extract_protein_name, batched=True)
                .rename_column("mod_res", "ptm_type")
                .rename_column("entity", "residue")
                .rename_column("mod_res_pos", "position")
                .rename_column("mod_res_catalyst_list", "catalysts")
                .rename_column("full_name", "substrate_full_name")
            )

            self.dataset = (
                self.dataset.map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "substrate"},
                )
                .map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "ptm_type"},
                )
                .map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "residue"},
                )
                .map(
                    self.filter_entity_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "position"},
                )
                .map(
                    self.filter_entity_list_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "catalysts"},
                )
            )

            # Add synonyms
            self.ncbi_gene_db = SqliteDict(
                "edel_repo_cache/gene_names.sqlite",
                autocommit=True,
                tablename="gene_names_tax_to_id",
            )

            self.dataset = (
                self.dataset.map(self.add_substrate_synonyms, batched=True)
                .map(self.add_ptm_type_synonyms, batched=True)
                .map(self.add_catalysts_synonyms, batched=True)
                .map(
                    self.filter_entity_synonyms_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "substrate"},
                )
                .map(
                    self.filter_entity_synonyms_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "ptm_type"},
                )
                .map(
                    self.filter_entity_synonyms_in_text,
                    batched=True,
                    fn_kwargs={"entity_type": "catalysts"},
                )
            )

            # Add (amino acid) residue and position synonyms handling
            self.dataset = self.dataset.map(
                self.filter_res_synonyms_in_text, batched=True
            ).map(self.filter_pos_synonyms_in_text, batched=True)

            print("Finished processing the dataset")
            print("Example entries:")
            print(self.dataset[0])
            print(self.dataset[1])
            print(self.dataset[2])
            print("Number of entries:")
            print(len(self.dataset))

            if cache:
                self.dataset.save_to_disk(cache_dir)

        # Split across UniProt primary accession IDs
        all_uniprot_ids = set(self.dataset["primary_accession"])
        uniprot_ids_with_catalysts = set(
            [
                entry["primary_accession"]
                for entry in self.dataset
                if len(entry["catalysts"]) > 0
            ]
        )
        np.random.seed(42)
        permutation = np.random.permutation(sorted(uniprot_ids_with_catalysts))
        self.train_split = permutation[: int(len(permutation) * 0.7)]
        self.dev_split = permutation[
            int(len(permutation) * 0.7) : int(len(permutation) * 0.8)
        ]
        self.test_split = permutation[int(len(permutation) * 0.8) :]

        # Expand each data split with the same number of entries not containing catalysts
        ids_no_catalysts = sorted(
            all_uniprot_ids.difference(uniprot_ids_with_catalysts)
        )
        permutation = np.random.permutation(ids_no_catalysts)
        extra_train_split = permutation[: int(len(permutation) * 0.7)]
        # print(all_uniprot_ids)
        print(f"Length of train split before: {len(self.train_split)}")
        # print(type(self.train_split))
        # print(extra_train_split)
        # print(type(extra_train_split))
        self.train_split = np.concatenate(
            (self.train_split, extra_train_split[: len(self.train_split)])
        )
        print(f"Length of train split after: {len(self.train_split)}")
        extra_dev_split = permutation[
            int(len(permutation) * 0.7) : int(len(permutation) * 0.8)
        ]
        self.dev_split = np.concatenate(
            (self.dev_split, extra_dev_split[: len(self.dev_split)])
        )
        extra_test_split = permutation[int(len(permutation) * 0.8) :]
        self.test_split = np.concatenate(
            (self.test_split, extra_test_split[: len(self.test_split)])
        )

        # Filter dataset
        self.train = self.dataset.filter(
            lambda x: x["primary_accession"] in self.train_split
        )
        self.dev = self.dataset.filter(
            lambda x: x["primary_accession"] in self.dev_split
        )
        self.test = self.dataset.filter(
            lambda x: x["primary_accession"] in self.test_split
        )

        print("Train split:")
        print(f"Number of entries: {len(self.train)}")
        print(
            f"""Number of entries with catalysts: {len(self.train.filter(lambda x: len(x["catalysts"]) > 0))}"""
        )
        print(f"Number of substrates: {len(set(self.train_split))}")
        print("Dev split:")
        print(f"Number of entries: {len(self.dev)}")
        print(
            f"""Number of entries with catalysts: {len(self.dev.filter(lambda x: len(x["catalysts"]) > 0))}"""
        )
        print(f"Number of substrates: {len(set(self.dev_split))}")
        print("Test split:")
        print(f"Number of entries: {len(self.test)}")
        print(
            f"""Number of entries with catalysts: {len(self.test.filter(lambda x: len(x["catalysts"]) > 0))}"""
        )
        print(f"Number of substrates: {len(set(self.test_split))}")

    def return_gene_synonyms(self, symbol: str) -> List[str]:
        if symbol in SPECIAL_ENTITY_DICT:
            return SPECIAL_ENTITY_DICT[symbol]
        elif symbol in GENE_SYMBOL_TO_NCBI:
            return self.return_gene_synonyms(GENE_SYMBOL_TO_NCBI[symbol][1])
        try:
            return filter_short_synonyms(
                [self.ncbi_gene_db[f"{symbol}_9606"]["Symbol"]]
                + self.ncbi_gene_db[f"{symbol}_9606"]["Synonyms"]
                + [
                    self.ncbi_gene_db[f"{symbol}_9606"][
                        "Symbol_from_nomenclature_authority"
                    ]
                ]
                + [
                    self.ncbi_gene_db[f"{symbol}_9606"][
                        "Full_name_from_nomenclature_authority"
                    ]
                ]
            )
        except KeyError:
            if symbol not in self.not_found_symbols:
                self.not_found_symbols.add(symbol)
                print(f"Symbol {symbol} not found in the database")
            return [symbol]

    def extract_protein_name(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Extract protein name from the protein ID
        samples["substrate"] = [
            samples["protein_id"][i].split("_")[0]
            for i in range(len(samples["protein_id"]))
        ]
        return samples

    def add_substrate_synonyms(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples["substrate_synonyms"] = [
            self.extend_synonyms_with_hyphens(
                list(
                    set(
                        [
                            synonym
                            for gene in substrate_genes
                            for synonym in self.return_gene_synonyms(gene)
                        ]
                    )
                )
                + samples["synonyms"][i]
            )
            for i, substrate_genes in enumerate(samples["gene_names"])
        ]
        return samples

    def return_catalyst_synonyms(self, catalyst: str) -> List[str]:
        if f"{catalyst}_9606" in self.ncbi_gene_db:
            return self.return_gene_synonyms(catalyst)
        elif catalyst in SPECIAL_ENTITY_DICT:
            return SPECIAL_ENTITY_DICT[catalyst]
        elif catalyst in GENE_SYMBOL_TO_NCBI:
            return self.return_gene_synonyms(GENE_SYMBOL_TO_NCBI[catalyst][1])
        elif catalyst in NON_CATALYSTS:
            return [""]
        else:
            # raise ValueError(f"Unknown catalyst: {catalyst}")
            print(f"Unknown catalyst: {catalyst}")
            return [catalyst]

    def add_catalysts_synonyms(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples["catalysts_synonyms"] = [
            [
                self.extend_synonyms_with_hyphens(
                    self.return_catalyst_synonyms(catalyst)
                )
                for catalyst in catalysts
            ]
            for catalysts in samples["catalysts"]
        ]
        return samples

    def add_ptm_type_synonyms(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples["ptm_type_synonyms"] = [
            (
                PTM_MOD_RES_MAPPING_REVERSE[ptm_type] + PTM_SYNONYMS[ptm_type]
                if ptm_type in PTM_SYNONYMS
                else [ptm_type]
            )
            for ptm_type in samples["ptm_type"]
        ]
        return samples

    def extend_synonyms_with_hyphens(self, synonyms: Iterable[str]) -> List[str]:
        extended_synonyms = set()
        for synonym in synonyms:
            if "-" not in synonym:
                extended_synonyms.add(synonym)
            else:
                # Replace the last hyphen with a space
                synonym_with_space = (
                    synonym.rsplit("-", 1)[0] + " " + synonym.rsplit("-", 1)[1]
                )

                # Replace the last hyphen with an empty string
                synonym_without_hyphen = (
                    synonym.rsplit("-", 1)[0] + synonym.rsplit("-", 1)[1]
                )

                # Add the original synonym, the one with space, and the one without hyphen
                extended_synonyms.add(synonym_without_hyphen)
                extended_synonyms.add(synonym_with_space)
                extended_synonyms.add(synonym)

        return sorted(extended_synonyms)

    def filter_entity_synonyms_in_text(
        self, samples: Dict[str, Any], entity_type: str
    ) -> Dict[str, Any]:
        text_type = "evidence_" + self.mode
        entity_synonyms_in_text = f"{entity_type}_synonyms_in_" + self.mode
        entity_synonyms_index_text = f"{entity_type}_synonyms_index_" + self.mode
        samples[entity_synonyms_in_text] = []
        samples[entity_synonyms_index_text] = []
        for i, text in enumerate(samples[text_type]):
            samples[entity_synonyms_in_text].append([])
            samples[entity_synonyms_index_text].append([])

            entities_synonyms = samples[f"{entity_type}_synonyms"][i]
            if entities_synonyms == []:
                continue
            elif not isinstance(
                entities_synonyms[0], list
            ):  # if it's a single list of synonyms
                entities_synonyms = [entities_synonyms]  # wrap it in another list

            for j, entity_synonyms in enumerate(entities_synonyms):
                # Create a sublist for each entity's synonyms
                any_synonym_in_text = False
                samples[entity_synonyms_index_text][-1].append([])
                found_mentions: set[str] = set()
                entity = samples[entity_type][i]
                if type(entity) == list:  # Multiple entities
                    entity = entity[j]
                # synonym_subset = self.get_shortest_synonyms_subset(
                #     entity, entity_synonyms
                # )
                for synonym in sorted(entity_synonyms):
                    # if text.startswith(
                    #     "Identification of 14-3-3zeta as a protein kinase B/Akt substrate."
                    # ):
                    #     print(entity_type)
                    #     print(synonym)
                    for match in self.match_string_with_stops(synonym, text):
                        any_synonym_in_text = True
                        # break  # For the indexes, we need to find all mentions
                        # start_char = text.lower().find(synonym.lower())
                        # end_char = start_char + len(synonym)
                        # substring = text[start_char:end_char].lower()
                        substring = match.group(1)
                        start_char = match.start(1)
                        end_char = match.end(1)
                        if not self.is_substring_in_set(substring, found_mentions):
                            found_mentions.add(substring)
                            # Pyarrow expects same types for all entries, i.e., string
                            samples[entity_synonyms_index_text][-1][-1].append(
                                (substring, str(start_char), str(end_char))
                            )
                    # Heuristic: Check if all subwords are included in the text
                    # TODO: We need to check if multiple variations of the subword are
                    # included in the text
                    # all_subwords_in_text = self.check_all_subwords_in_text_with_stops(
                    #     synonym, text
                    # )
                    # if all_subwords_in_text:
                    #     any_synonym_in_text = True
                    #     # break  # For the indexes, we need to find all mentions
                    #     # Find positions of subword occurrences
                    #     synonym_subwords = re.split(r"\W+", synonym.lower())
                    #     substring, (start_char, end_char) = (
                    #         self.min_window_with_indices(text.lower(), synonym_subwords)
                    #     )
                    #     if (
                    #         not self.is_substring_in_set(substring, found_mentions)
                    #         and substring != ""
                    #         # Substring must not be too long (, i.e., span the whole doc)
                    #         and len(substring)
                    #         < self.maximum_synonym_length_ratio * len(synonym)
                    #     ):
                    #         found_mentions.add(substring)
                    #         # Pyarrow expects same types for all entries, i.e., string
                    #         samples[entity_synonyms_index_text][-1][-1].append(
                    #             (substring, str(start_char), str(end_char))
                    #         )
                samples[entity_synonyms_in_text][-1].append(any_synonym_in_text)
        # print(samples[entity_synonyms_index_text])
        # print(samples[entity_synonyms_in_text])
        return samples

    def filter_res_synonyms_in_text(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples["residue_synonyms_in_" + self.mode] = []
        samples["residue_synonyms_index_" + self.mode] = []

        text_type = "evidence_" + self.mode
        for i, text in enumerate(samples[text_type]):
            # Define acceptable surrounding/stop characters for amino acid codes
            stop_chars = r"[ \s,\.\(\)\[\]\{\}:;!\?\"\'\-]"

            # Construct regex patterns:
            # For one-letter codes, potentially followed by numbers if within certain bounds like parentheses
            residue = samples["residue"][i]
            one_letter_code = AMINO_ACIDS[residue][1]
            three_letter_code = AMINO_ACIDS[residue][0]

            one_letter_pattern = rf"""
                (?:{stop_chars})(?P<code>{one_letter_code})(?:\d*)(?:{stop_chars})
            """
            three_letter_pattern = (
                rf"(?:{stop_chars})(?P<code>{three_letter_code})(?:{stop_chars})"
            )

            # Find all occurrences of the one-letter code
            one_letter_matches = re.finditer(one_letter_pattern, text, re.IGNORECASE)
            three_letter_matches = re.finditer(
                three_letter_pattern, text, re.IGNORECASE
            )
            full_matches = re.finditer(residue, text, re.IGNORECASE)

            # Check if the residue is found in the text
            matched = False

            # Add the indices of all the matches
            residue_synonyms_index = []
            for match in one_letter_matches:
                residue_synonyms_index.append(
                    (
                        match.group("code"),
                        str(match.start("code")),
                        str(match.end("code")),
                    )
                )
                if not matched:
                    matched = True
                    samples["residue_synonyms_in_" + self.mode].append([True])
            for match in three_letter_matches:
                residue_synonyms_index.append(
                    (
                        match.group("code"),
                        str(match.start("code")),
                        str(match.end("code")),
                    )
                )
                if not matched:
                    matched = True
                    samples["residue_synonyms_in_" + self.mode].append([True])
            for match in full_matches:
                residue_synonyms_index.append(
                    (match.group(), str(match.start()), str(match.end()))
                )
                if not matched:
                    matched = True
                    samples["residue_synonyms_in_" + self.mode].append([True])
            samples["residue_synonyms_index_" + self.mode].append(
                [residue_synonyms_index]
            )

            if not matched:
                samples["residue_synonyms_in_" + self.mode].append([False])

        return samples

    def filter_pos_synonyms_in_text(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples["position_synonyms_in_" + self.mode] = []
        samples["position_synonyms_index_" + self.mode] = []

        text_type = "evidence_" + self.mode
        for i, text in enumerate(samples[text_type]):
            # Construct regex pattern for the position
            position = samples["position"][i]
            position_pattern = rf"\b{position}\b"

            # Find all occurrences of the position
            position_matches = re.finditer(position_pattern, text, re.IGNORECASE)

            # Check if the position is found in the text
            if position_matches:
                samples["position_synonyms_in_" + self.mode].append([True])
            else:
                samples["position_synonyms_in_" + self.mode].append([False])

            # Add the indices of all the matches
            position_synonyms_index = []
            for match in position_matches:
                position_synonyms_index.append(
                    (position, str(match.start()), str(match.end()))
                )
            samples["position_synonyms_index_" + self.mode].append(
                [position_synonyms_index]
            )

        return samples

    def split_catalysts(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        def clean_and_split(text):
            # Remove the starting " by "
            cleaned_text = re.sub(r"^\s*by\s*", "", text)
            # Split the text using the specified delimiters
            result = re.split(r",\s+| and | or ", cleaned_text)
            # Iterate over the results and remove the NON_CATALYSTS
            result = [catalyst for catalyst in result if catalyst not in NON_CATALYSTS]
            return result

        # Examplary catalyst
        # " by MAPKAPK2, MAPKAPK3 and MAPKAPK5"
        # Remove " by " and split by ", " and " and "
        samples["mod_res_catalyst_list"] = [
            clean_and_split(catalyst) for catalyst in samples["mod_res_catalyst"]
        ]
        return samples

    def split_mod_res_product(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Split the mod_res_product into mod_res and entity
        samples["mod_res"] = [
            MOD_RES_MAPPING.get(mod_res, ("UNKNOWN", "UNKNOWN"))[0]
            for mod_res in samples["mod_res_product"]
        ]
        samples["entity"] = [
            MOD_RES_MAPPING.get(mod_res, ("UNKNOWN", "UNKNOWN"))[1]
            for mod_res in samples["mod_res_product"]
        ]
        return samples

    def add_dummy_pmids(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Add empty PMIDs if none in the list entry
        samples["mod_res_pmids"] = [
            pmids if pmids else [-1] for pmids in samples["mod_res_pmids"]
        ]
        return samples

    def manual_explode_pmids(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Explode the PMIDs into separate entries
        # Make sure that all other columns are also exploded
        # print(len(samples["mod_res_pmids"]))
        # print(len(samples))
        # print(samples)
        new_samples = {}
        for key in samples.keys():
            if key != "mod_res_pmids":
                new_samples[key] = [
                    samples[key][i]
                    for i, pmids in enumerate(samples["mod_res_pmids"])
                    for _ in pmids
                ]
            else:
                new_samples[key] = [
                    pmid for pmids in samples["mod_res_pmids"] for pmid in pmids
                ]
        return new_samples

    def create_dataset(self) -> Dataset:
        # Initialize the current context
        current_protein_id = None
        primary_accession = None
        synonyms = []
        gene_names = []
        species_name = None
        tax_id = None

        mod_res_pos = None
        mod_res_evidence = None
        mod_res_note = None

        mod_res_capturing = False
        mod_res_capturing_evidence = False

        dataset_dicts = {
            "protein_id": [],
            "primary_accession": [],
            "synonyms": [],
            "gene_names": [],
            "species_name": [],
            "tax_id": [],
            "mod_res_pos": [],
            "mod_res_evidence": [],
            "mod_res_note": [],
            "full_name": [],
        }

        def process_mod_res_entry():
            # Also filter for human tax id
            if mod_res_pos and mod_res_note and mod_res_evidence and tax_id == "9606":
                dataset_dicts["protein_id"].append(current_protein_id)
                dataset_dicts["primary_accession"].append(primary_accession)
                dataset_dicts["synonyms"].append(synonyms)
                dataset_dicts["gene_names"].append(gene_names)
                dataset_dicts["species_name"].append(species_name)
                dataset_dicts["tax_id"].append(tax_id)
                dataset_dicts["mod_res_pos"].append(mod_res_pos)
                dataset_dicts["mod_res_evidence"].append(mod_res_evidence)
                dataset_dicts["mod_res_note"].append(mod_res_note)
                dataset_dicts["full_name"].append(full_name)

        # Iterate over the text file by line
        with open(self.file, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines):
            if line.startswith("ID"):
                # ID   BRAF_HUMAN              Reviewed;         766 AA.
                # New entry starts like this
                current_protein_id = line.split()[1]
                primary_accession = None
                synonyms = []
                gene_names = []
                full_name = None
                species_name = None
                tax_id = None
            elif line.startswith("AC"):
                # AC   P15056; A4D1T4; B6HY61; B6HY62; B6HY63; B6HY64; B6HY65; B6HY66; Q13878;
                # AC   Q3MIN6; Q9UDP8; Q9Y6T3;
                # Only parse the first AC numbers aka the primary accession number
                primary_accession = line.split(";")[0].split()[1]
            elif line.startswith("DE"):
                # DE   RecName: Full=Serine/threonine-protein kinase B-raf {ECO:0000305};
                # DE            EC=2.7.11.1 {ECO:0000269|PubMed:21441910, ECO:0000269|PubMed:29433126};
                # DE   AltName: Full=Proto-oncogene B-Raf;
                # DE   AltName: Full=p94;
                # DE   AltName: Full=v-Raf murine sarcoma viral oncogene homolog B1;
                # Parse RecName and AltName in Full and Short forms
                # Ignore other fields, particulary evidence ECO codes
                # Define one regex for Full and one for Short and ignore ECO codes
                regex_str = r"DE.*?(?:Full|Short)=(.*?)(?=\s+\{ECO:|\s*;|$)"
                synonym = re.search(regex_str, line)
                if synonym:
                    synonyms.append(synonym.group(1))
                    if not full_name:
                        full_name = synonym.group(1)
            elif line.startswith("GN"):
                # GN   Name=BRAF {ECO:0000312|HGNC:HGNC:1097}; Synonyms=BRAF1, RAFB1;
                # Parse Gene Name without ECO codes
                gene_names = re.findall(r"Name=(.*?)(?=\s+\{ECO:|\s*;|$)", line)
                if not gene_names:
                    gene_names = ["UNDEFINED"]
                    # print("Gene name not found")
                    # print(line)
                    # print(primary_accession)
                    # input()
                synonyms.extend(gene_names)
                gene_synonyms = re.findall(r"Synonyms=(.*?)(?=\s+\{ECO:|\s*;|$)", line)
                synonyms.extend(gene_synonyms)
            elif line.startswith("OS"):
                # OS   Homo sapiens (Human).
                # Parse species name
                species_name = line.split(maxsplit=1)[1].strip()
            elif line.startswith("OX"):
                # OX   NCBI_TaxID=9606;
                tax_id_match = re.search(r"NCBI_TaxID=(\d+);", line)
                if tax_id_match:
                    tax_id = tax_id_match.group(1)
                else:
                    tax_id = "-1"
                    # print("Tax ID not found")
            elif ft_match := re.compile(r"^FT\s+(\w+)\s+(\d+|\d+\.\.\d+)$").match(line):
                # FT   SITE            438..439
                # FT                   /note="Breakpoint for translocation to form KIAA1549-BRAF
                # FT                   fusion protein"
                # FT   MOD_RES         2
                # FT                   /note="N-acetylalanine"
                # FT                   /evidence="ECO:0000269|Ref.8"
                # New feature starting line
                ft_type = ft_match.group(1)
                if ft_type == "MOD_RES":
                    process_mod_res_entry()
                    mod_res_pos = ft_match.group(2)
                    mod_res_capturing = True
                else:
                    mod_res_capturing = False
                    if mod_res_pos and mod_res_note and mod_res_evidence:
                        process_mod_res_entry()
                    mod_res_pos = None
                mod_res_note = None
                mod_res_evidence = None
                mod_res_capturing_evidence = False
            elif line.startswith("FT") and mod_res_capturing:
                # Check if the line contains note or evidence
                note_match = re.search(r'/note="(.+?)"', line)
                evidence_match = re.search(r'/evidence="(.+)"?', line)
                if note_match:
                    mod_res_note = note_match.group(1)
                elif evidence_match:
                    mod_res_evidence = evidence_match.group(1)
                    mod_res_capturing_evidence = True
                elif (
                    mod_res_capturing_evidence and "ECO:" in line
                ):  # This handles continued lines of evidence
                    # Append continued evidence to the last entry if available
                    mod_res_evidence = mod_res_evidence + " " + line.strip("FT ")
            elif mod_res_capturing:  # Process mod_res entries
                process_mod_res_entry()
                mod_res_pos = None
                mod_res_note = None
                mod_res_evidence = None
                mod_res_capturing = False

        return Dataset.from_dict(dataset_dicts)

    def extract_pmids(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Extract PMIDs from the text
        samples["mod_res_pmids"] = [
            re.compile(r"PubMed:(\d+)").findall(evidences)
            for evidences in samples["mod_res_evidence"]
        ]
        # Change string PMIDs to integers
        samples["mod_res_pmids"] = [
            [int(pmid) for pmid in pmids] for pmids in samples["mod_res_pmids"]
        ]
        return samples

    def split_notes(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        # Split the notes into product and catalyst
        samples["mod_res_product"] = [
            note.split(";")[0] for note in samples["mod_res_note"]
        ]
        samples["mod_res_catalyst"] = [
            (
                (note.split(";")[1])
                if len(note.split(";")) > 1
                and not (note.split(";")[1]).startswith(" in")
                else ""
            )
            for note in samples["mod_res_note"]
        ]
        return samples

    def detailed_dataset_stats(self):
        examples_with_catalysts = self.dataset.filter(
            lambda example: len(example["catalysts"]) > 0
        )
        train_with_catalysts = self.train.filter(
            lambda example: len(example["catalysts"]) > 0
        )
        dev_with_catalysts = self.dev.filter(
            lambda example: len(example["catalysts"]) > 0
        )
        test_with_catalysts = self.test.filter(
            lambda example: len(example["catalysts"]) > 0
        )
        # Sum up the examples with catalysts
        number_of_examples_with_catalysts = sum(
            [len(example["catalysts"]) for example in examples_with_catalysts]
        )
        number_of_examples_with_variants = len(
            self.dataset.filter(
                lambda example: len(example["ptm_type"]) > 0
                or len(example["residue"]) > 0
                or len(example["position"]) > 0
            )
        )
        number_of_examples_with_substrates = len(
            self.dataset.filter(lambda example: len(example["substrate"]) > 0)
        )
        examples_with_all_entities = self.dataset.filter(
            lambda example: len(example["substrate"]) > 0
            and (
                len(example["ptm_type"]) > 0
                or len(example["residue"]) > 0
                or len(example["position"]) > 0
            )
            and len(example["catalysts"]) > 0
        )
        number_of_entries_with_all_entities = sum(
            [
                min(1, len(example["catalysts"]))
                for example in examples_with_all_entities
            ]
        )
        number_of_examples_with_all_entities = sum(
            [len(example["catalysts"]) for example in examples_with_all_entities]
        )

        number_of_unique_substrates_train = len(
            set([example["substrate"] for example in self.train])
        )
        number_of_unique_substrates_dev = len(
            set([example["substrate"] for example in self.dev])
        )
        number_of_unique_substrates_test = len(
            set([example["substrate"] for example in self.test])
        )
        number_of_unique_substrate_variants_train = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in self.train
                ]
            )
        )
        number_of_unique_substrate_variants_train_with_catalysts = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in train_with_catalysts
                ]
            )
        )
        number_of_unique_substrate_variants_dev = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in self.dev
                ]
            )
        )
        number_of_unique_substrate_variants_dev_with_catalysts = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in dev_with_catalysts
                ]
            )
        )
        number_of_unique_substrate_variants_test = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in self.test
                ]
            )
        )
        number_of_unique_substrate_variants_test_with_catalysts = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                    )
                    for example in test_with_catalysts
                ]
            )
        )

        number_of_unique_ptm_types = len(
            set([example["ptm_type"] for example in self.dataset])
        )

        number_of_unique_substrate_variants_pmid_train = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        example["citation_id"],
                    )
                    for example in train_with_catalysts
                ]
            )
        )
        number_of_unique_substrate_variants_pmid_dev = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        example["citation_id"],
                    )
                    for example in dev_with_catalysts
                ]
            )
        )
        number_of_unique_substrate_variants_pmid_test = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        example["citation_id"],
                    )
                    for example in test_with_catalysts
                ]
            )
        )
        number_of_unique_substrate_variants_catalyst_train = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        catalyst,
                    )
                    for example in train_with_catalysts
                    for catalyst in example["catalysts"]
                ]
            )
        )
        number_of_unique_substrate_variants_catalyst_dev = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        catalyst,
                    )
                    for example in dev_with_catalysts
                    for catalyst in example["catalysts"]
                ]
            )
        )
        number_of_unique_substrate_variants_catalyst_test = len(
            set(
                [
                    (
                        example["substrate"],
                        example["ptm_type"],
                        example["residue"],
                        example["position"],
                        catalyst,
                    )
                    for example in test_with_catalysts
                    for catalyst in example["catalysts"]
                ]
            )
        )

        details = [
            "Number of unique entries/Number of entries counting all catalysts separately",
            f"Total number of examples: {len(self.dataset)}",
            f"Train examples: {len(self.train)}",
            f"Dev examples: {len(self.dev)}",
            f"Test examples: {len(self.test)}",
            f"Number of unique substrates in train: {number_of_unique_substrates_train}",
            f"Number of unique substrates in dev: {number_of_unique_substrates_dev}",
            f"Number of unique substrates in test: {number_of_unique_substrates_test}",
            f"Number of unique substrate-variant pairs in train: {number_of_unique_substrate_variants_train}",
            f"  - with catalysts: {number_of_unique_substrate_variants_train_with_catalysts}",
            f"Number of unique substrate-variant pairs in dev: {number_of_unique_substrate_variants_dev}",
            f"  - with catalysts: {number_of_unique_substrate_variants_dev_with_catalysts}",
            f"Number of unique substrate-variant pairs in test: {number_of_unique_substrate_variants_test}",
            f"  - with catalysts: {number_of_unique_substrate_variants_test_with_catalysts}",
            f"Number of examples with a substrate entity: {number_of_examples_with_substrates}",
            "  ",
            f"Number of unique PTM types: {number_of_unique_ptm_types}",
            "  ",
            f"Number of unique substrate-variant pairs with PMIDs in train: {number_of_unique_substrate_variants_pmid_train}",
            f"Number of unique substrate-variant pairs with PMIDs in dev: {number_of_unique_substrate_variants_pmid_dev}",
            f"Number of unique substrate-variant pairs with PMIDs in test: {number_of_unique_substrate_variants_pmid_test}",
            "  ",
            f"Number of unique substrate-variant-catalyst triples in train: {number_of_unique_substrate_variants_catalyst_train}",
            f"Number of unique substrate-variant-catalyst triples in dev: {number_of_unique_substrate_variants_catalyst_dev}",
            f"Number of unique substrate-variant-catalyst triples in test: {number_of_unique_substrate_variants_catalyst_test}",
            "  ",
            self.return_stats(
                ["substrate_synonyms_in_" + self.mode],
            ),
            self.return_stats(
                ["substrate_in_" + self.mode],
            ),
            f"Number of examples with a variant entity: {number_of_examples_with_variants}",
            f"Number of examples with at least one catalyst: {number_of_examples_with_catalysts}",
            self.return_stats(
                ["catalysts_synonyms_in_" + self.mode],
            ),
            self.return_stats(
                ["catalysts_in_" + self.mode],
            ),
            f"Number of examples with all entities: {number_of_entries_with_all_entities}/{number_of_examples_with_all_entities}",
            # self.dataset[:1],
        ]

        return "\n".join(details)


if __name__ == "__main__":
    # example = CiVICSynonymExamples()  # Used for the retriever

    # https://huggingface.co/docs/datasets/cache
    from datasets import disable_caching

    disable_caching()

    # Get some statistics about the datasets
    group_by_citation_id = False

    examples = UniProtPTMExamples(
        bool_group_by_citation_id=group_by_citation_id, cache=True, mode="raw_text"
    )  # Used for the reader
    # for i, entry in enumerate(examples.dataset):
    #     if entry["catalysts"] != ["autocatalysis"]:
    #         continue
    #     print(i)
    #     print(entry)
    #     continue_ = input("Continue?")
    #     if continue_ == "n":
    #         break
    print(examples.dataset[0])
    print(examples.detailed_dataset_stats())
