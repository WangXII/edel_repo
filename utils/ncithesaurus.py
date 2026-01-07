from typing import Dict, List, Set

import ahocorasick
import pandas as pd


class NCITheSaurusMapper:
    def __init__(self):
        self.mapper = pd.read_csv("Thesaurus.txt", delimiter="\t", header=None)
        column_headings = [
            "code",
            "concept IRI",
            "parents",
            "synonyms",
            "definition",
            "display name",
            "concept status",
            "semantic type",
            "concept in subset",
        ]
        self.mapper.columns = column_headings
        self.mapper = self.add_child_concept_mapping(self.mapper)

    def add_child_concept_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        child_mapping: Dict[str, List[str]] = {}
        for i, row in df.iterrows():
            for parent in str(row["parents"]).split("|"):
                if parent not in child_mapping:
                    child_mapping[parent] = []
                child_mapping[parent].append(row["code"])
        df["child_mapping"] = df["code"].apply(
            lambda x: child_mapping[x] if x in child_mapping else []
        )
        return df

    def find_child_concepts_recursively(
        self, df: pd.DataFrame, nci_thesaurus_codes: Set
    ) -> Set[str]:
        all_child_concepts: Set[str] = set()
        parent_concepts = nci_thesaurus_codes
        while (
            True
        ):  # As long as they are children add them to list, Breadth-first search
            # Do not assume that they use a DAG structure, so there may be cycles
            child_concepts_list = df.loc[df["code"].isin(parent_concepts)][
                "child_mapping"
            ].tolist()
            child_concepts: set[str] = set(
                [item for sublist in child_concepts_list for item in sublist]
            )
            parent_concepts = child_concepts.difference(all_child_concepts)
            all_child_concepts.update(child_concepts)
            if len(parent_concepts) == 0:
                break
        return all_child_concepts

    def get_synonyms_from_concepts(
        self, df: pd.DataFrame, nci_thesaurus_codes: Set
    ) -> Set[str]:
        synonyms_list = df.loc[df["code"].isin(nci_thesaurus_codes)][
            "synonyms"
        ].tolist()
        synonyms = set()
        for synonym_str in synonyms_list:
            for item in synonym_str.split("|"):
                synonyms.add(item.lower())
        return synonyms

    def build_automaton(self, patterns) -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for pattern in patterns:
            # Has to be exact match, so we add whitespace to the beginning and end
            automaton.add_word(" " + pattern + " ", pattern)
        automaton.make_automaton()
        return automaton

    def has_pattern(self, text, automaton) -> str:
        for _, pattern in automaton.iter(text):
            return pattern
        return ""
