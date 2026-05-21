from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class MarginConfig:
    name: str
    bm25_margin_values_tuple: List[Tuple[str, List]]
    margin_value_dict: Dict[str, float]
    margin_fn: Callable


# Minimum margin value is 0, maximum margin value is 2
# This is the cosine distance that we use
# Since cosine is not a linear function, we need to map values if we want to
# preserve the angular relationships

margin_classes_v1 = MarginConfig(
    name="margin_classes_v1",
    # Ascending order is important to not repeat the same PubMed IDs!
    bm25_margin_values_tuple=[],
    margin_value_dict={
        "positive": 0,
        "positive_two_entities_treatment_match": 0.2,
        "positive_gene_variant_match": 0.6,
        "positive_one_entity_match": 1.0,
        "positive_no_entities_match": 1.2,
        "negative_same_gene_not_other_variant_any_treatment_match": 0.2,
        "negative_same_gene_other_variant_any_treatment_match": 0.6,
        "negative_same_gene_variant_no_treatment_match": 0.6,
        "negative_same_gene_other_variant_no_treatment_match": 0.8,
        "negative_other_gene_any_treatment_match": 0.8,
        "negative_other_gene_no_treatment_match": 1.0,
        # New negative
        "negative_same_substrate_bm25": 0.8,
        "negative_pubmed": 1.0,
    },
    margin_fn=lambda x: 1 - np.cos(x * np.pi / 2),
)

