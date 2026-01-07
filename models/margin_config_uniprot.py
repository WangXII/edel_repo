import numpy as np

from models.margin_config import MarginConfig

margin_classes_uniprot_v1 = MarginConfig(
    name="margin_classes_uniprot_v1",
    # Ascending order is important to not repeat the same PubMed IDs!
    bm25_margin_values_tuple=[],
    margin_value_dict={
        "positive": 0,
        "positive_three_entities_catalyst_match": 0.4,
        "positive_substrate_catalyst_match": 0.6,
        "positive_three_entities_no_catalyst_match": 0.6,
        "positive_ptm_catalyst_match": 0.6,
        "positive_other_two_entities_match": 0.6,
        "positive_substrate_match": 0.8,
        "positive_other_one_entity_match": 0.8,
        "positive_no_entities_match": 1.0,
        # Any catalyst matches
        "negative_same_substrate_ptm_res_no_other_pos_any_catalyst_match": 0.4,
        "negative_same_substrate_ptm_res_other_pos_any_catalyst_match": 0.4,
        "negative_same_substrate_ptm_no_other_respos_any_catalyst_match": 0.4,
        "negative_same_substrate_ptm_other_res_no_other_pos_any_catalyst_match": 0.4,
        "negative_same_substrate_ptm_other_respos_any_catalyst_match": 0.4,
        "negative_same_substrate_no_other_ptm_respos_any_catalyst_match": 0.6,
        "negative_same_substrate_other_ptm_no_other_respos_any_catalyst_match": 0.6,
        "negative_same_substrate_other_ptm_res_no_other_pos_any_catalyst_match": 0.6,
        "negative_same_substrate_other_ptm_respos_any_catalyst_match": 0.6,
        "negative_same_no_substrate_any_catalyst_match": 0.8,
        # No catalyst matches
        "negative_same_substrate_ptm_res_no_other_pos_no_catalyst_match": 0.6,
        "negative_same_substrate_ptm_res_other_pos_no_catalyst_match": 0.6,
        "negative_same_substrate_ptm_no_other_respos_no_catalyst_match": 0.6,
        "negative_same_substrate_ptm_other_res_no_other_pos_no_catalyst_match": 0.6,
        "negative_same_substrate_ptm_other_respos_no_catalyst_match": 0.6,
        "negative_same_substrate_other_ptm_no_other_respos_no_catalyst_match": 0.8,
        "negative_same_substrate_other_ptm_res_no_other_pos_no_catalyst_match": 0.8,
        "negative_same_substrate_other_ptm_respos_no_catalyst_match": 0.8,
		"negative_same_no_substrate_no_catalyst_match": 1.0,
        # Other substrate matches
        "negative_other_substrate_any_catalyst_match": 0.8,
        "negative_other_substrate_no_catalyst_match": 1.0,
        # BM25 negatives
        "negative_same_substrate_not_uniprot_bm25": 0.8,
        # BioASQ negatives
        "negative_bioasq": 1.2,
    },
    margin_fn=lambda x: 1 - np.cos(x * np.pi / 2),
)
