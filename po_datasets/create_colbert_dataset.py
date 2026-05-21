import random
import re
from pathlib import Path
from typing import Optional

import argparse
import numpy as np
import polars as pl
import ujson
from datasets import load_from_disk


def _extract_entity_from_query(query: str) -> str:
    # Treatment:
    # Treatment for gene XRCC1 and variant Q399R.
    m = re.search(r"Treatment for gene\s+(.+?)\s+and variant", query, flags=re.I)
    if m:
        return m.group(1).strip()

    # PTM with full name:
    # Catalyst for the phosphorylation of Cyclin Y (CCNY) at serine position 73.
    m = re.search(
        r"Catalyst for the\s+.+?\s+of\s+.+?\((.+?)\)\s+at\s+.+?\s+position\s+\d+",
        query,
        flags=re.I,
    )
    if m:
        return m.group(1).strip()

    # PTM without full name:
    # Catalyst for the phosphorylation of CCNY at serine position 73.
    m = re.search(
        r"Catalyst for the\s+.+?\s+of\s+(.+?)\s+at\s+.+?\s+position\s+\d+",
        query,
        flags=re.I,
    )
    if m:
        return m.group(1).strip()

    raise ValueError(f"Could not extract entity from query: {query}")


def _add_ptm_full_name_to_query(query: str, substrate_full_name: str) -> str:
    if not query.lower().startswith("catalyst for the") or "(" in query:
        return query

    m = re.match(
        r"(Catalyst for the\s+.+?\s+of\s+)(.+?)(\s+at\s+.+?\s+position\s+\d+\.)",
        query,
        flags=re.I,
    )
    if not m:
        return query

    substrate_name = m.group(2).strip()
    return f"{m.group(1)}{substrate_full_name} ({substrate_name}){m.group(3)}"


def _split_title_abstract(text: str) -> tuple[str, str]:
    parts = text.split(". ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _load_pubmed_text_df(pubmed_cache_file: str) -> pl.DataFrame:
    print("Loading PubMed cache")
    pubmed_cache = load_from_disk(pubmed_cache_file)
    df = pl.from_arrow(pubmed_cache.data.table)

    text_col = (
        pl.when((pl.col("title") != "") & (pl.col("abstract") != ""))
        .then(pl.col("title") + ". " + pl.col("abstract"))
        .when(pl.col("title") == "")
        .then(pl.col("abstract"))
        .otherwise(pl.col("title"))
        .alias("text")
    )

    return df.with_columns(text_col).select(["pmid", "text"])


def _get_bm25_cache_matches(
    bm25_cache: dict,
    dataset_name: str,
    substrate_name: str,
) -> tuple[list[tuple[int, str]], Optional[str]]:
    prefix = f"{dataset_name}_{substrate_name}_".lower()
    candidates = {}
    substrate_full_name = None

    for key, pmids in bm25_cache.items():
        if key.startswith("pmid_"):
            continue

        if key.lower().startswith(prefix):
            if substrate_full_name is None:
                substrate_full_name = key[len(f"{dataset_name}_{substrate_name}_"):]

            for pmid in pmids:
                text = bm25_cache.get("pmid_" + str(pmid))
                if text:
                    candidates[int(pmid)] = text

    return list(candidates.items()), substrate_full_name


def create_colbert_datasets(mode="strict_pos", dataset_name="civic_oncokb", dataset_split="train", num_repeats=8, batch_size=48):
    input_dir = f'/vol/tmp/wangxida/treatment_explorer/medcpt_fix_splits_{mode}_examples/'
    output_dir = f'/vol/tmp/wangxida/treatment_explorer/colbert_datasets_{mode}_examples/'
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    qid2info_json = input_dir + f'qid2info_{dataset_name}_{dataset_split}.json'
    queries_out = output_dir + f'{dataset_name}_{dataset_split}_queries.tsv'
    qid_map = {}
    with open(qid2info_json, 'r') as f:
        qid2info = ujson.load(f)
    # Write .tsv
    with open(queries_out, 'w') as f:
        for i, (qid, query) in enumerate(qid2info.items()):
            qid_map[int(qid)] = i
            f.write(f"{i}\t{query}\t{qid}\n")

    pmid2info_json = input_dir + f'pmid2info_{dataset_name}_{dataset_split}.json'
    collection_out = output_dir + f'{dataset_name}_{dataset_split}_collection.tsv'
    pmid_map = {}
    with open(pmid2info_json, 'r') as f:
        pmid2info = ujson.load(f)
    # Write .tsv
    with open(collection_out, 'w') as f:
        for i, (pmid, info) in enumerate(pmid2info.items()):
            pmid_map[int(pmid)] = i
            f.write(f"{i}\t{info[1]}\t{info[0]}\t{pmid}\n")

    pair_jsonl = input_dir + f'train_{dataset_name}_{dataset_split}.jsonl'
    json_out = output_dir + f'{dataset_split}_{dataset_name}_examples.json'
    rel_map = {}
    corpus_length = len(pmid_map)
    with open(pair_jsonl, 'r') as f:
        for line in f:
            example = ujson.loads(line)
            # print(qid_map)
            qid = qid_map[int(example['qid'])]
            doc_id = pmid_map[int(example['pmid'])]
            rel_map.setdefault(qid, []).append(doc_id)
    with open(json_out, 'w') as out_f:
        # Grab a random negative from the corpus
        # But make sure it's not a positive
        overflow_queue = []
        current_batch = []
        current_batch_qids = set()
        for _ in range(num_repeats):
            for qid in np.random.permutation(list(rel_map.keys())):
                pos_doc_ids = rel_map[qid]
                for pos_doc_id in pos_doc_ids:
                    neg_doc_id = np.random.randint(0, corpus_length)
                    while neg_doc_id in pos_doc_ids:
                        neg_doc_id = np.random.randint(0, corpus_length)
                    out_list = [int(qid), pos_doc_id, neg_doc_id]
                    if qid in current_batch_qids or len(current_batch) >= batch_size:
                        overflow_queue.append(out_list)
                    else:
                        current_batch.append(out_list)
                        current_batch_qids.add(qid)
                if len(current_batch) == batch_size:
                    for out_list in current_batch:
                        out_f.write(ujson.dumps(out_list) + '\n')
                    current_batch = []
                    current_batch_qids = set()
                    # Try to add as many examples from the overflow queue as possible
                    new_overflow_queue = []
                    for el in overflow_queue:
                        if el[0] not in current_batch_qids and len(current_batch) < batch_size:
                            current_batch.append(el)
                            current_batch_qids.add(el[0])
                        else:
                            new_overflow_queue.append(el)
                    overflow_queue = new_overflow_queue
        # Write any remaining examples in current_batch
        for out_list in current_batch:
            out_f.write(ujson.dumps(out_list) + '\n')
        for out_list in overflow_queue:
            if out_list[0] not in current_batch_qids:  # Just do a simple check for last few examples, to avoid duplicates
                out_f.write(ujson.dumps(out_list) + '\n')
                current_batch_qids.add(out_list[0])


def create_colbert_dataset_explicit_negs(
    mode="strict_pos",
    dataset_name="civic_oncokb",
    dataset_split="train",
    num_repeats=8,
    batch_size=48,
    bm25_cache_file="/vol/tmp/wangxida/tmp/treatment_explorer_bm25_query_cache.json",
    pubmed_cache_file="/vol/tmp/wangxida/datasets/20250113_pubmed.dataset",
    seed=42,
):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    input_dir = f"/vol/tmp/wangxida/treatment_explorer/medcpt_fix_splits_{mode}_examples/"
    output_dir = f"/vol/tmp/wangxida/treatment_explorer/colbert_datasets_{mode}_examples/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    qid2info_json = input_dir + f"qid2info_{dataset_name}_{dataset_split}.json"
    pmid2info_json = input_dir + f"pmid2info_{dataset_name}_{dataset_split}.json"
    pair_jsonl = input_dir + f"train_{dataset_name}_{dataset_split}.jsonl"

    queries_out = output_dir + f"{dataset_name}_{dataset_split}_explicit_negs_queries.tsv"
    collection_out = output_dir + f"{dataset_name}_{dataset_split}_explicit_negs_collection.tsv"
    json_out = output_dir + f"{dataset_split}_{dataset_name}_explicit_negs_examples.json"

    with open(bm25_cache_file, "r") as f:
        bm25_cache = ujson.load(f)

    with open(qid2info_json, "r") as f:
        qid2info = ujson.load(f)

    qid_map = {}
    qid_to_query = {}
    qid_to_entity = {}

    with open(queries_out, "w") as f:
        for i, (qid, query) in enumerate(qid2info.items()):
            original_qid = int(qid)
            entity_name = _extract_entity_from_query(query)

            _, substrate_full_name = _get_bm25_cache_matches(
                bm25_cache=bm25_cache,
                dataset_name=dataset_name,
                substrate_name=entity_name,
            )

            if substrate_full_name:
                query = _add_ptm_full_name_to_query(query, substrate_full_name)

            qid_map[original_qid] = i
            qid_to_query[i] = query
            qid_to_entity[i] = entity_name

            f.write(f"{i}\t{query}\t{qid}\n")

    with open(pmid2info_json, "r") as f:
        pmid2info = ujson.load(f)

    pmid_map = {}
    collection_rows = []

    for i, (pmid, info) in enumerate(pmid2info.items()):
        pmid_int = int(pmid)
        pmid_map[pmid_int] = i
        collection_rows.append((i, info[1], info[0], pmid_int))

    rel_map = {}

    with open(pair_jsonl, "r") as f:
        for line in f:
            example = ujson.loads(line)
            qid = qid_map[int(example["qid"])]
            doc_id = pmid_map[int(example["pmid"])]
            rel_map.setdefault(qid, []).append(doc_id)

    docid_to_pmid = {doc_id: pmid for pmid, doc_id in pmid_map.items()}

    rel_pmids = {
        qid: {docid_to_pmid[doc_id] for doc_id in doc_ids}
        for qid, doc_ids in rel_map.items()
    }

    pubmed_df = _load_pubmed_text_df(pubmed_cache_file)
    pubmed_len = pubmed_df.height

    bm25_candidates_by_qid = {}

    for qid in rel_map:
        entity_name = qid_to_entity[qid]

        candidates, _ = _get_bm25_cache_matches(
            bm25_cache=bm25_cache,
            dataset_name=dataset_name,
            substrate_name=entity_name,
        )

        candidates = [
            (pmid, text)
            for pmid, text in candidates
            if pmid not in rel_pmids[qid]
        ]

        rng.shuffle(candidates)
        bm25_candidates_by_qid[qid] = candidates

    def add_doc_to_collection(pmid: int, text: str) -> int:
        if pmid in pmid_map:
            return pmid_map[pmid]

        title, abstract = _split_title_abstract(text)
        doc_id = len(pmid_map)
        pmid_map[pmid] = doc_id
        collection_rows.append((doc_id, abstract, title, pmid))
        return doc_id

    def sample_random_pubmed_neg(qid: int) -> int:
        while True:
            idx = int(np_rng.integers(0, pubmed_len))
            row = pubmed_df.row(idx, named=True)
            pmid = int(row["pmid"])

            if pmid in rel_pmids[qid]:
                continue

            return add_doc_to_collection(pmid, row["text"])

    def sample_bm25_neg(qid: int) -> Optional[int]:
        candidates = bm25_candidates_by_qid[qid]

        while candidates:
            pmid, text = candidates.pop()

            if pmid not in rel_pmids[qid]:
                return add_doc_to_collection(pmid, text)

        return None

    triples = [[] for _ in range(num_repeats)]

    for qid in np_rng.permutation(list(rel_map.keys())):
        pos_doc_ids = rel_map[qid]

        for pos_doc_id in pos_doc_ids:
            neg_sources = (
                ["bm25"] * (num_repeats // 2)
                + ["random"] * (num_repeats - num_repeats // 2)
            )
            rng.shuffle(neg_sources)

            for i, source in enumerate(neg_sources):
                neg_doc_id = None

                if source == "bm25":
                    neg_doc_id = sample_bm25_neg(qid)

                if neg_doc_id is None:
                    neg_doc_id = sample_random_pubmed_neg(qid)

                triples[i].append([int(qid), int(pos_doc_id), int(neg_doc_id)])
    

    # Shuffle each triples sublist, then concatenate
    for sublist in triples:
        rng.shuffle(sublist)
    triples = [item for sublist in triples for item in sublist]

    def clean_tsv_field(x):
        if x is None:
            return ""
        return str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()

    with open(collection_out, "w") as f:
        for doc_id, abstract, title, pmid in collection_rows:
            abstract = clean_tsv_field(abstract)
            title = clean_tsv_field(title)
            f.write(f"{doc_id}\t{abstract}\t{title}\t{pmid}\n")

    with open(json_out, "w") as out_f:
        overflow_queue = []
        current_batch = []
        current_batch_qids = set()

        for out_list in triples:
            qid = out_list[0]

            if qid in current_batch_qids or len(current_batch) >= batch_size:
                overflow_queue.append(out_list)
            else:
                current_batch.append(out_list)
                current_batch_qids.add(qid)

            if len(current_batch) == batch_size:
                for el in current_batch:
                    out_f.write(ujson.dumps(el) + "\n")

                current_batch = []
                current_batch_qids = set()

                new_overflow_queue = []
                # rng.shuffle(overflow_queue)

                for el in overflow_queue:
                    if el[0] not in current_batch_qids and len(current_batch) < batch_size:
                        current_batch.append(el)
                        current_batch_qids.add(el[0])
                    else:
                        new_overflow_queue.append(el)

                overflow_queue = new_overflow_queue

        for el in current_batch:
            out_f.write(ujson.dumps(el) + "\n")

        for out_list in overflow_queue:
            if out_list[0] not in current_batch_qids:  # Just do a simple check for last few examples, to avoid duplicates
                out_f.write(ujson.dumps(out_list) + '\n')
                current_batch_qids.add(out_list[0])

if __name__ == "__main__":
    mode = "strict_pos"
    dataset_name = "uniprot"
    dataset_split = "train"

    batch_size = 48
    num_repeats = 8

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="civic_oncokb")
    parser.add_argument("--dataset_split", type=str, default="train")

    args = parser.parse_args()

    # create_colbert_datasets(mode=mode, dataset_name=dataset_name, dataset_split=dataset_split, num_repeats=num_repeats, batch_size=batch_size)
    create_colbert_dataset_explicit_negs(mode=mode, dataset_name=args.dataset_name, dataset_split=args.dataset_split, num_repeats=num_repeats, batch_size=batch_size)
