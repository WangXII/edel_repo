import json

from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

dataset_file = "edel_repo_cache/datasets/pubmed.dataset"
dataset = load_from_disk(dataset_file)

num_shards = 8
dataset = concatenate_datasets(
    [dataset.shard(num_shards, i) for i in range(num_shards)]
)

# abstracts_text = [x["title"] + "[SEP]" + x["abstract"] for x in tqdm(dataset)]
abstracts_id = [int(x["pmid"]) for x in tqdm(dataset)]

# Map the abstracts_id to the index
abstract_mapping = {}
for idx, abstract_id in tqdm(enumerate(abstracts_id)):
    abstract_mapping[abstract_id] = idx
# Save the mapping as simple json file
with open(
    "edel_repo_cache/faiss_pmid_sharded_"
    + str(num_shards)
    + "_mapping.json",
    "w",
    encoding="utf-8",
) as fOut:
    json.dump(abstract_mapping, fOut)
