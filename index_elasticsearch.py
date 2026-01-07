from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, load_from_disk
from elasticsearch import Elasticsearch, helpers

from index_pubmed import process_pubmed_xmls


class Indexer:
    def __init__(self, dataset_file, es_hosts, index_name, num_workers, max_docs=-1):
        self.es = Elasticsearch(
            es_hosts,
            request_timeout=60,
            basic_auth=("elastic", "ROkvX4e9CFV5xFIWBJmI"),
            verify_certs=True,
            ca_certs="edel_repo_cache/elasticsearch-8.11.1/config/certs/http_ca.crt",
        )
        self.index_name = index_name
        self.max_docs = max_docs
        self.num_workers = num_workers

        # Load dataset
        if Path(dataset_file).exists():
            print("Loading dataset from local cache")
            self.dataset = load_from_disk(dataset_file)
            print("Dataset loaded")
        else:
            shards = [
                f"edel_repo_cache/pubmed/pubmed23n{i:04d}.xml"
                for i in range(1, 1167)
            ]
            self.dataset = Dataset.from_generator(
                process_pubmed_xmls,
                gen_kwargs={"shards": shards},
                num_proc=16,
            )
            self.dataset.save_to_disk(dataset_file)

    def read_data(self, start, end):
        # Replace with your data loading logic
        for i in range(start, end):
            yield {
                "pmid": self.dataset[i]["pmid"],
                "title": self.dataset[i]["title"],
                "abstract": self.dataset[i]["abstract"],
                "text": self.dataset[i]["title"] + ". " + self.dataset[i]["abstract"],
                "year": self.dataset[i]["year"][0],
                "month": self.dataset[i]["month"][0],
                "day": self.dataset[i]["day"][0],
            }

    def generate_actions(self, abstracts):
        for abstract in abstracts:
            yield {
                "_index": self.index_name,
                "_id": abstract["pmid"],
                "_source": abstract,
            }

    def bulk_index(self, start, end):
        abstracts = self.read_data(start, end)
        helpers.bulk(
            self.es, self.generate_actions(abstracts), chunk_size=1000
        )  # Adjust chunk_size based on your environment
        print(f"Indexed documents from {start} to {end}")

    def run(self):
        if self.max_docs > -1:
            docs_per_worker = min(self.max_docs, len(self.dataset) // self.num_workers)
        else:
            docs_per_worker = len(self.dataset) // self.num_workers

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.bulk_index, i, min(i + docs_per_worker, len(self.dataset))
                )
                for i in range(0, len(self.dataset), docs_per_worker)
            ]

            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    from datasets.fingerprint import disable_caching
    disable_caching()
    shards = [
        f"edel_repo_cache/pubmed/pubmed23n{i:04d}.xml"
        for i in range(1, 2)
    ]

    indexer = Indexer(
        "edel_repo_cache/datasets/pubmed.dataset",
        "https://localhost:9200",
        "20231127_pubmed",
        4,
    )  # Adjust parameters as needed
    indexer.run()
