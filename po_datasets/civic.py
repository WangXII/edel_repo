from typing import Any

from datasets import Dataset, load_dataset

from .dataset import DatasetExamples


class CiVICExamples(DatasetExamples):
    def __init__(
        self,
        file: str = "01-Nov-2022-ClinicalEvidenceSummaries.tsv",
        mode="raw_full_text",
        filter_pubmed: bool = True,
        bool_group_by_citation_id: bool = False,
        bool_group_by_alteration: bool = False,
        cache_dir_prefix: str = "edel_repo_cache/civic/examples_",
        cache: bool = True,
    ):
        super().__init__(
            file=file,
            mode=mode,
            filter_pubmed=filter_pubmed,
            bool_group_by_citation_id=bool_group_by_citation_id,
            bool_group_by_alteration=bool_group_by_alteration,
            cache_dir_prefix=cache_dir_prefix,
            cache=cache,
        )

    def create_dataset(self) -> Dataset:
        # Load the csv data
        dataset = load_dataset(
            "csv",
            data_files=self.file,
            delimiter="\t",
            split="train",
        ).filter(lambda x: x["source_type"] == "PubMed")
        # TODO: Add other source types like ASCO

        return dataset


if __name__ == "__main__":
    # example = CiVICSynonymExamples()  # Used for the retriever

    # https://huggingface.co/docs/datasets/cache
    from datasets import disable_caching

    disable_caching()

    # Get some statistics about the datasets
    group_by_citation_id = False

    example = CiVICExamples(
        bool_group_by_citation_id=group_by_citation_id, cache=True, mode="raw_text"
    )  # Used for the reader
    print(example.dataset[0])
    print(example.detailed_dataset_stats())
