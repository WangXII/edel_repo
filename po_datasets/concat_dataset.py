from typing import List

import numpy as np
from datasets import Dataset, concatenate_datasets

from po_datasets.dataset import DatasetExamples


class ConcatExamples(DatasetExamples):
    """
    Concatenate multiple datasets

    """

    def __init__(
        self,
        datasets: List[DatasetExamples],
        make_new_datasplits: bool = True,
        mode="raw_full_text",
    ):
        # print(datasets[0].dataset[:5])
        # print(datasets[0].dataset)
        # We take the CiVIC data splits as our baseline model was trained on it
        self.dataset = concatenate_datasets(
            [self.select_columns(dataset.dataset, mode=mode) for dataset in datasets]
        )
        # print(datasets)
        # print(datasets[0].dataset.features)
        # print(datasets[1].dataset.features)
        print(self.dataset.features)
        print(len(datasets[0].train))
        print(len(datasets[0].dev))
        print(len(datasets[0].test))
        print(len(datasets[1].train))
        print(len(datasets[1].dev))
        print(len(datasets[1].test))
        # print(len(self.test))
        self.train_split = datasets[0].train_split
        self.dev_split = datasets[0].dev_split
        self.test_split = datasets[0].test_split

        self.mode = mode

        entrez_ids = [
            np.concatenate(
                (dataset.train_split, dataset.dev_split, dataset.test_split), axis=0
            )
            for dataset in datasets
        ]
        if make_new_datasplits:
            if len(datasets) > 2:
                raise ValueError("Only concatenation of two datasets are supported for new splits")
            np.random.seed(42)
            dataset_0_unique = np.setdiff1d(entrez_ids[0], entrez_ids[1])
            dataset_1_unique = np.setdiff1d(entrez_ids[1], entrez_ids[0])
            dataset_intersection = np.intersect1d(entrez_ids[0], entrez_ids[1])
            ds_0_split = np.split(np.random.permutation(dataset_0_unique), [int(0.7 * len(dataset_0_unique)), int(0.8 * len(dataset_0_unique))])
            ds_1_split = np.split(np.random.permutation(dataset_1_unique), [int(0.7 * len(dataset_1_unique)), int(0.8 * len(dataset_1_unique))])
            ds_intersection_split = np.split(np.random.permutation(dataset_intersection), [int(0.7 * len(dataset_intersection)), int(0.8 * len(dataset_intersection))])
            self.train_split = np.concatenate((ds_0_split[0], ds_1_split[0], ds_intersection_split[0]))
            self.dev_split = np.concatenate((ds_0_split[1], ds_1_split[1], ds_intersection_split[1]))
            self.test_split = np.concatenate((ds_0_split[2], ds_1_split[2], ds_intersection_split[2]))

            # Logging
            print("Dataset 0 unique:", len(dataset_0_unique))
            print("  - splits:", len(ds_0_split[0]), len(ds_0_split[1]), len(ds_0_split[2]))
            print("Dataset 1 unique:", len(dataset_1_unique))
            print("  - splits:", len(ds_1_split[0]), len(ds_1_split[1]), len(ds_1_split[2]))
            print("Dataset intersection:", len(dataset_intersection))
            print("  - splits:", len(ds_intersection_split[0]), len(ds_intersection_split[1]), len(ds_intersection_split[2]))
        else:
            # For all other datasets than CiVIC, add Entrez IDs to train split if they are
            # not already in the train/dev split
            for index, entrez_ids_list in enumerate(entrez_ids):
                if index != 0:
                    for entrez_id in entrez_ids_list:
                        if (
                            entrez_id not in self.train_split
                            and entrez_id not in self.dev_split
                            and entrez_id not in self.test_split
                        ):
                            self.train_split = np.append(self.train_split, [entrez_id])

        # Filter dataset
        self.train = self.dataset.filter(lambda x: x["entrez_id"] in self.train_split)
        self.dev = self.dataset.filter(lambda x: x["entrez_id"] in self.dev_split)
        self.test = self.dataset.filter(lambda x: x["entrez_id"] in self.test_split)

        print(self.train)
        print(self.dev)
        print(self.test)
        # exit()

    def select_columns(self, dataset: Dataset, mode: str = "raw_full_text") -> Dataset:
        cols_to_remove = dataset.column_names
        cols_to_keep = [
            "citation_id",
            "gene",
            "variant",
            "drugs",
            "entrez_id",
            "source_type",
            "disease",
            f"evidence_{mode}",
            "gene_synonyms",
            "gene_full_name",
            "variant_synonyms",
            "drugs_synonyms",
            f"gene_in_{mode}",
            f"variant_in_{mode}",
            f"drugs_in_{mode}",
            f"gene_synonyms_in_{mode}",
            f"variant_synonyms_in_{mode}",
            f"drugs_synonyms_in_{mode}",
        ]
        for col in cols_to_keep:
            cols_to_remove.remove(col)
        return dataset.remove_columns(cols_to_remove)

    def create_dataset(self) -> Dataset:
        return super().create_dataset()


if __name__ == "__main__":
    from po_datasets.civic import CiVICExamples
    from po_datasets.onco_kb import OncoKBExamples

    mode = "raw_text"

    civic = CiVICExamples(mode=mode)
    onco_kb = OncoKBExamples(mode=mode)

    # example = ConcatExamples([civic, onco_kb], mode=mode)
    example = ConcatExamples([civic, onco_kb], mode=mode, make_new_datasplits=True)

    print(example.dataset[0])
    print(example.detailed_dataset_stats())
