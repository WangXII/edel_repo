import json

import pandas as pd
from datasets import Dataset

from po_datasets.dataset import DatasetExamples


class OncoKBExamples(DatasetExamples):
    def __init__(
        self,
        file: str = "20230607_OncoKB_References.json",
        mode="raw_full_text",
        filter_pubmed: bool = True,
        bool_group_by_citation_id: bool = False,
        bool_group_by_alteration: bool = False,
        cache_dir_prefix: str = "edel_repo_cache/onco_kb/examples_",
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

    def explode_and_normalize_column(
        self, df: pd.DataFrame, column: str, normalize: bool = True
    ):
        # Explode the column
        df = df.explode(column)

        if normalize:
            # Normalize the column
            tmp_df = pd.json_normalize(df[column])

            # Prepend the original column name to the flattened column names
            tmp_df.columns = [f"{column}.{col}" for col in tmp_df.columns]

            # Drop original columns from df and merge with flattened data
            df = df.drop(columns=[column]).reset_index(drop=True)
            df = pd.concat([df, tmp_df], axis=1)

        else:
            # Just reset the index
            df = df.reset_index(drop=True)

        return df

    def extract_columns(self, df: pd.DataFrame, column_str: str):
        """Extract the relevant data from the given column_str"""
        if column_str in [
            "treatments",
            "diagnosticImplications",
            "prognosticImplications",
        ]:
            df = self.explode_and_normalize_column(df, column_str)
            df[f"{column_str}.pmids_list"] = df[f"{column_str}.pmids"]
        # Skip mutationEffect for now as it has no concrete alteration information
        # elif column_str == "mutationEffect":

        df = self.explode_and_normalize_column(df, f"{column_str}.pmids", False)

        # Change treatment PMIDs to integers and change dtype to object
        df[f"{column_str}.pmids"] = (
            df[f"{column_str}.pmids"]
            .apply(lambda x: int(x) if not pd.isnull(x) else pd.NA)
            .convert_dtypes()
        )

        # Duplicate remaining column names to match CiVIC column names
        df["citation_id"] = df[f"{column_str}.pmids"]
        df["gene"] = df["query.hugoSymbol"]
        df["variant"] = df["query.alteration"]
        df["entrez_id"] = df["query.entrezGeneId"]
        df["source_type"] = "PubMed"
        if column_str == "treatments":
            # Remove entries with empty treatment.drugs
            df = df[df["treatments.drugs"].notna()]
            # print(df["treatments.drugs"])
            df["drugs"] = df["treatments.drugs"].apply(
                lambda x: [dictionary["drugName"] for dictionary in x]
            )
            df["drugs_ncit_id"] = df["treatments.drugs"].apply(
                lambda x: [dictionary["ncitCode"] for dictionary in x]
            )
            df["disease"] = df["treatments.levelAssociatedCancerType.mainType.name"]
        else:  # column_str in ["diagnosticImplications", "prognosticImplications"]:
            # Empty lists for drugs and drugs_ncit_id
            df["drugs"] = [[] for _ in range(len(df))]
            df["drugs_ncit_id"] = [[] for _ in range(len(df))]
            df["disease"] = df[f"{column_str}.tumorType.mainType.name"]

        return df

    def create_dataset(self) -> Dataset:
        # Load the JSON data
        with open(self.file, "r") as f:
            data = json.load(f)

        # Convert the data to a Pandas DataFrame
        df = pd.json_normalize(data)

        df.to_csv("onco_kb_original.csv", index=False)

        # TODO: Check overlap between treatment PMIDs,
        # mutationEffect.citations.pmids, diagnosticImplications.pmids
        # and prognosticImplications.pmids

        # TODO: Ignore mutationEffect for now as it has no concrete alteration
        # information
        # Split df across treatment, mutationEffect, diagnosticImplications and
        # prognosticImplications and concatentate them back together
        # Add a column to indicate the source of the data
        df = pd.concat(
            [
                self.extract_columns(df, column_str)
                for column_str in [
                    "treatments",
                    "diagnosticImplications",
                    "prognosticImplications",
                ]
            ]
        )

        df.to_csv("onco_kb_tmp.csv", index=False)

        # Drop nested columns
        df = df.drop(
            columns=[
                "treatments",
                "treatments.drugs",
                "treatments.abstracts",
                "treatments.levelExcludedCancerTypes",
                "mutationEffect.citations.abstracts",
                "diagnosticImplications",
                "diagnosticImplications.abstracts",
                "prognosticImplications",
                "prognosticImplications.abstracts",
            ]
        )

        # Skip the entries with no pmids for now
        # TODO: Include those ASCOPUB entries
        df = df[df["citation_id"].notna()]

        # Filter valid Entrez IDs
        df = df[df["entrez_id"] > 0]

        # Print the DataFrame
        print(df)
        print(len(df))

        df.to_csv("onco_kb.csv", index=False)

        dataset = Dataset.from_pandas(df)

        return dataset


if __name__ == "__main__":
    # https://huggingface.co/docs/datasets/cache
    from datasets import disable_caching

    disable_caching()

    # Get some statistics about the datasets
    group_by_citation_id = False

    example = OncoKBExamples(
        bool_group_by_citation_id=group_by_citation_id,
        cache=True,
        mode="raw_text",
    )  # Used for the reader
    print(example.dataset[0])
    print(example.detailed_dataset_stats())

    # print(example.dataset[:3])
    # print(example.dataset["drugs_in_raw_full_text"])

    # for sample in example.dataset:
    #     print()
    #     print(f"Gene: {sample['gene']}")
    #     print(f"Gene in raw full text: {sample['gene_in_raw_full_text']}")
    #     print(f"Variant: {sample['variant']}")
    #     print(f"Variant in raw full text: {sample['variant_in_raw_full_text']}")
    #     print(f"Drugs: {sample['drugs']}")
    #     print(f"Drugs in raw full text: {sample['drugs_in_raw_full_text']}")
    #     print(f"PMIDs: {sample['citation_id']}")
    #     print(f"Text: {sample['evidence_raw_full_text']}")
    #     continue_input = input("Continue? (y/n)")
    #     if continue_input == "n":
    #         break
