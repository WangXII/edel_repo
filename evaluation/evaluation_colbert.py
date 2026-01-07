import sys

sys.path.insert(0, "../ColBERT/")

from pathlib import Path

from colbert import Searcher
from colbert.infra import Run, RunConfig
from datasets import load_from_disk

from evaluation.evaluation_retriever import RetrieverEvaluator


class ColBERTRetrieverEvaluator(RetrieverEvaluator):
    def __init__(
        self,
        es_index="20231127_pubmed",
        num_gpu=3,
        index_dir="edel_repo_cache/colbert_2.0_index/",
        index_name="colbert_v2.pubmed.384len.2bits",
        top_k_hits=[10, 100],
    ):
        with Run().context(RunConfig(nranks=num_gpu, experiment=index_dir)):
            self.searcher = Searcher(
                index=index_name,
                checkpoint="edel_repo_cache/pretrained_llm/colbertv2.0",
            )

        dataset_file = "edel_repo_cache/datasets/pubmed.dataset"
        if Path(dataset_file).exists():
            print("Loading dataset from local cache")
            self.dataset = load_from_disk(dataset_file)

        super().__init__(
            es_index=es_index,
            top_k_hits=top_k_hits,
        )

    def get_faiss_results(self, query):
        # Find the top-k passages for this query
        results = self.searcher.search(query, k=max(self.top_k_hits))

        # Print out the top-k retrieved passages
        faiss_scores, faiss_pm_ids = [], []
        for passage_id, _, passage_score in zip(*results):
            faiss_scores.append(passage_score)
            faiss_pm_ids.append(int(self.dataset[passage_id]["pmid"]))

        return faiss_scores, faiss_pm_ids


if __name__ == "__main__":
    import argparse

    from datasets.utils import logging

    from evaluation.data import DRUG_KEYWORDS
    from evaluation.evaluation_dataset import DatasetEvaluator
    from evaluation.model_list import MODEL_LIST
    from po_datasets.civic import CiVICExamples
    from po_datasets.concat_dataset import ConcatExamples
    from po_datasets.onco_kb import OncoKBExamples
    from po_datasets.uniprot_ptms import UniProtPTMExamples
    from utils.utils import (
        get_dataset_dict,
        get_dataset_dict_from_csv,
        get_dataset_dict_uniprot,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="raw_text")
    parser.add_argument(
        "--use_dot_product",
        action="store_true",
        help="Use dot product instead of cosine for retrieval",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        default="uniprot",
        help="Data set to use for evaluation",
    )
    parser.add_argument(
        "--only_use_flat_index",
        action="store_true",
        help="Only use flat index for retrieval",
        default=True,
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        help="Pooling strategy for sentence transformer",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="dev",
        help="Data split to use for evaluation",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="retriever_results.log",
        help="File to write results to",
    )
    parser.add_argument(
        "--top_k_hits",
        type=list,
        default=[10, 50, 200, 1000],
        help="Number of hits to retrieve from FAISS",
    )
    parser.add_argument(
        "--synonyms_in_query",
        action="store_true",
        help="Use synonyms in query",
    )
    parser.add_argument(
        "--full_name_in_query",
        action="store_true",
        help="Use full name in query",
    )
    parser.add_argument(
        "--use_asym_bi_encoder",
        action="store_true",
        help="Use asymmetric bi-encoder",
    )
    args = parser.parse_args()

    model_index = MODEL_LIST[args.model_index]

    retriever = ColBERTRetrieverEvaluator(
        top_k_hits=args.top_k_hits,
        es_index="20231127_pubmed",
    )

    logging.enable_progress_bar()

    if args.data_set in ["civic_oncokb", "civic", "oncokb", "onkopedia"]:
        civic_dataset_samples = CiVICExamples(mode=args.mode)
        oncokb_dataset_samples = OncoKBExamples(mode=args.mode)
        samples_split = ConcatExamples(
            [
                civic_dataset_samples,
                oncokb_dataset_samples,
            ],
            mode=args.mode,
        )

        if args.data_set == "civic_oncokb":
            dataset_dict_train, _, _ = get_dataset_dict(
                samples_split, datasplit=args.data_split, split_examples=samples_split
            )
        elif args.data_set == "civic":
            dataset_dict_train, _, _ = get_dataset_dict(
                civic_dataset_samples, datasplit=args.data_split, split_examples=samples_split
            )
        elif args.data_set == "oncokb":
            dataset_dict_train, _, _ = get_dataset_dict(
                oncokb_dataset_samples, datasplit=args.data_split, split_examples=samples_split
            )
        elif args.data_set == "onkopedia":
            dataset_dict_train = get_dataset_dict_from_csv("onkopedia_guidelines.csv")

        evaluator = DatasetEvaluator(
            retriever,
            args.data_set,
            DRUG_KEYWORDS,
            dataset_dict_train,
            file_prefix="data/retriever_results/",
            results_file=args.results_file,
            bool_bi_encoder_results=False,
            synonyms_in_query=args.synonyms_in_query,
            full_name_in_query=False,
            data_split=args.data_split,
            compare_models=["faiss", "bm25"],
            model_name=model_index[0],
            model_abbreviation=model_index[4],
            top_k_hits=args.top_k_hits,
        )
    elif args.data_set == "uniprot":
        uniprot_samples = UniProtPTMExamples(mode=args.mode)

        dataset_dict_train, _ = get_dataset_dict_uniprot(
            uniprot_samples, datasplit=args.data_split
        )

        evaluator = DatasetEvaluator(
            retriever,
            args.data_set,
            DRUG_KEYWORDS,
            dataset_dict_train,
            file_prefix="data/retriever_results/",
            results_file=args.results_file,
            bool_bi_encoder_results=False,
            synonyms_in_query=args.synonyms_in_query,
            full_name_in_query=args.full_name_in_query,
            data_split=args.data_split,
            compare_models=["faiss", "bm25"],
            model_name=model_index[0],
            model_abbreviation=model_index[4],
            top_k_hits=args.top_k_hits,
        )
