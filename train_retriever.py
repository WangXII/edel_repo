import argparse
import logging
import os

import datasets.utils
import torch
from sentence_transformers import models
from torch.utils.data import DataLoader

import models.margin_config as margin_config
import models.margin_config_uniprot as margin_config_uniprot
import wandb
from models.binary_classification_evaluator import (
    BinaryClassificationEvaluator,
    SequentialEvaluator,
)
from models.civic_oncokb_retriever import CiVICOncoKBRetriever
from models.transformers import (
    CachedMultipleNegativesRankingLoss,
    CustomSentenceTransformer,
    MarginContrastiveLoss,
)
from models.uniprot_retriever import UniprotRetriever
from po_datasets.civic import CiVICExamples
from po_datasets.concat_dataset import ConcatExamples
from po_datasets.dataset import DatasetExamples
from po_datasets.drugbank import DrugBankExamples
from po_datasets.onco_kb import OncoKBExamples
from po_datasets.uniprot_ptms import UniProtPTMExamples
from utils.utils import transform_input_examples_asym

datasets.utils.logging.disable_progress_bar()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_eval_scores(score, epoch, steps, learning_rate):
    # Score is a list of tuple scores, one for each evaluator
    # The tuple consists of (average_precision_score, f1_score, f1_threshold)
    print(f"Epoch: {epoch} | Steps: {steps}")
    print(f"Average Precision Scores [train|dev]: {score[0][0]}{score[1][0]}")
    print(f"F1 Scores [train|dev]: {score[0][1]}{score[1][1]}")
    print(f"F1 Score Thresholds [train|dev]: {score[0][2]}{score[1][2]}")
    print(f"Minimum distance scores [train|dev]: {score[0][3]}{score[1][3]}")
    print(f"Minimum distance thresholds [train|dev]: {score[0][4]}{score[1][4]}")
    print(f"Mean loss [train|dev]: {score[0][5]}{score[1][5]}")
    print(f"Current learning rate: {learning_rate}")
    wandb.log(
        {
            "train_ap": score[0][0],
            "train_f1": score[0][1],
            "train_f1_threshold": score[0][2],
            "dev_ap": score[1][0],
            "dev_f1": score[1][1],
            "dev_f1_threshold": score[1][2],
            "train_min_distance": score[0][3],
            "train_max_distance": score[0][4],
            "dev_min_distance": score[1][3],
            "dev_max_distance": score[1][4],
            "train_mean_loss": score[0][5],
            "dev_mean_loss": score[1][5],
            "learning_rate": learning_rate,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datasets", type=str, default="civic_onco_kb")
    parser.add_argument("--datasets", type=str, default="uniprot")
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lambda_weight", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="raw_text")
    parser.add_argument("--all_negatives", action="store_true", default=True)
    parser.add_argument("--asym_models", action="store_true", default=False)
    parser.add_argument("--freeze_query_model", action="store_true", default=False)
    parser.add_argument(
        "--cache_file_prefix",
        type=str,
        default="uniprot_margin_classes_v17",
    )
    parser.add_argument(
        "--margin_config",
        type=str,
        default="margin_config_uniprot.margin_classes_uniprot_v17",
    )
    parser.add_argument(
        "--train_on_eval",
        action="store_true",
        default=False,
        help="Train on the smaller eval variation",
    )
    parser.add_argument("--split_train", type=int, default=1)
    parser.add_argument("--train_split_number", type=int, default=0)
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="_model")
    parser.add_argument("--batch_negatives", action="store_true", default=False)
    parser.add_argument("--random_negatives", action="store_true", default=False)
    parser.add_argument("--check_seen_pmids", action="store_true", default=False)
    args = parser.parse_args()

    # dataset = CiVICExamples()

    civic: DatasetExamples = CiVICExamples(mode=args.mode)
    # drugbank = DrugBankExamples(mode=args.mode)
    onco_kb: DatasetExamples = OncoKBExamples(mode=args.mode)
    uniprot: DatasetExamples = UniProtPTMExamples(mode=args.mode)

    if args.datasets == "civic":
        dataset = civic
    # elif args.datasets == "drugbank":
    #     dataset = drugbank
    elif args.datasets == "onco_kb":
        dataset = onco_kb
    elif args.datasets == "civic_onco_kb":
        dataset = ConcatExamples([civic, onco_kb], mode=args.mode, make_new_datasplits=True)
    elif args.datasets == "uniprot":
        dataset = uniprot
    # elif args.datasets == "all_po_datasets":
    #     dataset = ConcatExamples([civic, drugbank, onco_kb], mode=args.mode)

    print(f"Dataset length: {len(dataset.dataset)}")  # type: ignore
    print("Train | Dev | Test")
    print(f"{len(dataset.train)} | {len(dataset.dev)} | {len(dataset.test)}")  # type: ignore

    if not args.train_on_eval:
        # Load examples from cache (DO NOT create here, margin values do not match)
        if args.datasets == "civic_onco_kb":
            retriever = CiVICOncoKBRetriever(
                gene_synonyms=True,
                variant_synonyms=True,
                drugs_synonyms=True,
                check_for_seen_pmids=args.check_seen_pmids,
                examples=dataset,
                synonyms_in_query=False,
                cache=True,
                cache_file_prefix=args.cache_file_prefix,
                margin_config=eval(args.margin_config),
                use_batch_negatives=args.batch_negatives,
                use_random_negatives=args.random_negatives,
                max_ratio_negatives=20,
                max_negatives=50,
                all_negatives=args.all_negatives,
                use_supervised_examples=True,
                use_distant_bm_25_examples=False,
                bm25_k=5,
                bm25_repeat_seen_pmids=False,
            )
        else:
            retriever = UniprotRetriever(
                substrate_synonyms=True,
                catalysts_synonyms=True,
                full_name_in_query=True,
                filter_no_catalysts=True,
                examples=dataset,
                synonyms_in_query=False,
                cache=True,
                cache_file_prefix=args.cache_file_prefix,
                margin_config=eval(args.margin_config),
                use_batch_negatives=args.batch_negatives,
                use_random_negatives=args.random_negatives,
                max_ratio_negatives=20,
                max_negatives=50,
                use_supervised_examples=True,
                use_distant_bm_25_examples=False,
                bm25_k=5,
                bm25_repeat_seen_pmids=False,
                filter_seen_pmids=True,
                all_negatives=args.all_negatives,
            )
        train_examples = retriever.train
        if args.split_train > 1 and args.train_split_number < args.split_train:
            train_examples = train_examples[
                (args.train_split_number)
                * len(train_examples)
                // args.split_train : (args.train_split_number + 1)
                * len(train_examples)
                // args.split_train
            ]
        total_examples = len(train_examples) + len(retriever.dev) + len(retriever.test)

    # Evaluate on supervised examples only
    # Fix the same eval datasets for all model configurations
    if args.datasets == "civic_onco_kb":
        eval_dataset = CiVICOncoKBRetriever(
            examples=dataset,
            cache_file_prefix="civic_onco_kb_abstracts_margin_classes_v14_fix_splits",
            cache=True,
            margin_config=margin_config.margin_classes_v14,
            use_batch_negatives=False,
            use_random_negatives=False,
        )
        eval_examples = eval_dataset.test
    elif args.datasets == "uniprot":
        eval_dataset = UniprotRetriever(
            examples=dataset,
            cache_file_prefix="uniprot_margin_classes_v20",
            cache=True,
            margin_config=margin_config_uniprot.margin_classes_uniprot_v20,
            use_batch_negatives=False,
            use_random_negatives=False,
        )
        eval_examples = eval_dataset.dev

    if args.train_on_eval:
        train_examples = eval_dataset.train
        print(f"!! Train on eval examples instead: {len(train_examples)}")
    else:
        print(f"Total retriever examples: {total_examples}")
        print(f"Train retriever examples: {len(train_examples)}")
        print(f"Dev retriever examples: {len(retriever.dev)}")
        print(f"Test retriever examples: {len(retriever.test)}")
        print(train_examples[0])

    # ANN search using sentence-transformers (no training and no index yet)
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_wikipedia_qa.py

    print(args.asym_models)
    if args.asym_models:
        if args.pretrained_checkpoint == "medcpt":
            query_word_embedding_model = query_word_embedding_model = (
                models.Transformer("ncbi/MedCPT-Query-Encoder")
            )
        else:
            query_word_embedding_model = models.Transformer(
                "michiyasunaga/BioLinkBERT-base"
            )
        query_pooling_model = models.Pooling(
            query_word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="cls",
        )
        query_model = CustomSentenceTransformer(
            modules=[query_word_embedding_model, query_pooling_model]
        )
        if args.freeze_query_model:
            # Freeze the parameters of the query model
            for param in query_model.parameters():
                param.requires_grad = False

        if args.pretrained_checkpoint == "medcpt":
            doc_word_embedding_model = models.Transformer("ncbi/MedCPT-Article-Encoder")
        else:
            doc_word_embedding_model = models.Transformer(
                "michiyasunaga/BioLinkBERT-base"
            )
        doc_pooling_model = models.Pooling(
            doc_word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls"
        )
        doc_model = CustomSentenceTransformer(
            modules=[doc_word_embedding_model, doc_pooling_model]
        )
        asym_model = models.Asym(
            {"query": query_model, "doc": doc_model},
        )
        bi_encoder = CustomSentenceTransformer(modules=[asym_model])

        # Transforming input examples to asym format
        train_examples = transform_input_examples_asym(train_examples)
        eval_examples = transform_input_examples_asym(eval_examples)

    else:
        word_embedding_model = models.Transformer("michiyasunaga/BioLinkBERT-base")
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls"
        )
        bi_encoder = CustomSentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )

    # Training loop

    wandb.init(
        project="treatment-explorer",
        name=f"{args.cache_file_prefix}{args.wandb_name}",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lambda_weight": args.lambda_weight,
            "mode": args.mode,
            "cache_file_prefix": args.cache_file_prefix,
            "train_on_eval": args.train_on_eval,
            "asym_models": args.asym_models,
            "pretrained_checkpoint": args.pretrained_checkpoint,
            "split_train": args.split_train,
            "train_split_number": args.train_split_number,
        },
    )

    if args.batch_negatives:
        print("Using batch negatives")
        train_dataloader: DataLoader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=args.batch_size * 4,
        )
    else:
        train_dataloader: DataLoader = DataLoader(
            train_examples, shuffle=True, batch_size=args.batch_size  # type: ignore
        )
    # train_loss = losses.ContrastiveLoss(model=bi_encoder)
    if args.batch_negatives:
        train_loss = CachedMultipleNegativesRankingLoss(
            model=bi_encoder, mini_batch_size=args.batch_size
        )
    else:
        if args.asym_models:
            train_loss = MarginContrastiveLoss(
                model=bi_encoder, lambda_weight=args.lambda_weight, asym_dot_product=True
            )
        else:
            train_loss = MarginContrastiveLoss(
                model=bi_encoder, lambda_weight=args.lambda_weight
            )

    evaluator_train = BinaryClassificationEvaluator.from_input_examples(
        train_examples,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    evaluator_dev = BinaryClassificationEvaluator.from_input_examples(
        eval_examples,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    evaluator = SequentialEvaluator(
        [evaluator_train, evaluator_dev], main_score_function=lambda scores: scores
    )

    asym_suffix = "_asym" if args.asym_models else ""
    pretrained_suffix = (
        f"_{args.pretrained_checkpoint}" if args.pretrained_checkpoint else ""
    )
    if args.split_train == 1:
        train_split_suffix = ""
    else:
        train_split_suffix = (
            f"_split_{args.train_split_number}_of_{args.split_train - 1}"
        )

    scheduler = "WarmupCosine"
    total_training_steps = len(train_dataloader) * args.num_epochs
    warmup_ratio = 0.1
    warmup_steps = int(total_training_steps * warmup_ratio)
    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = "WarmupLinear"
    warmup_steps = 10000

    bi_encoder.fit(
        [(train_dataloader, train_loss)],
        epochs=args.num_epochs,
        show_progress_bar=True,
        evaluator=evaluator,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        output_path=f"edel_repo_cache/{args.datasets}/pubmed_retriever_{args.cache_file_prefix}{args.wandb_name}{pretrained_suffix}{asym_suffix}{train_split_suffix}_best.pt",
        callback=log_eval_scores,
        optimizer_params={"lr": args.learning_rate},
    )

    bi_encoder.save(
        f"edel_repo_cache/{args.datasets}/pubmed_retriever_{args.cache_file_prefix}{args.wandb_name}{pretrained_suffix}{asym_suffix}{train_split_suffix}.pt"
    )

    wandb.finish()
