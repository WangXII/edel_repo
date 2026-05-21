import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import ujson

sys.path.insert(0, "../ColBERT/")

from colbert import Trainer
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score
from colbert.training.lazy_batcher import LazyBatcher

np.random.seed(42)

def train(mode="all_pos", dataset_name="civic_oncokb", negs="batch", dataset_split="train", n_gpus=3, batch_size=48, lr=1e-5, warmup=25, max_steps=1000):
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    if negs == "batch":
        name = f'colbertv2.0.{dataset_name}.{dataset_split}.{mode}.bs{batch_size}.lr{lr}.warmup{warmup}.maxsteps{max_steps}'
    elif negs == "explicit":
        name = f'colbertv2.0.{dataset_name}.explicit_negs.{dataset_split}.{mode}.bs{batch_size}.lr{lr}.warmup{warmup}.maxsteps{max_steps}'
    else:
        raise ValueError(f"Invalid negs value: {negs}")
    input_dir = f'/vol/tmp/wangxida/treatment_explorer/colbert_datasets_{mode}_examples/'
    root_dir = '/vol/tmp/wangxida/treatment_explorer/colbert_models/'
    experiments = 'trained'
    with Run().context(RunConfig(nranks=n_gpus, root=root_dir, experiment=experiments, name=name)):
        if negs == "batch":
            triples = input_dir + f'{dataset_split}_{dataset_name}_examples.json'
            queries = input_dir + f'{dataset_name}_{dataset_split}_queries.tsv'
            collection = input_dir + f'{dataset_name}_{dataset_split}_collection.tsv'
        elif negs == "explicit":
            triples = input_dir + f'{dataset_split}_{dataset_name}_explicit_negs_examples.json'
            queries = input_dir + f'{dataset_name}_{dataset_split}_explicit_negs_queries.tsv'
            collection = input_dir + f'{dataset_name}_{dataset_split}_explicit_negs_collection.tsv'

        # doc_maxlen=384
        config = ColBERTConfig(
            bsize=batch_size, lr=lr, warmup=warmup, maxsteps=max_steps, save_every=100, doc_maxlen=512, dim=128,
            attend_to_mask_tokens=False, nway=2, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint='/vol/tmp/wangxida/pretrained_llm/colbertv2.0')  # or start from scratch, like `bert-base-uncased`


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    dataset_name,
    dataset_split,
    mode="all_pos",
    bsize=48,
    nway=2,
    doc_maxlen=512,
    query_maxlen=32,
):
    input_dir = f'/vol/tmp/wangxida/treatment_explorer/colbert_datasets_{mode}_examples/'
    config = ColBERTConfig(
        bsize=bsize,
        nway=nway,
        doc_maxlen=doc_maxlen,
        query_maxlen=query_maxlen,
        use_ib_negatives=False,
        checkpoint=checkpoint_path,
    )

    triples = input_dir + f'{dataset_split}_{dataset_name}_examples.json'
    queries = input_dir + f'{dataset_name}_{dataset_split}_queries.tsv'
    collection = input_dir + f'{dataset_name}_{dataset_split}_collection.tsv'

    model = Checkpoint(checkpoint_path, colbert_config=config)
    model = model.cuda()
    model.eval()

    reader = LazyBatcher(
        config=config,
        triples=triples,
        queries=queries,
        collection=collection,
        rank=0,
        nranks=1,
    )

    losses = []
    pos_scores = []
    neg_scores = []
    margins = []

    for batch_group in reader:
        # tensorize_triples returns a list of microbatches
        for batch in batch_group:
            if len(batch) == 3:
                queries_tensor, passages_tensor, target_scores = batch
            elif len(batch) == 2:
                queries_tensor, passages_tensor = batch
                target_scores = None
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            Q = model.query(*queries_tensor)

            D, D_mask = model.doc(
                *passages_tensor,
                keep_dims="return_mask"
            )

            Q = Q.repeat_interleave(nway, dim=0).contiguous()

            scores = colbert_score(Q, D, D_mask, config=config)
            scores = scores.view(-1, nway)

            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = torch.nn.functional.cross_entropy(scores, labels)

            pos = scores[:, 0]
            neg = scores[:, 1]
            margin = pos - neg

            losses.append(loss.item())
            pos_scores.extend(pos.detach().cpu().tolist())
            neg_scores.extend(neg.detach().cpu().tolist())
            margins.extend(margin.detach().cpu().tolist())

    return {
        "checkpoint": checkpoint_path,
        "loss": float(np.mean(losses)),
        "pos_avg": float(np.mean(pos_scores)),
        "neg_avg": float(np.mean(neg_scores)),
        "margin_avg": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "accuracy": float(np.mean(np.array(margins) > 0)),
    }


if __name__ == '__main__':
    mode = "strict_pos"
    dataset_name = "uniprot"
    negs = "explicit"
    dataset_split = "train"
    # 456 rel_pairs for strict_pos, 3385 for all_pos in civic
    # For 8 repeats, we end up at 2665 and 25774 examples, respectively, divided by the batch size, the number of steps are 55 and 536
    # 114 rel_pairs for strict_pos, 3208 for all_pos in uniprot
    # For 8 repeats, we end up at 908 and 25659 examples, respectively, divided by the batch size, the number of steps are 19 and 534
    n_gpus = 3
    batch_size = 16 * n_gpus
    lr = 1e-5
    warmup = 30
    max_steps = 600
    # For civic_oncokb and all_pos, we set warmup_steps to 30 and max_steps to 600
    # For civic_oncokb and strict_pos, we set warmup_steps to 6 and max_steps to 60
    # For uniprot and strict_pos, we set warmup_steps to 2 and max_steps to 20
    # For uniprot and all_pos, we set warmup_steps to 30 and max_steps to 600

    num_repeats = 8
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="civic_oncokb")
    parser.add_argument("--dataset_split", type=str, default="train")

    args = parser.parse_args()

    train(mode=mode, dataset_name=args.dataset_name, negs=negs, dataset_split=args.dataset_split, n_gpus=n_gpus, batch_size=batch_size, lr=lr, warmup=warmup, max_steps=max_steps)

    # ckpt_path = "/vol/tmp/wangxida/pretrained_llm/colbertv2.0"
    negs_suffix = ".explicit_negs" if negs == "explicit" else ""
    ckpt_path = f'/vol/tmp/wangxida/treatment_explorer/colbert_models/trained/none/colbertv2.0.{dataset_name}{negs_suffix}.train.{mode}.bs{batch_size}.lr{lr}.warmup{warmup}.maxsteps{max_steps}/checkpoints/colbert'
    eval_results = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        dataset_name=dataset_name,
        dataset_split="dev",
        bsize=8,
    )
    print(eval_results)
