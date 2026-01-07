import argparse
import datetime
import errno
import json
import os

# import faiss.contrib.torch_utils  # use if you want to use PyTorch tensors
import time
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

from models.transformers import CustomSentenceTransformer
from utils.utils import process_pubmed_xmls


def chunker(lst, size, start=0):
    for i in range(start * size, len(lst), size):
        yield lst[i : i + size]


def compute_embeddings(
    abstracts_text: list[str],
    bi_encoder: CustomSentenceTransformer,
    chunking_size: int = 1000000,
    load_from_cache_file: bool = False,
    model_name: str = "",
    embeddings_path: str = "edel_repo_cache/treatment_explorer_embeddings/",
) -> npt.NDArray[np.float32]:
    # Find the largest chunk number in millions
    max_chunk = len(abstracts_text) // chunking_size
    last_processed_chunk = -1
    embeddings = np.empty((0, embedding_size), dtype="float32")
    if load_from_cache_file:
        print(
            "Loading embeddings from cache file in path ",
            embeddings_path,
            " with model name ",
            model_name,
        )
        # If no chunk was ret
        # Identify the last chunk that was computed and load it
        # If _best is in model name, ensure that _best is in the file name
        last_processed_chunk = max(
            [
                int(str(x).split("_")[-1].split(".")[0])
                for x in Path(embeddings_path).glob(model_name + "_*.npy")
            ]
            + [-1]
        )

        print("Last processed chunk: ", last_processed_chunk)
        print("Max chunk: ", max_chunk)
        if last_processed_chunk > -1:
            embeddings = np.load(
                embeddings_path + model_name + f"_{last_processed_chunk}.npy",
            )
            # Delete the last two million embeddings for debugging purposes
            # embeddings = embeddings[:-2000000]
            # TODO: Delete this part of the code
    if not load_from_cache_file or last_processed_chunk < max_chunk:
        print("Computing embeddings")
        for i, chunk in enumerate(
            chunker(abstracts_text, chunking_size, last_processed_chunk + 1)
        ):
            current_chunk = i + last_processed_chunk + 1
            print(
                f"Processing {current_chunk}-th million chunk. Current Time {datetime.datetime.now()}"
            )
            doc_embeddings = bi_encoder.encode(
                chunk, convert_to_tensor=True, show_progress_bar=False
            )
            doc_embeddings = doc_embeddings.cpu().numpy().astype("float32")
            embeddings = np.append(embeddings, doc_embeddings, axis=0)
            np.save(
                embeddings_path + model_name + f"_{current_chunk}.npy",
                embeddings,
            )
            # Delete the previous file chunk to save memory
            if current_chunk > 0:
                Path(embeddings_path + model_name + f"_{current_chunk-1}.npy").unlink()

    return embeddings


def compute_in_chunks(
    abstracts_text: list[str],
    shard_index: int,
    start_chunk: int,
    end_chunk: int,
    bi_encoder: CustomSentenceTransformer,
    chunking_size: int = 500000,
    load_from_cache_file: bool = False,
    model_name: str = "",
    embeddings_path: str = "edel_repo_cache/treatment_explorer_embeddings/",
) -> npt.NDArray[np.float32]:
    for i in range(start_chunk, end_chunk):
        # Check if the embeddings for the chunk have already been computed
        # Check for the existence of the file
        if not Path(
            embeddings_path + model_name + f"_shard_{shard_index}_chunk_{i}.npy"
        ).exists():
            compute_embedding_chunk(
                shard_index,
                i,
                abstracts_text,
                bi_encoder,
                chunking_size,
                model_name,
                embeddings_path,
            )
        else:
            print(
                f"Embeddings for shard {shard_index} and chunk {i} already exist. Skipping"
            )


def compute_embedding_chunk(
    shard_index: int,
    chunk_index: int,
    abstracts_text: list[str],
    bi_encoder: CustomSentenceTransformer,
    chunking_size: int = 500000,
    model_name: str = "",
    embeddings_path: str = "edel_repo_cache/treatment_explorer_embeddings/",
) -> npt.NDArray[np.float32]:
    # Find the largest chunk number in millions
    current_chunk = abstracts_text[
        chunk_index
        * chunking_size : min((chunk_index + 1) * chunking_size, len(abstracts_text))
    ]
    print(
        f"Processing {chunk_index}-th chunk of shard index. Current Time {datetime.datetime.now()}"
    )
    doc_embeddings = bi_encoder.encode(
        current_chunk, convert_to_tensor=True, show_progress_bar=False
    )
    doc_embeddings = doc_embeddings.cpu().numpy().astype("float32")
    np.save(
        embeddings_path + model_name + f"_shard_{shard_index}_chunk_{chunk_index}.npy",
        doc_embeddings,
    )

    return doc_embeddings


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_index",
        type=int,
        default=0,
        help="Index of the model to use for the embeddings",
    )
    parser.add_argument(
        "--flat_l2_index",
        action="store_true",
    )
    parser.add_argument(
        "--faiss_index",
        action="store_true",
    )
    parser.add_argument(
        "--load_embeddings_from_cache",
        action="store_true",
    )
    parser.add_argument(
        "--max_documents",
        type=int,
        default=1000000000,
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="L2 normalize embeddings before indexing",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        help="Pooling strategy for sentence transformer",
    )
    parser.add_argument(
        "--skip_no_abstracts",
        action="store_true",
        help="Skip PMIDs with no abstracts",
    )
    parser.add_argument(
        "--chunking_size",
        type=int,
        default=1000000,
        help="Chunk size for computing embeddings",
    )
    parser.add_argument(
        "--compute_chunks",
        action="store_true",
        help="Compute embeddings in chunks",
    )
    parser.add_argument(
        "--load_chunks",
        action="store_true",
        help="All chunks have already been pre-computed. Load them",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=0,
        help="Number of shards for the dataset",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index for the dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()

    faiss_mappings = [
        (
            # model_index = 0
            "edel_repo_cache/civic_onco_kb/pubmed_retriever_civic_onco_kb_margin_classes_v1_model.pt",
            "edel_repo_cache/civic_onco_kb/",
            "pubmed_retriever_civic_onco_kb_margin_classes_v1",
            768,
        ),
        (
            # model_index = 1
            "edel_repo_cache/uniprot/pubmed_retriever_uniprot_margin_classes_v1_model.pt",
            "edel_repo_cache/uniprot/",
            "pubmed_retriever_uniprot_margin_classes_v1",
            768,
        ),
        (
            # model_index = 2
            "edel_repo_cache/medcpt_all_pos_checkpoints_uniprot/doc_encoder",
            "edel_repo_cache/uniprot/",
            "medcpt_all_pos_finetuned_uniprot_dot_product",
            768,
        ),
    ]
    current_model_index = args.model_index
    pytorch_model = faiss_mappings[current_model_index][0]
    if "asym" in pytorch_model:
        modules_list = list(
            CustomSentenceTransformer(
                faiss_mappings[current_model_index][0],
                pooling=args.pooling,
            ).modules()
        )
        word_embedding_model = modules_list[0]._modules["0"].sub_modules["doc"][0]
        pooling_model = modules_list[0]._modules["0"].sub_modules["doc"][1]
        bi_encoder = CustomSentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
    else:
        bi_encoder = CustomSentenceTransformer(
            faiss_mappings[current_model_index][0],
            pooling=args.pooling,
        )
    pubmed_embeddings = Path("edel_repo_cache/pubmed_embeddings.npy")

    dataset_file = "edel_repo_cache/datasets/pubmed.dataset"
    if Path(dataset_file).exists():
        print("Loading dataset from local cache")
        dataset = load_from_disk(dataset_file)
        print("Dataset loaded")
    else:
        shards = [
            f"edel_repo_cache/pubmed/pubmed23n{i:04d}.xml"
            for i in range(1, 1167)
        ]
        dataset = Dataset.from_generator(
            process_pubmed_xmls,
            gen_kwargs={"shards": shards},
            num_proc=16,
        )
        dataset.save_to_disk(dataset_file)

    # Filter dataset for debugging purposes
    # debug_pmids = [20651371, 9330333, 9851515, 10840740, 10840739]
    # dataset = dataset.filter(lambda x: int(x["pmid"]) in debug_pmids)

    # abstracts_text = [x["title"] + ". " + x["abstract"] for x in dataset]
    if args.skip_no_abstracts:
        dataset = dataset.filter(
            lambda x: x["abstract"] is not None and len(x["abstract"]) > 0
        )
        print("Filtered number of PMIDs with abstracts:", len(dataset))
    if args.debug:
        # Just get the first one abstract for debugging purposes
        dataset = dataset.select(range(2))
        print("Sharded dataset length of debugging dataset", len(dataset))
    elif args.compute_chunks:
        dataset = dataset.shard(args.num_shards, args.shard_index)
        shard_chunks = (len(dataset) // args.chunking_size) + 1
        print(f"Shard number {args.shard_index} Sharded dataset length {len(dataset)}")
    elif args.load_chunks:
        # Dataset is sharded via mod args.num_shards
        # To preserve abstracts_id order, we need to concatenate the dataset according to the shards
        print("Concatenating shards...")
        dataset = concatenate_datasets(
            [dataset.shard(args.num_shards, i) for i in range(args.num_shards)]
        )

    if args.compute_chunks or args.faiss_index or args.flat_l2_index:
        print("Loading abstracts...")
        abstracts_text, abstracts_id = zip(
            *[
                (x["title"] + "[SEP]" + x["abstract"], int(x["pmid"]))
                for x in tqdm(dataset)
            ]
        )

    embedding_size = faiss_mappings[current_model_index][3]  # Size of embeddings

    if not args.compute_chunks and not args.load_chunks:
        # Compute embeddings using one cached file only
        corpus_embeddings = compute_embeddings(
            abstracts_text,
            bi_encoder,
            chunking_size=args.chunking_size,
            load_from_cache_file=args.load_embeddings_from_cache,
            model_name=faiss_mappings[current_model_index][2],
        )
        if args.debug:
            print("Corpus embeddings shape", corpus_embeddings.shape)
            print(
                "First ten dimensions of the first embedding", corpus_embeddings[0][:10]
            )
            print("Norm of the first embedding", np.linalg.norm(corpus_embeddings[0]))
            print(
                "First ten dimensions of the last embedding", corpus_embeddings[-1][:10]
            )
            print("Norm of the last embedding", np.linalg.norm(corpus_embeddings[-1]))
            print("First abstract ID and text:", abstracts_id[0], abstracts_text[0])
            print("Last abstract ID and text:", abstracts_id[-1], abstracts_text[-1])
    elif args.compute_chunks:
        # Compute embeddings in chunks and save them to disk
        compute_in_chunks(
            abstracts_text,
            args.shard_index,
            0,
            shard_chunks,
            bi_encoder,
            chunking_size=args.chunking_size,
            model_name=faiss_mappings[current_model_index][2],
        )
        print("Finished computing chunks...")
    elif args.load_chunks:
        model_name = faiss_mappings[current_model_index][2]
        corpus_embeddings = np.empty((0, embedding_size), dtype="float32")
        dataset_0 = dataset.shard(args.num_shards, 0)
        shard_chunks = (len(dataset_0) // args.chunking_size) + 1
        missing_chunks = ""
        for shard in range(args.num_shards):
            for chunk in range(shard_chunks):
                print(f"Loading chunk {chunk} of shard {shard}")
                file_name = f"edel_repo_cache/treatment_explorer_embeddings/{model_name}_shard_{shard}_chunk_{chunk}.npy"
                if Path(file_name).exists():
                    chunk_embeddings = np.load(file_name)
                    corpus_embeddings = np.append(
                        corpus_embeddings, chunk_embeddings, axis=0
                    )
                else:
                    print(f"Chunk {chunk} of shard {shard} is missing")
                    missing_chunks = file_name

                if missing_chunks:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), file_name
                    )
    # corpus_embeddings = faiss.rand((30000000, 768), 42)
    # corpus_embeddings = faiss.rand((1000000, 768), 42)

    # FAISS index
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_quora_faiss.py

    # Either use a FlatL2 index or an IVF index
    if args.flat_l2_index:
        # print("Using FlatL2 index")
        # index = faiss.IndexFlatL2(embedding_size)
        # Now, instead use IP (Inner Product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        print("Using FlatIP index")
        index = faiss.IndexFlatIP(embedding_size)

        # First, we need to normalize vectors to unit length
        if args.normalize_embeddings:
            print("Normalizing embeddings with L2 norm")
            faiss.normalize_L2(corpus_embeddings)
            if args.debug:
                print("Corpus embeddings shape", corpus_embeddings.shape)
                print(
                    "First ten dimensions of the first embedding after normalizing",
                    corpus_embeddings[0][:10],
                )
                print(
                    "Norm of the first embedding after normalizing",
                    np.linalg.norm(corpus_embeddings[0]),
                )
                print(
                    "First ten dimensions of the last embedding after normalizing",
                    corpus_embeddings[-1][:10],
                )
                print(
                    "Norm of the last embedding after normalizing",
                    np.linalg.norm(corpus_embeddings[-1]),
                )
        index.add(corpus_embeddings)

        # Map the abstracts_id to the index
        abstract_mapping = {}
        for idx, abstract_id in enumerate(abstracts_id):
            abstract_mapping[abstract_id] = idx
        # Save the mapping as simple json file
        with open(
            faiss_mappings[current_model_index][1]
            + faiss_mappings[current_model_index][2]
            + "_mapping.json",
            "w",
            encoding="utf-8",
        ) as fOut:
            json.dump(abstract_mapping, fOut)

        # Save the FAISS index
        faiss.write_index(
            index,
            faiss_mappings[current_model_index][1]
            + faiss_mappings[current_model_index][2]
            + "_flat.faiss",
        )

    if args.faiss_index:
        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        n_clusters = 32
        top_k_hits = 3

        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        # https://github.com/facebookresearch/faiss/issues/2361
        # https://github.com/facebookresearch/faiss/blob/main/benchs/bench_gpu_sift1m.py
        # index = faiss.index_factory(embedding_size, "OPQ32_128,IVF262144_HNSW32,PQ32x4fsr")
        index = faiss.index_factory(
            embedding_size, "OPQ32_128,IVF262144,PQ32", faiss.METRIC_INNER_PRODUCT
        )
        # index = faiss.index_factory(embedding_size, "OPQ32_128,IVF512,PQ32")
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True

        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        # Create the FAISS index
        print("Start creating FAISS index")
        # First, we need to normalize vectors to unit length
        if args.normalize_embeddings:
            print("Normalizing embeddings with L2 norm")
            faiss.normalize_L2(corpus_embeddings)
        print(corpus_embeddings.shape)

        start_time = time.time()
        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)
        end_time = time.time()
        print("Training finished (after {:.3f} seconds):".format(end_time - start_time))
        # Finally we add all embeddings to the index
        index.add_with_ids(corpus_embeddings, np.array(abstracts_id))
        # index.add(corpus_embeddings)

        index = faiss.index_gpu_to_cpu(index)

        # Save the FAISS index
        faiss.write_index(
            index,
            faiss_mappings[current_model_index][1]
            + faiss_mappings[current_model_index][2]
            + ".faiss",
        )
