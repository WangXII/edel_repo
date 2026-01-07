model_index=0
device=2
results_file=20250212_retriever_results_edel_v1.log

CUDA_VISIBLE_DEVICES=$device python train_retriever.py --num_epochs=8 --batch_size=64 --datasets=civic_onco_kb --cache_file_prefix=civic_onco_margin_classes_v1 --margin_config=margin_config.margin_classes_v1 --wandb_name=_models --check_seen_pmids

# Indexing into FAISS can take some time. Save time by parallelizing and running index_pubmed.py on mutiple shards, e.g., --num_shards=8 and shard_index=0..7
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --load_embeddings_from_cache --compute_chunks --num_shards=1 --shard_index=0 --chunking_size=500000
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --flat_l2_index --load_embeddings_from_cache --load_chunks --num_shards=8 --chunking_size=500000 --normalize_embeddings

CUDA_VISIBLE_DEVICES=$device python -m evaluation.evaluation_dataset --model_index=$model_index --results_file=$results_file --data_set=civic_oncokb --data_split=test
