device=2
model_index=1
results_file=20250223_retriever_results_edel_uniprot_v1.log

CUDA_VISIBLE_DEVICES=$device python train_retriever.py --num_epochs=8 --batch_size=64 --datasets=uniprot --cache_file_prefix=uniprot_margin_classes_v1 --margin_config=margin_config_uniprot.margin_classes_uniprot_v1 --wandb_name=_model

# Indexing into FAISS can take some time. Save time by parallelizing and running index_pubmed.py on mutiple shards, e.g., --num_shards=8 and shard_index=0..7
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --load_embeddings_from_cache --compute_chunks --num_shards=1 --shard_index=0 --chunking_size=500000
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --flat_l2_index --load_embeddings_from_cache --load_chunks --num_shards=8 --chunking_size=500000 --normalize_embeddings

CUDA_VISIBLE_DEVICES=$device python -m evaluation.evaluation_dataset --model_index=$model_index --results_file=$results_file --data_set=uniprot --data_split=test --full_name_in_query 