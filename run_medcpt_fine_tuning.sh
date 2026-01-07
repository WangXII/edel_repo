# Clone MedCPT repo https://github.com/ncbi/MedCPT

checkpoint="edel_cache_repo/medcpt_${mode}_checkpoints_uniprot"
model_index=3
device=3
mode="all_pos"

# Create datasets
python -m po_datasets.civic
python -m po_datasets.uniprot_ptms
python -m po_datasets.transform_to_medcpt_data

CUDA_VISIBLE_DEVICES=$device python MedCPT/retriever/main.py \
    --bert_q_path "ncbi/MedCPT-Query-Encoder" \
    --bert_d_path "ncbi/MedCPT-Article-Encoder" \
    --output_dir "edel_cache_repo/medcpt_${mode}_checkpoints_uniprot" \
    --train_dataset "edel_cache_repo/medcpt_${mode}_examples/train_uniprot_train.jsonl" \
    --pmid2info_path "edel_cache_repo/medcpt_${mode}_examples/pmid2info_uniprot_train.json" \
    --qid2info_path "edel_cache_repo/medcpt_${mode}_examples/qid2info_uniprot_train.json" \
    --warmup_steps 128 \
    --num_train_epochs 8

mkdir -p $checkpoint/query_encoder
cp $checkpoint/special_tokens_map.json $checkpoint/query_encoder/special_tokens_map.json
cp $checkpoint/tokenizer_config.json $checkpoint/query_encoder/tokenizer_config.json
cp $checkpoint/tokenizer.json $checkpoint/query_encoder/tokenizer.json
cp $checkpoint/training_args.bin $checkpoint/query_encoder/training_args.bin
cp $checkpoint/vocab.txt $checkpoint/query_encoder/vocab.txt

mkdir -p $checkpoint/doc_encoder
cp $checkpoint/special_tokens_map.json $checkpoint/doc_encoder/special_tokens_map.json
cp $checkpoint/tokenizer_config.json $checkpoint/doc_encoder/tokenizer_config.json
cp $checkpoint/tokenizer.json $checkpoint/doc_encoder/tokenizer.json
cp $checkpoint/training_args.bin $checkpoint/doc_encoder/training_args.bin
cp $checkpoint/vocab.txt $checkpoint/doc_encoder/vocab.txt


# Indexing into FAISS can take some time. Save time by parallelizing and running index_pubmed.py on mutiple shards, e.g., --num_shards=8 and shard_index=0..7
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --load_embeddings_from_cache --compute_chunks --num_shards=1 --shard_index=0 --chunking_size=500000
CUDA_VISIBLE_DEVICES=$device python index_pubmed.py --model_index=$model_index --flat_l2_index --faiss_index --load_embeddings_from_cache --load_chunks --num_shards=1 --chunking_size=500000
CUDA_VISIBLE_DEVICES=$device python -m evaluation.evaluation_dataset --model_index=$model_index --results_file=20250112_retriever_results_medcpt_all_pos_tuned_ptm.log --data_set=uniprot --data_split=test --full_name_in_query --use_dot_product