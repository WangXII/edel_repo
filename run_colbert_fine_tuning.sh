# Create dataset
python -m po_datasets.create_colbert_dataset --dataset_name=civic_oncokb --dataset_split=train

# For all this part, you need to switch to a different python venv with ColBERT installed
# https://github.com/stanford-futuredata/ColBERT
# conda env create -f conda_env[_cpu].yml
# conda activate colbert

# Train ColBERT retriever
python -m train_colbert --dataset_name=civic_oncokb --dataset_split=train

# Example evaluation scripts
python -m po_datasets.create_colbert_dataset --dataset_name=civic_oncokb --dataset_split=test
python -m evaluation.evaluation_colbert --model_index=10 --data_set=civic_oncokb --data_split=test
