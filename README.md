# EDEL: Enhancing Denser Retrievers for Cuation of Biomedical Knowledge Bases
## Prepare Environment
1. Install required packages using pip (requirements.txt).
2. Create a new folder for the data `mkdir edel_repo_cache`.
2. Install Lucene/ElasticSearch for document retrieval using the BM25 baseline using the version available under \url{https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.0-linux-x86_64.tar.gz} into `./edel_repo_cache`.

## Get the Datasets
3. Download and process the document corpus
    - Create new folders `mkdir edel_repo_cache/pubmed` and `mkdir edel_repo_cache/datasets`.
    - Download the PubMed Baseline via its website under \url{https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/} into `./edel_repo_cache/pubmed`.
    - Process the PubMed files for faster access via `python -m utils.util` (which saves them as a arrow tables)
    - Start Elasticsearch via .`/elasticsearch-8.11.1/bin/elasticsearch -E "discovery.type=single-node"`.
    - Build and index the Pubmed corpus using elasticsearch via `python -m index_elasticsearch`.
4. Download the training data:
    - Download the CIViC monthly dump from \url{https://civicdb.org/downloads/01-Nov-2022/01-Nov-2022-ClinicalEvidenceSummaries.tsv} into `./edel_repo_cache`.
    - Download OncoKB data, available upon request using the API \url{https://www.oncokb.org/api-access}. Process using `python -m extract_oncokb` (and filter for entries before 20230607 for reproducing the experiments).
    - Download UniProtKB data (SwissProt, text) available under \url{https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz} into `./edel_repo_cache`.
    - For synonym mappings, download NCITheSaurus \url{https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/2022/22.11d_Release/Thesaurus.FLAT.zip} and EntrezGene \url{https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz} and unpack the files into `./edel_repo_cache`.
        - Run `python -m utils.build_ncbi_gene_id_db.py` to cache the dataset for faster access.
    
## Process the training data; train and evaluate the models
5. Run run_edel_po.sh for processing the Precision Oncology datasets; then conduct training and evaluating the retrieval model on them. Instead of training, pre-trained model checkpoint is available under \url{https://huggingface.co/xdawang/edel_po}.
6. Run run_edel_ptm.sh for processing the Post-Translational Modification Dataset dataset; then conduct training and evaluating the retrieval model on them. Instead of training, pre-trained model checkpoint is available under \url{https://huggingface.co/xdawang/edel_ptm}.
