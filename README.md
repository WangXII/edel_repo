# EDEL: Enhancing Denser Retrievers for Cuation of Biomedical Knowledge Bases
## General Setup
1. Install required packages using pip (requirements.txt)
2. Install Lucene/ElasticSearch for document retrieval (using the BM25 baseline) \url{https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.0-linux-x86_64.tar.gz} into `./edel_repo_cache`
3. Download Datasets
    - Download PubMed Baseline via its website under \url{https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/} into `./edel_repo_cache`. (Filter documents before 2024 for reproducing the experiments.)
    - Download CIViC monthly dump from \url{https://civicdb.org/downloads/01-Nov-2022/01-Nov-2022-ClinicalEvidenceSummaries.tsv} into this folder
    - Download OncoKB data, available upon request using the API \url{https://www.oncokb.org/api-access}. Process using `python -m extract_oncokb` (and filter for entries before 20230607 for reproducing the experiments).
    - Download UniProtKB data (SwissProt, text) is available under \url{https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz} into `./edel_repo_cache`
    - Download NCITheSaurus \url{https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/2022/22.11d_Release/Thesaurus.FLAT.zip} and EntrezGene \url{https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz} for synonym mappings and unpack files into `./edel_repo_cache`. Run `python -m utils.build_ncbi_gene_id_db.py` for faster access.
    - Download BioASQ (Edition 12B Train Set) for random PuMed Negatives \url{https://participants-area.bioasq.org/datasets/} into `./BioASQ-training12b/training12b_new.json`
    - Build and index the Pubmed corpus using elasticsearch via `python -m index_elasticsearch`. Start Elasticsearch.
5. Run run_edel_po.sh for training and evaluating the retrieval model on the Precision Oncology Dataset
6. Run run_edel_ptm.sh for training and evaluating the retrieval model on the Post-Translational Modification Dataset