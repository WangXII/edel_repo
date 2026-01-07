import json

import models.margin_config as margin_config
import models.margin_config_uniprot as margin_config_uniprot
from models.civic_oncokb_retriever import CiVICOncoKBRetriever
from models.uniprot_retriever import UniprotRetriever
from po_datasets.civic import CiVICExamples
from po_datasets.concat_dataset import ConcatExamples
from po_datasets.dataset import DatasetExamples
from po_datasets.onco_kb import OncoKBExamples
from po_datasets.uniprot_ptms import UniProtPTMExamples

mode = "raw_text"

civic: DatasetExamples = CiVICExamples(mode=mode)
onco_kb: DatasetExamples = OncoKBExamples(mode=mode)
civic_oncokb: DatasetExamples = ConcatExamples([civic, onco_kb], mode=mode, make_new_datasplits=True)
uniprot: DatasetExamples = UniProtPTMExamples(mode=mode)

po_retriever = CiVICOncoKBRetriever(
    examples=civic_oncokb,
    cache_file_prefix="civic_oncokb_margin_classes_v1",
    cache=True,
    margin_config=margin_config.margin_classes_v1,
    use_batch_negatives=True,
)
uniprot_retriever = UniprotRetriever(
    examples=uniprot,
    cache_file_prefix="uniprot_margin_classes_v1",
    cache=True,
    margin_config=margin_config_uniprot.margin_classes_uniprot_v1,
    use_batch_negatives=True,
)

all_datasets = [
    (po_retriever.train, "civic_oncokb_train"),
    (po_retriever.dev, "civic_oncokb_dev"),
    (po_retriever.test, "civic_oncokb_test"),
    (uniprot_retriever.train, "uniprot_train"),
    (uniprot_retriever.dev, "uniprot_dev"),
    (uniprot_retriever.test, "uniprot_test"),
]
print("All datasets loaded")

print(po_retriever.train[0])

cache_prefix = "edel_repo_cache/medcpt_all_pos_examples/"

# Example data
# <InputExample> label: 1, margin: 0.04894348370484647, query: Treatment for gene LRP1B and variant EXON 12-22 DELETION. ,
# document: LRP1B deletion in high-grade serous ovarian cancers is associated with acquired chemotherapy resistance to liposomal doxorubicin.[SEP]High-grade serous cancer (HGSC),
# the most common subtype of ovarian cancer, often becomes resistant to chemotherapy, leading to poor patient outcomes... ,
# qid: 0, doc_id: 22896685

# MedCPT files

# train.jsonl is a jsonline file where each line contains a json of query-article article and the number of click
# $ head train_example.jsonl
# {"qid": "0", "pmid": "15858239", "click": 1}
# {"qid": "0", "pmid": "15829955", "click": 1}
# {"qid": "0", "pmid": "6650562", "click": 1}
# {"qid": "0", "pmid": "12239580", "click": 1}
# {"qid": "0", "pmid": "21995290", "click": 1}
# {"qid": "0", "pmid": "23001136", "click": 1}
# {"qid": "0", "pmid": "15617541", "click": 1}
# {"qid": "0", "pmid": "8896569", "click": 1}
# {"qid": "0", "pmid": "20598273", "click": 1}
# {"qid": "1", "pmid": "23959273", "click": 1}

# # qid2info.json is a json dict where keys are qids and values are the queries (BioASQ questions in the example)
# $ head qid2info_example.json
# {
#     "0": "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
#     "1": "List signaling molecules (ligands) that interact with the receptor EGFR?",
#     "2": "Is the protein Papilin secreted?",
#     "3": "Are long non coding RNAs spliced?",
#     "4": "Is RANKL secreted from the cells?",
#     "5": "Does metformin interfere thyroxine absorption?",
#     "6": "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
#     "7": "Which acetylcholinesterase inhibitors are used for treatment of myasthenia gravis?",
#     "8": "Has Denosumab (Prolia) been approved by FDA?",

# # pmid2info.json is a json dict where keys are pmids and values are a tuple (list) of title and abstract.
# $ head pmid2info_example.json
# {
#     "15858239": [
#         "[The role of ret gene in the pathogenesis of Hirschsprung disease].",
#         "Hirschsprung disease is a congenital disorder with the incidence of 1 per 5000 live births, characterized by the absence of intestinal ganglion cells. In the etiology of Hirschsprung disease various genes play a role; these are: RET, EDNRB, GDNF, EDN3 and SOX10, NTN3, ECE1, Mutations in these genes may result in dominant, recessive or multifactorial patterns of inheritance. Diverse models of inheritance, co-existence of numerous genetic disorders and detection of numerous chromosomal aberrations together with involvement of various genes confirm the genetic heterogeneity of Hirschsprung disease. Hirschsprung disease might well serve as a model for many complex disorders in which the search for responsible genes has only just been initiated. It seems that the most important role in its genetic etiology plays the RET gene, which is involved in the etiology of at least four diseases. This review focuses on recent advances of the importance of RET gene in the etiology of Hirschsprung disease."
#     ],
#     "15829955": [
#         "A common sex-dependent mutation in a RET enhancer underlies Hirschsprung disease risk.",
#         "The identification of common variants that contribute to the genesis of human inherited disorders remains a significant challenge. Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes. We have used family-based association studies to identify a disease interval, and integrated this with comparative and functional genomic analysis to prioritize conserved and functional elements within which mutations can be sought. We now show that a common non-coding RET variant within a conserved enhancer-like sequence in intron 1 is significantly associated with HSCR susceptibility and makes a 20-fold greater contribution to risk than rare alleles do. This mutation reduces in vitro enhancer activity markedly, has low penetrance, has different genetic effects in males and females, and explains several features of the complex inheritance pattern of HSCR. Thus, common low-penetrance variants, identified by association studies, can underlie both common and rare diseases."
#     ],
# }

for dataset, file_prefix in all_datasets:
    relevance_list = []
    qid2info = {}
    pmid2info = {}
    for example in dataset:
        qid = example.query_id
        pmid = example.doc_id
        relevance = example.label
        query = example.texts[0]
        doc_abst, doc_title = example.texts[1].split("[SEP]")
        relevance_list.append({"qid": qid, "pmid": pmid, "click": relevance})
        qid2info[qid] = query
        pmid2info[pmid] = [doc_abst, doc_title]
    with open(f"{cache_prefix}train_{file_prefix}.jsonl", "w") as f:
        for line in relevance_list:
            f.write(json.dumps(line) + "\n")
    with open(f"{cache_prefix}qid2info_{file_prefix}.json", "w") as f:
        json.dump(qid2info, f)
    with open(f"{cache_prefix}pmid2info_{file_prefix}.json", "w") as f:
        json.dump(pmid2info, f)
