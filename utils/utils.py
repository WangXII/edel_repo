from typing import Dict, Generator, List

import polars as pl
from lxml import etree
from tqdm import tqdm

from models.transformers import InputExample


def transform_input_examples_asym(examples: List[InputExample]) -> List[Dict]:
    asym_examples = []
    for example in tqdm(examples, desc="Transforming input examples to asym format"):
        query = {"query": example.texts[0]}
        doc = {"doc": example.texts[1]}
        try:
            new_input_example = InputExample(
                guid=example.guid,
                texts=[query, doc],
                label=example.label,
                margin=example.margin,
                noisy_bool=example.noisy_bool,
                query_id=example.query_id,
                doc_id=example.doc_id,
            )
        except AttributeError:
            new_input_example = InputExample(
                guid=example.guid,
                texts=[query, doc],
                label=example.label,
                margin=example.margin,
                noisy_bool=example.noisy_bool,
                query_id=-1,
                doc_id=-1,
            )
        asym_examples.append(new_input_example)
    return asym_examples


def process_pubmed_xmls(shards: List[str]) -> Generator[Dict, None, None]:
    for xml_file in shards:
        tree = etree.parse(xml_file)
        pubmed_articles = tree.xpath("//PubmedArticle")
        for article in pubmed_articles:
            pmid = article.xpath("./MedlineCitation/PMID/text()")[0]
            if len(article.xpath("./MedlineCitation/Article/ArticleTitle/text()")) == 0:
                continue
            title = article.xpath("./MedlineCitation/Article/ArticleTitle/text()")[0]
            # TODO May have multiple labels, split into different texts in a future version
            abstract = article.xpath(
                "./MedlineCitation/Article/Abstract/AbstractText/text()"
            )
            abstract = " ".join(abstract)
            year = article.xpath(
                "./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year/text()"
            )
            month = article.xpath(
                "./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Month/text()"
            )
            day = article.xpath(
                "./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Day/text()"
            )
            contents = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "month": month,
                "day": day,
            }
            yield contents


def create_pubmed_dataset_from_xmls(output_file: str):
    shards = [
        f"edel_repo_cache/pubmed/pubmed26n{i:04d}.xml"
        for i in range(1, 1167)
    ]
    dataset = Dataset.from_generator(
        process_pubmed_xmls,
        gen_kwargs={"shards": shards},
        num_proc=16,
    )
    dataset.save_to_disk(output_file)


def get_retriever_query(
    gene_name: str,
    gene_synonyms: List[str],
    gene_full_name: str,
    variant: str,
    synonyms_in_query: bool = True,
) -> str:
    if synonyms_in_query:
        genes = [gene_name]
        if gene_full_name:
            genes.append(gene_full_name)
        k = 4
        # Additionally, get the two shortest synonyms which do not share
        # any prefix greater equal three characters
        for gene in sorted(gene_synonyms, key=len):
            if len(gene) >= 3 and not any(
                [gene.startswith(prefix[:3]) for prefix in genes]
            ):
                genes.append(gene)
            if len(genes) >= k:
                break
            if len(gene) > 8:  # For long synonyms, we only need one
                break

        # If the k + 1 st or k + 2 nd synonym are shorter than 5 characters, keep them
        if len(genes) >= k + 1 and len(genes[k]) > 5:
            genes = genes[:k]
        elif len(genes) >= k + 2 and len(genes[k + 1]) > 5:
            genes = genes[: k + 1]

        # Join the genes with a comma and "and" for the last one
        if len(genes) == 2:
            gene_synonyms_str = " (also known as {})".format(genes[1])
        elif len(genes) > 2:
            gene_synonyms_str = " (also known as {} and {})".format(
                ", ".join(genes[1:-1]), genes[-1]
            )
        else:
            gene_synonyms_str = ""

        query = "Treatment for gene {}{} and variant {}.".format(
            genes[0], gene_synonyms_str, variant
        )
    else:
        query = "Treatment for gene {} and variant {}.".format(gene_name, variant)
    return query


def get_dataset_dict(examples, datasplit="dev", split_examples=None):
    current_examples = examples.dataset
    current_split = None
    if datasplit == "dev":
        current_split = examples.dev_split
        if split_examples is not None:
            current_split = split_examples.dev_split
    elif datasplit == "test":
        current_split = examples.test_split
        if split_examples is not None:
            current_split = split_examples.test_split
    elif datasplit == "train":
        current_split = examples.train_split
        if split_examples is not None:
            current_split = split_examples.train_split

    dataset_dict = {}
    count = 0
    count_with_drugs = 0
    unique_variants_with_drugs = set()
    for eg_id in tqdm(current_split, desc="Processing dataset"):
        variants = set(
            current_examples.filter(lambda x: x["entrez_id"] == eg_id)["variant"]
        )
        for variant in sorted(variants):
            possible_drug_dataset = current_examples.filter(
                lambda x: x["entrez_id"] == eg_id
                and x["variant"] == variant
                and len(x["drugs"]) > 0
            )

            count += 1
            for drug_entry in possible_drug_dataset:
                if (eg_id, variant) not in unique_variants_with_drugs:
                    unique_variants_with_drugs.add((eg_id, variant))
                    count_with_drugs += 1
                dataset_dict.setdefault((eg_id, variant), {})
                dataset_dict[(eg_id, variant)]["entrez_id"] = eg_id
                dataset_dict[(eg_id, variant)]["gene"] = drug_entry["gene"]
                dataset_dict[(eg_id, variant)]["gene_synonyms"] = drug_entry[
                    "gene_synonyms"
                ]
                dataset_dict[(eg_id, variant)]["gene_full_name"] = drug_entry[
                    "gene_full_name"
                ]
                dataset_dict[(eg_id, variant)]["variant"] = variant
                dataset_dict[(eg_id, variant)]["variant_synonyms"] = drug_entry[
                    "variant_synonyms"
                ]
                dataset_dict[(eg_id, variant)].setdefault("drugs", []).extend(
                    drug_entry["drugs"]
                )
                dataset_dict[(eg_id, variant)].setdefault("drugs_synonyms", []).extend(
                    drug_entry["drugs_synonyms"]
                )
                dataset_dict[(eg_id, variant)].setdefault("drugs_list", []).append(
                    drug_entry["drugs"]
                )
                dataset_dict[(eg_id, variant)].setdefault(
                    "drugs_synonyms_list", []
                ).append(drug_entry["drugs_synonyms"])
                dataset_dict[(eg_id, variant)].setdefault("source_type", []).append(
                    drug_entry["source_type"]
                )
                dataset_dict[(eg_id, variant)].setdefault("citation_id", []).append(
                    drug_entry["citation_id"]
                )

                dataset_dict[(eg_id, variant)].setdefault("gene_synonyms_in_raw_text", []).append(
                    drug_entry["gene_synonyms_in_raw_text"]
                )
                dataset_dict[(eg_id, variant)].setdefault("variant_in_raw_text", []).append(
                    drug_entry["variant_in_raw_text"]
                )
                dataset_dict[(eg_id, variant)].setdefault("drugs_synonyms_in_raw_text", []).append(
                    drug_entry["drugs_synonyms_in_raw_text"]
                )

                dataset_dict[(eg_id, variant)].setdefault("disease", []).append(
                    drug_entry["disease"]
                )
                # Only is relevant for CIViC dataset till now
                # dataset_dict[(eg_id, variant)].setdefault("evidence_type", []).append(
                #     drug_entry["evidence_type"]
                # )
                # dataset_dict[(eg_id, variant)].setdefault(
                #     "evidence_direction", []
                # ).append(drug_entry["evidence_direction"])
                # dataset_dict[(eg_id, variant)].setdefault("evidence_level", []).append(
                #     drug_entry["evidence_level"]
                # )
                # dataset_dict[(eg_id, variant)].setdefault(
                #     "clinical_significance", []
                # ).append(drug_entry["clinical_significance"])

    return dataset_dict, count, count_with_drugs


def get_dataset_dict_uniprot(examples, datasplit="dev"):
    current_examples = None
    current_split = None
    if datasplit == "dev":
        current_examples = examples.dev
        current_split = examples.dev_split
    elif datasplit == "test":
        current_examples = examples.test
        current_split = examples.test_split
    elif datasplit == "train":
        current_examples = examples.train
        current_split = examples.train_split

    dataset_dict = {}
    count = 0
    count_with_ptm = 0
    unique_ptms_with_catalysts = set()
    for uniprot_id in tqdm(current_split, desc="Processing dataset"):
        ptms = set(
            (example["ptm_type"], example["residue"], example["position"])
            for example in current_examples.filter(
                lambda x: x["primary_accession"] == uniprot_id
            )
        )
        for ptm in sorted(ptms):
            possible_catalyst_dataset = current_examples.filter(
                lambda x: x["primary_accession"] == uniprot_id
                and x["ptm_type"] == ptm[0]
                and x["residue"] == ptm[1]
                and x["position"] == ptm[2]
                and len(x["catalysts"]) > 0
            )

            count += 1
            for catalyst_entry in possible_catalyst_dataset:
                if (uniprot_id, ptm) not in unique_ptms_with_catalysts:
                    unique_ptms_with_catalysts.add((uniprot_id, ptm))
                    count_with_ptm += 1
                dataset_dict.setdefault((uniprot_id, ptm), {})
                dataset_dict[(uniprot_id, ptm)]["primary_accession"] = uniprot_id
                dataset_dict[(uniprot_id, ptm)]["substrate"] = catalyst_entry[
                    "substrate"
                ]
                dataset_dict[(uniprot_id, ptm)]["substrate_full_name"] = catalyst_entry[
                    "substrate_full_name"
                ]
                dataset_dict[(uniprot_id, ptm)]["substrate_synonyms"] = catalyst_entry[
                    "substrate_synonyms"
                ]
                dataset_dict[(uniprot_id, ptm)]["ptm_type"] = ptm[0]
                dataset_dict[(uniprot_id, ptm)]["residue"] = ptm[1]
                dataset_dict[(uniprot_id, ptm)]["position"] = ptm[2]
                dataset_dict[(uniprot_id, ptm)].setdefault("catalysts", []).extend(
                    catalyst_entry["catalysts"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault(
                    "catalysts_synonyms", []
                ).extend(catalyst_entry["catalysts_synonyms"])
                dataset_dict[(uniprot_id, ptm)].setdefault("catalysts_list", []).append(
                    catalyst_entry["catalysts"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault(
                    "catalysts_synonyms_list", []
                ).append(catalyst_entry["catalysts_synonyms"])
                dataset_dict[(uniprot_id, ptm)].setdefault("citation_id", []).append(
                    catalyst_entry["citation_id"]
                )

                dataset_dict[(uniprot_id, ptm)].setdefault("substrate_synonyms_in_raw_text", []).append(
                    catalyst_entry["substrate_synonyms_in_raw_text"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault("ptm_type_in_raw_text", []).append(
                    catalyst_entry["ptm_type_in_raw_text"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault("residue_in_raw_text", []).append(
                    catalyst_entry["residue_in_raw_text"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault("position_in_raw_text", []).append(
                    catalyst_entry["position_in_raw_text"]
                )
                dataset_dict[(uniprot_id, ptm)].setdefault("catalysts_synonyms_in_raw_text", []).append(
                    catalyst_entry["catalysts_synonyms_in_raw_text"]
                )

    return dataset_dict, count, count_with_ptm


def get_dataset_dict_from_csv(csv_file):
    df = pl.read_csv(csv_file, separator=";")

    dataset_dict = {}
    for row in df.rows(named=True):
        gene = row["Gene"]
        gene_synonyms = [gene]
        variant = row["Alteration"]
        variant_synonyms = [variant]
        disease = row["Disease"]
        drugs = row["Therapy"].split(",")
        drugs_synonyms = [[drug] for drug in drugs]
        pmid = row["PMID"]

        dataset_dict.setdefault((gene, variant), {})
        dataset_dict[(gene, variant)]["entrez_id"] = gene
        dataset_dict[(gene, variant)]["gene"] = gene
        dataset_dict[(gene, variant)]["gene_synonyms"] = gene_synonyms
        dataset_dict[(gene, variant)]["gene_full_name"] = gene
        dataset_dict[(gene, variant)]["variant"] = variant
        dataset_dict[(gene, variant)]["variant_synonyms"] = variant_synonyms
        dataset_dict[(gene, variant)].setdefault("drugs", []).extend(drugs)
        dataset_dict[(gene, variant)].setdefault("drugs_synonyms", []).extend(
            drugs_synonyms
        )
        dataset_dict[(gene, variant)].setdefault("drugs_list", []).append(drugs)
        dataset_dict[(gene, variant)].setdefault("drugs_synonyms_list", []).append(
            drugs_synonyms
        )
        dataset_dict[(gene, variant)].setdefault("source_type", []).append("PubMed")
        dataset_dict[(gene, variant)].setdefault("citation_id", [])
        if pmid:
            dataset_dict[(gene, variant)]["citation_id"].append(pmid)

        dataset_dict[(gene, variant)].setdefault("disease", []).append(disease)
        dataset_dict[(gene, variant)].setdefault("evidence_type", []).append("")
        dataset_dict[(gene, variant)].setdefault("evidence_direction", []).append("")
        dataset_dict[(gene, variant)].setdefault("evidence_level", []).append("")
        dataset_dict[(gene, variant)].setdefault("clinical_significance", []).append("")

    return dataset_dict

if __name__ == "__main__":
    create_pubmed_dataset_from_xmls("edel_repo_cache/datasets/pubmed.dataset")
