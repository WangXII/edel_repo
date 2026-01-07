# Description: This script is used to build a sqlite database from the NCBI gene_info file.
from pathlib import Path

from sqlitedict import SqliteDict
from tqdm import tqdm

GENE_NAMES_CACHE = "edel_repo_cache/gene_names.sqlite"

NCBI_GENE_INFO_FILE = "edel_repo_cache/gene_info"


def build_ncbi_gene_id_db():
    document_path = NCBI_GENE_INFO_FILE
    gene_name_dict = {}
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            index = 0

            columns = [
                "tax_id",  # 0
                "GeneID",  # 1
                "Symbol",  # 2
                "LocusTag",  # 3
                "Synonyms",  # 4
                "dbXrefs",  # 5
                "chromosome",  # 6
                "map_location",  # 7
                "description",  # 8
                "type_of_gene",  # 9
                "Symbol_from_nomenclature_authority",  # 10
                "Full_name_from_nomenclature_authority",  # 11
                "Nomenclature_status",  # 12
                "Other_designations",  # 13
                "Modification_date",  # 14
                "Feature_type",  # 15
            ]

            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split("\t")
                values[-1] = values[-1].strip()

                if len(values) != len(columns):
                    continue

                key_value = values[1]
                gene_name_dict[key_value] = {}
                for i, column in enumerate(columns):
                    if column == "GeneID":
                        continue
                    elif i in [4, 13]:
                        gene_name_dict[key_value][column] = values[i].split("|")
                    elif i in [0, 2, 8, 10, 11]:
                        gene_name_dict[key_value][column] = values[i]
                    else:
                        continue

                # if index > 100:
                #     print(key_value)
                #     print(gene_name_dict[key_value])
                index += 1
                # if index > 110:
                #     break

    gene_names_sql_dict = SqliteDict(
        GENE_NAMES_CACHE, tablename="gene_id_to_names", flag="w", autocommit=False
    )
    gene_names_sql_dict.update(gene_name_dict.items())
    gene_names_sql_dict.commit()
    gene_names_sql_dict.close()


def build_ncbi_gene_names_db():
    document_path = NCBI_GENE_INFO_FILE
    gene_name_dict = {}
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            index = 0

            columns = [
                "tax_id",  # 0
                "GeneID",  # 1
                "Symbol",  # 2
                "LocusTag",  # 3
                "Synonyms",  # 4
                "dbXrefs",  # 5
                "chromosome",  # 6
                "map_location",  # 7
                "description",  # 8
                "type_of_gene",  # 9
                "Symbol_from_nomenclature_authority",  # 10
                "Full_name_from_nomenclature_authority",  # 11
                "Nomenclature_status",  # 12
                "Other_designations",  # 13
                "Modification_date",  # 14
                "Feature_type",  # 15
            ]

            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split("\t")
                values[-1] = values[-1].strip()

                if len(values) != len(columns):
                    continue

                key_value = f"{values[2]}_{values[0]}"
                gene_name_dict[key_value] = {}
                for i, column in enumerate(columns):
                    if column == "GeneID":
                        continue
                    elif i in [4, 13]:
                        gene_name_dict[key_value][column] = values[i].split("|")
                    elif i in [0, 1, 2, 8, 10, 11]:
                        gene_name_dict[key_value][column] = values[i]
                    else:
                        continue

                # if index > 100:
                #     print(key_value)
                #     print(gene_name_dict[key_value])
                index += 1
                # if index > 110:
                #     break

    gene_names_sql_dict = SqliteDict(
        GENE_NAMES_CACHE, tablename="gene_names_tax_to_id", flag="w", autocommit=False
    )
    gene_names_sql_dict.update(gene_name_dict.items())
    gene_names_sql_dict.commit()
    gene_names_sql_dict.close()


# Main
if __name__ == "__main__":
    # build_ncbi_gene_id_db()
    build_ncbi_gene_names_db()
