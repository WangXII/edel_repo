import csv
import json
import requests

filename = '20251030_OncoKB_ActionableGenes.tsv'

def get_post_query(hugo_symbol, alteration):
    return {
        "alteration": alteration,
        "alteration_type": None,
        "evidenceTypes": [
        ],
        "gene": {
        "entrezGeneId": 0,
        "hugoSymbol": hugo_symbol
        },
        "id": None,
        "proteinEnd": None,
        "proteinStart": None,
        "referenceGenome": "GRCh37",
        "tumorType": None
    }

def get_post_queries(alterations):
    queries = []
    for alteration in alterations:
        hugo_symbol = alteration[0]
        variant = alteration[1]
        queries.append(get_post_query(hugo_symbol, variant))
    return queries

all_alterations = set()

with open(filename, 'r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            # Skip header
            continue
        level, gene, alterations, cancer_types, drugs = row

        first_alteration = alterations.split(', ')[0]
        all_alterations.add((gene, first_alteration))

        print(row)

print(len(all_alterations))

url = 'https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange'

queries = get_post_queries(all_alterations)

payload = json.dumps(queries[:])

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer INSERT_YOUR_API_KEY_HERE'
}

response = requests.post(url, data=payload, headers=headers)

if response.status_code == 200:  # Check if the request was successful (status code 200)
    print("Got the response")  # Print the response content
else:
    print('Request failed with status code', response.status_code)

with open('20251030_OncoKB_References.json', 'w') as file:
    file.write(response.text)
