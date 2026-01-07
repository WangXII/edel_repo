import json

import faiss
import numpy as np
import sklearn.preprocessing
from sklearn.metrics.pairwise import paired_cosine_distances

from models.transformers import CustomSentenceTransformer
from po_datasets.dataset import ElasticsearchHelper


class RetrieverEvaluator:
    def __init__(
        self,
        es_index="20231127_pubmed",
        top_k_hits=[10, 100],
    ):

        self.top_k_hits = top_k_hits
        ElasticsearchHelper.es_index = es_index

    def get_bm25_results(self, query):
        # BM25 retrieval from vector space model
        # We do not use gene synonyms for BM25 as there might be too many

        bm25_hits = ElasticsearchHelper.lexical_query(
            query, number=max(self.top_k_hits)
        )

        bm25_scores, bm25_pm_ids, bm25_texts = [], [], []
        for i, hit in enumerate(bm25_hits):
            # retrieved_text = hit["_source"]["content"]
            retrieved_text = hit["_source"]["text"]
            bm25_scores.append(hit["_score"])
            # bm25_pm_ids.append(hit["_source"]["name"])
            bm25_pm_ids.append(int(hit["_source"]["pmid"]))
            bm25_texts.append(retrieved_text)

        return bm25_scores, bm25_pm_ids, bm25_texts


class BiEncoderRetrieverEvaluator(RetrieverEvaluator):
    def __init__(
        self,
        bi_encoder_articles,
        bi_encoder_queries,
        document_index,
        flat_index_file,
        flat_index_mapping_file,
        pooling="cls",
        use_asym_bi_encoder=False,
        use_dot_product=False,
        only_use_flat_index=False,
        top_k_hits=[10, 100, 1000],
        es_index="20231127_pubmed",
    ):
        if use_asym_bi_encoder:
            # In asym biencoder, we need to get both query and document bi-encoders out of the specified pytorch bi_encoder
            print("Loading asym bi-encoder")
            modules_list = list(
                CustomSentenceTransformer(
                    bi_encoder_articles,
                    pooling=pooling,
                ).modules()
            )
            doc_word_embedding_model = modules_list[0]._modules["0"].sub_modules["doc"][0]
            doc_pooling_model = modules_list[0]._modules["0"].sub_modules["doc"][1]
            self.bi_encoder_articles = CustomSentenceTransformer(
                modules=[doc_word_embedding_model, doc_pooling_model]
            )
            query_word_embedding_model = modules_list[0]._modules["0"].sub_modules["query"][0]
            query_pooling_model = modules_list[0]._modules["0"].sub_modules["query"][1]
            self.bi_encoder_queries = CustomSentenceTransformer(
                modules=[query_word_embedding_model, query_pooling_model]
            )
            # print(modules_list[0]._modules["0"].sub_modules)
        else:
            self.bi_encoder_articles = CustomSentenceTransformer(
                bi_encoder_articles, pooling=pooling
            )
            if bi_encoder_queries != bi_encoder_articles:
                self.bi_encoder_queries = CustomSentenceTransformer(
                    bi_encoder_queries, pooling=pooling
                )
            else:
                self.bi_encoder_queries = self.bi_encoder_articles

        co = faiss.GpuClonerOptions()
        co.useFloat16 = True

        res = faiss.StandardGpuResources()

        # TODO: Enable this again
        # self.index = faiss.read_index(document_index)
        # Test out baseline models
        # TODO: Enable this again
        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
        self.index = None

        self.flat_index = faiss.read_index(flat_index_file)
        self.flat_index_mapping_file = flat_index_mapping_file

        self.only_use_flat_index = only_use_flat_index
        self.use_asym_bi_encoder = use_asym_bi_encoder

        self.use_dot_product = use_dot_product

        # Read the json file into a dictionary
        count_missing = 0
        with open(self.flat_index_mapping_file, "r") as mapping_file:
            self.flat_index_mapping = json.load(mapping_file)
        # Get the inverse mapping
        self.flat_index_mapping_inverted = {
            v: k for k, v in self.flat_index_mapping.items()
        }
        index_variable = 0
        for key, _ in self.flat_index_mapping_inverted.items():
            if key != index_variable:
                count_missing += 1
                index_variable = key
            index_variable += 1
        print(f"Missing {count_missing} entries in flat index mapping")

        super().__init__(
            es_index=es_index,
            top_k_hits=top_k_hits,
        )

    def distance_function(self, query_embedding, document_embeddings):
        query_embeddings = np.tile(query_embedding[0], (len(document_embeddings), 1))
        if self.use_dot_product:
            return (query_embeddings * document_embeddings).sum(axis=1)
        else:
            return 1 - paired_cosine_distances(
                document_embeddings,
                np.tile(query_embedding[0], (len(document_embeddings), 1)),
            )

    def get_query_embedding(self, query):
        if self.use_dot_product:
            # query_embedding = np.array(
            #     [
            #         self.bi_encoder_queries.encode(query, convert_to_numpy=True).astype(
            #             "float32"
            #         )
            #     ]
            # )
            query_embedding = self.get_bi_encoded_results(
                [query], "query", self.bi_encoder_queries
            )
        else:
            # Normalize the query embedding
            # query_embedding = self.bi_encoder_queries.encode(
            #     query, convert_to_numpy=True
            # )
            query_embedding = self.get_bi_encoded_results(
                [query], "query", self.bi_encoder_queries
            )
            query_embedding = sklearn.preprocessing.normalize(query_embedding, axis=1)
            # Print l2 norm of query embedding
            # print(query_embedding)
            # print(np.linalg.norm(query_embedding))
        return query_embedding

    def get_faiss_results(self, query):
        query_embedding = self.get_query_embedding(query)
        faiss_k = 1000

        # Faiss retrieval from embeddings
        if not self.only_use_flat_index:
            distances, corpus_ids = self.index.search(query_embedding, faiss_k)

            faiss_hits = [
                {"corpus_id": int(id), "score": score}
                for id, score in zip(corpus_ids[0], distances[0])
            ]
        else:
            distances, corpus_ids = self.flat_index.search(
                query_embedding, max(self.top_k_hits)
            )

            faiss_hits = [
                {
                    "corpus_id": int(self.flat_index_mapping_inverted[id]),
                    "idx": id,
                    "score": score,
                }
                for id, score in zip(corpus_ids[0], distances[0])
                if id in self.flat_index_mapping_inverted.keys()
            ]

        faiss_scores, faiss_pm_ids = [], []
        for i, hit in enumerate(faiss_hits[0 : max(self.top_k_hits)]):
            faiss_scores.append(hit["score"])
            faiss_pm_ids.append(hit["corpus_id"])

        return faiss_scores, faiss_pm_ids

    def get_bi_encoded_results(self, texts, doc_key, encoder):
        # if self.use_asym_bi_encoder:
        #     texts = [{doc_key: text} for text in texts]
        embeddings = encoder.encode(texts, convert_to_numpy=True, batch_size=16)
        return embeddings

    def get_bi_encoder_results(self, query, faiss_texts, faiss_pm_ids):
        # Bi-encoder retrieval from embeddings
        # Embed faiss_k documents and compute their cosine similarity with the query embedding
        # Then sort the documents by their cosine similarity and get the top_k_hits
        # if self.use_asym_bi_encoder:
        #     faiss_texts = [{"doc": text} for text in faiss_texts]
        bi_encoder_scores, bi_encoder_pm_ids = [], []
        # bi_encoder_embeddings = self.bi_encoder_articles.encode(
        #     faiss_texts, convert_to_numpy=True, batch_size=16
        # )
        bi_encoder_embeddings = self.get_bi_encoded_results(
            faiss_texts, "doc", self.bi_encoder_articles
        )

        query_embedding = self.get_query_embedding(query)
        bi_encoder_cosine_scores = self.distance_function(
            query_embedding, bi_encoder_embeddings
        )

        # print(bi_encoder_cosine_scores.shape)
        # print(bi_encoder_cosine_scores)

        sorted_bi_encoder_scores = sorted(
            zip(bi_encoder_cosine_scores, range(len(bi_encoder_cosine_scores))),
            reverse=True,
        )

        for i, (score, idx) in enumerate(
            sorted_bi_encoder_scores[0 : max(self.top_k_hits)]
        ):
            bi_encoder_scores.append(score)
            bi_encoder_pm_ids.append(faiss_pm_ids[idx])

        return bi_encoder_scores, bi_encoder_pm_ids

    def get_bm25_faiss_results(self, query):
        # BM25 retrieval from vector space model and faiss for reconstruction
        # We do not use gene or variant synonyms for BM25 as there might be too many
        # TODO: Enable this again
        # BM25 FAISS queries changed to natural language queries as well
        # bm25_faiss_hits = self.dataset_examples.query_keywords(
        #     [[example["gene"]], [example["variant"]], self.drug_triggers],
        #     number=1000,
        # )
        # bm25_faiss_hits.extend(
        #     self.dataset_examples.query_keywords(
        #         [[example["gene"]], self.drug_triggers], number=2000
        #     )
        # )
        # bm25_faiss_hits.extend(
        #     self.dataset_examples.query_keywords([[example["gene"]]], number=4000)
        # )
        bm25_faiss_hits = ElasticsearchHelper.lexical_query(query, number=1000)

        bm25_faiss_scores, tmp_pm_ids, bm25_faiss_texts_dict = [], [], {}
        for i, hit in enumerate(bm25_faiss_hits):
            retrieved_text = hit["_source"]["text"]
            bm25_faiss_scores.append(hit["_score"])
            pmid = int(hit["_source"]["pmid"])
            tmp_pm_ids.append(pmid)
            bm25_faiss_texts_dict[int(pmid)] = retrieved_text

        # https://stackoverflow.com/questions/74432548/c-faiss-how-to-search-in-subsets
        # https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query
        filter_ids = [
            int(self.flat_index_mapping[str(pmid)])
            for pmid in tmp_pm_ids
            if str(pmid) in self.flat_index_mapping.keys()
        ]
        # filter_ids = [34896892]
        id_selector = faiss.IDSelectorArray(filter_ids)
        # print(filter_ids)
        # print(len(filter_ids))
        # print(id_selector)

        # filter_ids = [0, 1, 2, 3]
        # id_selector = faiss.IDSelectorArray(filter_ids)
        # filtered_distances, filtered_indices = self.index.search(query_embedding, top_k_hits, params=faiss.SearchParametersIVF(sel=id_selector))

        query_embedding = self.get_query_embedding(query)
        # print("Query embedding shape in bm25_faiss_results")
        # print(query_embedding.shape)
        # print(tmp_pm_ids)
        # print(filter_ids)
        # print(len(filter_ids))
        filtered_distances, filtered_corpus_ids = self.flat_index.search(
            query_embedding, 2048, params=faiss.SearchParametersIVF(sel=id_selector)
        )

        # print("Filtered IDs")
        # print(filtered_corpus_ids[0])
        # print(len(filtered_corpus_ids[0]))

        faiss_hits = [
            {
                "corpus_id": self.flat_index_mapping_inverted[id],
                "idx": id,
                "score": score,
            }
            for id, score in zip(filtered_corpus_ids[0], filtered_distances[0])
            if id in self.flat_index_mapping_inverted.keys()
        ]

        # Remove duplicates
        bm25_faiss_corpus_ids = set()
        for hit in faiss_hits:
            bm25_faiss_corpus_ids.add(hit["corpus_id"])

        # print(bm25_faiss_corpus_ids)

        unique_faiss_hits = []
        for faiss_hit in faiss_hits:
            if faiss_hit["corpus_id"] in bm25_faiss_corpus_ids:
                unique_faiss_hits.append(faiss_hit)
                bm25_faiss_corpus_ids.remove(faiss_hit["corpus_id"])

        bm25_faiss_scores, bm25_faiss_pm_ids, bm25_faiss_texts = [], [], []
        for i, hit in enumerate(unique_faiss_hits[0 : max(self.top_k_hits)]):
            if hit["idx"] in filter_ids:
                bm25_faiss_scores.append(hit["score"])
                bm25_faiss_pm_ids.append(int(hit["corpus_id"]))
                bm25_faiss_texts.append(bm25_faiss_texts_dict[int(hit["corpus_id"])])

        return bm25_faiss_scores, bm25_faiss_pm_ids, bm25_faiss_texts

    def check_similarity_scores(
        self, query, gold_doc_texts, faiss_texts, bm25_faiss_texts, bm25_texts
    ):
        query_embedding = self.get_query_embedding(query)
        gold_text_embeddings = self.get_bi_encoded_results(
            gold_doc_texts, "doc", self.bi_encoder_articles
        )
        faiss_text_embeddings = self.get_bi_encoded_results(
            faiss_texts[: max(self.top_k_hits)], "doc", self.bi_encoder_articles
        )
        bm25_faiss_text_embeddings = self.get_bi_encoded_results(
            bm25_faiss_texts[: max(self.top_k_hits)], "doc", self.bi_encoder_articles
        )
        bm25_text_embeddings = self.get_bi_encoded_results(
            bm25_texts[: max(self.top_k_hits)], "doc", self.bi_encoder_articles
        )

        # print(query_embedding.shape)
        # print(gold_text_embeddings.shape)
        # print(faiss_text_embeddings.shape)
        # print(bm25_faiss_text_embeddings.shape)
        # print(bm25_text_embeddings.shape)

        # print("Norm of query embedding")
        # print(np.linalg.norm(query_embedding))
        # print("Norm of first faiss text embedding")
        # print(np.linalg.norm(faiss_text_embeddings[0]))
        # print("Values of first faiss text embedding (first ten dimensions)")
        # print(faiss_text_embeddings[0][:10])
        # print("Norm of second faiss text embedding")
        # print(np.linalg.norm(faiss_text_embeddings[1]))
        # print("Values of second faiss text embedding (first ten dimensions)")
        # print(faiss_text_embeddings[1][:10])
        # print("First faiss text")
        # print(faiss_texts[0])
        # print("Second faiss text")
        # print(faiss_texts[1])
        # print("Cosine similarity between query and first faiss text")
        # print(faiss_text_cosine_scores[0])

        gold_text_cosine_scores = self.distance_function(
            query_embedding, gold_text_embeddings
        )
        faiss_text_cosine_scores = self.distance_function(
            query_embedding, faiss_text_embeddings
        )
        bm25_faiss_text_cosine_scores = self.distance_function(
            query_embedding, bm25_faiss_text_embeddings
        )
        bm25_text_cosine_scores = self.distance_function(
            query_embedding, bm25_text_embeddings
        )

        return (
            gold_text_cosine_scores,
            faiss_text_cosine_scores,
            bm25_faiss_text_cosine_scores,
            bm25_text_cosine_scores,
        )

    def get_gold_pmids_scores(self, query, gold_pmids, faiss_pm_ids):
        query_embedding = self.get_query_embedding(query)
        # Get gold pmids FAISS scores
        gold_pmids_faiss_mapped = [
            int(self.flat_index_mapping[str(pmid)])
            for pmid in gold_pmids
            if str(pmid) in self.flat_index_mapping.keys()
        ]
        gold_pmid_selector = faiss.IDSelectorArray(gold_pmids_faiss_mapped)
        (
            gold_filtered_distances,
            gold_pmids_faiss_mapped_sorted_by_score,
        ) = self.flat_index.search(
            query_embedding,
            len(gold_pmids),
            params=faiss.SearchParametersIVF(sel=gold_pmid_selector),
        )
        gold_faiss_scores = []
        for pmid, pmid_mapped in zip(gold_pmids, gold_pmids_faiss_mapped):
            # Find the index of the pmid in the sorted list
            index_pos = (
                gold_pmids_faiss_mapped_sorted_by_score[0].tolist().index(pmid_mapped)
            )
            gold_faiss_scores.append(gold_filtered_distances[0].tolist()[index_pos])
        if len(gold_faiss_scores) != len(gold_pmids):
            # Pad with -1
            gold_faiss_scores.extend(
                [-1 for i in range(len(gold_pmids) - len(gold_faiss_scores))]
            )

        # Also get gold BM25 scores
        gold_bm25_scores = [
            ElasticsearchHelper.explain_lexical_query(query, doc_id)
            for doc_id in gold_pmids
        ]

        # Check sanity check
        # Check that FAISS scores are the same as the cosine similarity scores
        faiss_pmids_mapped_to_faiss_ids = [
            int(self.flat_index_mapping[str(pmid)])
            for pmid in faiss_pm_ids[: max(self.top_k_hits)]
            if str(pmid) in self.flat_index_mapping.keys()
        ]
        faiss_pmid_selector = faiss.IDSelectorArray(faiss_pmids_mapped_to_faiss_ids)
        faiss_filtered_distances, _ = self.flat_index.search(
            query_embedding,
            len(faiss_pmids_mapped_to_faiss_ids),
            params=faiss.SearchParametersIVF(sel=faiss_pmid_selector),
        )

        return (
            gold_faiss_scores,
            gold_bm25_scores,
            faiss_filtered_distances,
        )
