import csv
import logging
import os
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import dot_score
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from .transformers import InputExample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value
    """

    def __init__(
        self,
        evaluators: Iterable[SentenceEvaluator],
        main_score_function=lambda scores: scores[-1],
    ):
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(
        self,
        model,
        model_2=None,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        scores = []
        for evaluator in self.evaluators:
            scores.append(evaluator(model, model_2, output_path, epoch, steps))

        return self.main_score_function(scores)


class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[int],
        margins: List[float],
        noisy_bools: List[bool],
        calculate_dot_product: bool = False,
        lambda_weight: float = 0.9,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.margins = margins
        self.noisy_bools = noisy_bools
        self.calculate_dot_product = calculate_dot_product
        self.lambda_weight = lambda_weight

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "binary_classification_evaluation"
            + ("_" + name if name else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "cossim_accuracy",
            "cossim_accuracy_threshold",
            "cossim_f1",
            "cossim_precision",
            "cossim_recall",
            "cossim_f1_threshold",
            "cossim_ap",
            # "manhattan_accuracy",
            # "manhattan_accuracy_threshold",
            # "manhattan_f1",
            # "manhattan_precision",
            # "manhattan_recall",
            # "manhattan_f1_threshold",
            # "manhattan_ap",
            # "euclidean_accuracy",
            # "euclidean_accuracy_threshold",
            # "euclidean_f1",
            # "euclidean_precision",
            # "euclidean_recall",
            # "euclidean_f1_threshold",
            # "euclidean_ap",
            # "dot_accuracy",
            # "dot_accuracy_threshold",
            # "dot_f1",
            # "dot_precision",
            # "dot_recall",
            # "dot_f1_threshold",
            # "dot_ap",
        ]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []
        margins = []
        noisy_bools = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
            margins.append(example.margin)
            noisy_bools.append(example.noisy_bool)
        return cls(sentences1, sentences2, scores, margins, noisy_bools, **kwargs)

    def __call__(
        self,
        model,
        model_2: SentenceTransformer = None,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        current_lr: float = -1,
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(
            "Binary Accuracy Evaluation of the model on "
            + self.name
            + " dataset"
            + out_txt
        )

        scores = self.compute_metrices(model, model_2)

        ap_score = max(scores[short_name]["ap"] for short_name in scores)
        f1_score = max(scores[short_name]["f1"] for short_name in scores)
        f1_threshold = max(scores[short_name]["f1_threshold"] for short_name in scores)
        min_distance_score = min(
            scores[short_name]["min_distance_score"] for short_name in scores
        )
        max_distance_score = max(
            scores[short_name]["max_distance_score"] for short_name in scores
        )
        loss = max(scores[short_name]["loss"] for short_name in scores)

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if "_" in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return (
            ap_score,
            f1_score,
            f1_threshold,
            min_distance_score,
            max_distance_score,
            loss,
        )

    def compute_metrices(self, model, model_2=None):
        if model_2 is not None:
            sentences1 = list(set(self.sentences1))
            sentences2 = list(set(self.sentences2))
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            embeddings2 = model_2.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            emb_dict1 = {sent: emb for sent, emb in zip(sentences1, embeddings1)}
            emb_dict2 = {sent: emb for sent, emb in zip(sentences2, embeddings2)}
            embeddings1 = [emb_dict1[sent] for sent in self.sentences1]
            embeddings2 = [emb_dict2[sent] for sent in self.sentences2]
        elif isinstance(self.sentences1[0], dict):
            tuples1 = {tuple(d.items()) for d in self.sentences1}
            lists1 = [tuple(d.items()) for d in self.sentences1]
            sentences1 = [dict(t) for t in tuples1]
            tuples2 = {tuple(d.items()) for d in self.sentences2}
            lists2 = [tuple(d.items()) for d in self.sentences2]
            sentences2 = [dict(t) for t in tuples2]
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            emb_dict1 = {sent: emb for sent, emb in zip(tuples1, embeddings1)}
            emb_dict2 = {sent: emb for sent, emb in zip(tuples2, embeddings2)}
            embeddings1 = [emb_dict1[sent] for sent in lists1]
            embeddings2 = [emb_dict2[sent] for sent in lists2]
        else:
            sentences = list(set(self.sentences1 + self.sentences2))
            embeddings = model.encode(
                sentences,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
            embeddings1 = [emb_dict[sent] for sent in self.sentences1]
            embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        if self.calculate_dot_product:
            cosine_scores = dot_score(embeddings1, embeddings2)
        else:
            # These scores should be between 0 and 2
            cosine_distances = paired_cosine_distances(embeddings1, embeddings2)
            # We are only interested in cosine similarity for now
            # The minimum score should be -1 and the maximum score should be 1
            cosine_scores = 1 - cosine_distances
            # manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
            # euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

            # embeddings1_np = np.asarray(embeddings1)
            # embeddings2_np = np.asarray(embeddings2)
            # dot_scores = [
            #     np.dot(embeddings1_np[i], embeddings2_np[i])
            #     for i in range(len(embeddings1_np))
            # ]

        labels = np.asarray(self.labels)
        margins = np.asarray(self.margins)
        noisy_bools = np.asarray(self.noisy_bools)

        # Calculate the loss
        # Relu function is np.maximum(0, x)
        losses = 0.5 * (
            labels * np.power(np.maximum(cosine_distances - margins, 0), 2)
            + (1 - labels) * np.power(np.maximum(margins - cosine_distances, 0), 2)
        )
        lambda_weights = np.where(
            noisy_bools,
            np.full_like(noisy_bools, self.lambda_weight),
            np.full_like(noisy_bools, 1 - self.lambda_weight),
        )
        losses = losses * lambda_weights
        loss = np.mean(losses)

        output_scores = {}
        for short_name, name, scores, reverse in [
            ["cossim", "Cosine-Similarity", cosine_scores, True],
            # ["manhattan", "Manhattan-Distance", manhattan_distances, False],
            # ["euclidean", "Euclidean-Distance", euclidean_distances, False],
            # ["dot", "Dot-Product", dot_scores, True],
        ]:
            acc, acc_threshold = self.find_best_acc_and_threshold(
                scores, labels, reverse
            )
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(
                scores, labels, reverse
            )
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info(
                "Minimum distance score with {}:      {:.2f}".format(name, min(scores))
            )

            logger.info(
                "Maximum distance score with {}:      {:.2f}".format(name, max(scores))
            )

            logger.info(
                "Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(
                    name, acc * 100, acc_threshold
                )
            )
            logger.info(
                "F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(
                    name, f1 * 100, f1_threshold
                )
            )
            logger.info(
                "Precision with {}:          {:.2f}".format(name, precision * 100)
            )
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "min_distance_score": min(scores),
                "max_distance_score": max(scores),
                "loss": loss,
            }

        return output_scores

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
