from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn.functional as F
import tqdm
from sentence_transformers import util
from sentence_transformers.SentenceTransformer import (
    __MODEL_HUB_ORGANIZATION__,
    ModelCardTemplate,
    Pooling,
    SentenceEvaluator,
    SentenceTransformer,
    Transformer,
    __version__,
    batch_to_device,
    fullname,
    snapshot_download,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)


class InputExample:
    """
    Structure for one input example with texts, the label, a unique id and a margin value
    """

    def __init__(
        self,
        guid: str = "",
        texts: List[str] = None,
        label: Union[int, float] = 0,
        margin: float = 0.5,
        noisy_bool: bool = False,
        query_id: int = -1,
        doc_id: int = -1,
    ):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        :param margin
            the margin for the example
        :param noisy_bool
            whether the example is noisy (coming from BM25) or not
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.margin = margin
        self.noisy_bool = noisy_bool
        self.query_id = query_id
        self.doc_id = doc_id

    def __str__(self):
        if type(self.texts[0]) == dict:
            return """<InputExample> label: {}, margin: {} query: {} document: {}, qid: {}, doc_id: {}""".format(
                str(self.label),
                str(self.margin),
                self.texts[0]["query"],
                self.texts[1]["doc"],
                str(self.query_id),
                str(self.doc_id),
            )
        else:
            try:
                return """<InputExample> label: {}, margin: {} query: {} document: {}, qid: {}, doc_id: {}""".format(
                    str(self.label),
                    str(self.margin),
                    self.texts[0],
                    self.texts[1],
                    str(self.query_id),
                    str(self.doc_id),
                )
            except AttributeError:
                return """<InputExample> label: {}, margin: {} query: {} document: {}""".format(
                    str(self.label),
                    str(self.margin),
                    self.texts[0],
                    self.texts[1],
                )
  
    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        if not isinstance(value, InputExample):
            return False
        return (
            self.texts[0] == value.texts[0]
            and self.texts[1] == value.texts[1]
            and self.label == value.label
            and self.margin == value.margin
        )


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class MarginContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)

    """

    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        size_average: bool = True,
        lambda_weight: float = 0.9,
        asym_dot_product: bool = False,
    ):
        super(MarginContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.model = model
        self.size_average = size_average
        self.lambda_weight = lambda_weight
        self.asym_dot_product = asym_dot_product

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {
            "distance_metric": distance_metric_name,
            "size_average": self.size_average,
        }

    def forward(
        self,
        sentence_features: Iterable[Dict[str, Tensor]],
        labels: Tensor,
        margins: Tensor,
        noisy_bools: Tensor,
        query_ids: Tensor,
        doc_ids: Tensor,
    ):
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        assert len(reps) == 2
        rep_anchor, rep_other = reps

        # If asym_model pre-trained on dot product embeddings, i.e., MedCPT model, normalize the embeddings for training stability
        if self.asym_dot_product:
            rep_anchor = F.normalize(rep_anchor, p=2, dim=1)
            rep_other = F.normalize(rep_other, p=2, dim=1)

        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * F.relu(distances - margins).pow(2)
            + (1 - labels).float() * F.relu(margins - distances).pow(2)
        )
        lambda_weights = torch.where(
            noisy_bools,
            torch.full_like(noisy_bools, self.lambda_weight),
            torch.full_like(noisy_bools, 1 - self.lambda_weight),
        )
        losses = losses * lambda_weights
        return losses.mean() if self.size_average else losses.sum()


class CustomSentenceTransformer(SentenceTransformer):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models
    :param use_auth_token: HuggingFace authentication token to download private models.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        pooling: Optional[str] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
    ):
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv(
                            "TORCH_HOME",
                            os.path.join(
                                os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"
                            ),
                        )
                    )

                cache_folder = os.path.join(torch_cache_home, "sentence_transformers")

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(
                "Load pretrained SentenceTransformer: {}".format(model_name_or_path)
            )

            # Old models that don't belong to any organization
            basic_transformer_models = [
                "albert-base-v1",
                "albert-base-v2",
                "albert-large-v1",
                "albert-large-v2",
                "albert-xlarge-v1",
                "albert-xlarge-v2",
                "albert-xxlarge-v1",
                "albert-xxlarge-v2",
                "bert-base-cased-finetuned-mrpc",
                "bert-base-cased",
                "bert-base-chinese",
                "bert-base-german-cased",
                "bert-base-german-dbmdz-cased",
                "bert-base-german-dbmdz-uncased",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "bert-base-uncased",
                "bert-large-cased-whole-word-masking-finetuned-squad",
                "bert-large-cased-whole-word-masking",
                "bert-large-cased",
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                "bert-large-uncased-whole-word-masking",
                "bert-large-uncased",
                "camembert-base",
                "ctrl",
                "distilbert-base-cased-distilled-squad",
                "distilbert-base-cased",
                "distilbert-base-german-cased",
                "distilbert-base-multilingual-cased",
                "distilbert-base-uncased-distilled-squad",
                "distilbert-base-uncased-finetuned-sst-2-english",
                "distilbert-base-uncased",
                "distilgpt2",
                "distilroberta-base",
                "gpt2-large",
                "gpt2-medium",
                "gpt2-xl",
                "gpt2",
                "openai-gpt",
                "roberta-base-openai-detector",
                "roberta-base",
                "roberta-large-mnli",
                "roberta-large-openai-detector",
                "roberta-large",
                "t5-11b",
                "t5-3b",
                "t5-base",
                "t5-large",
                "t5-small",
                "transfo-xl-wt103",
                "xlm-clm-ende-1024",
                "xlm-clm-enfr-1024",
                "xlm-mlm-100-1280",
                "xlm-mlm-17-1280",
                "xlm-mlm-en-2048",
                "xlm-mlm-ende-1024",
                "xlm-mlm-enfr-1024",
                "xlm-mlm-enro-1024",
                "xlm-mlm-tlm-xnli15-1024",
                "xlm-mlm-xnli15-1024",
                "xlm-roberta-base",
                "xlm-roberta-large-finetuned-conll02-dutch",
                "xlm-roberta-large-finetuned-conll02-spanish",
                "xlm-roberta-large-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-german",
                "xlm-roberta-large",
                "xlnet-base-cased",
                "xlnet-large-cased",
            ]

            if os.path.exists(model_name_or_path):
                # Load from path
                model_path = model_name_or_path
            else:
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if (
                    "/" not in model_name_or_path
                    and model_name_or_path.lower() not in basic_transformer_models
                ):
                    # A model from sentence-transformers
                    model_name_or_path = (
                        __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path
                    )

                model_path = os.path.join(
                    cache_folder, model_name_or_path.replace("/", "_")
                )

                if not os.path.exists(os.path.join(model_path, "modules.json")):
                    # Download from hub with caching
                    snapshot_download(
                        model_name_or_path,
                        cache_dir=cache_folder,
                        library_name="sentence-transformers",
                        library_version=__version__,
                        ignore_files=[
                            "flax_model.msgpack",
                            "rust_model.ot",
                            "tf_model.h5",
                        ],
                        use_auth_token=use_auth_token,
                    )

            if os.path.exists(
                os.path.join(model_path, "modules.json")
            ):  # Load as SentenceTransformer model
                modules = self._load_sbert_model(model_path)
            elif pooling:  # Load with AutoModel
                modules = self._load_auto_model(model_path, pooling)
            else:
                modules = self._load_auto_model(model_path)

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)]
            )

        super(SentenceTransformer, self).__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

    def _load_auto_model(self, model_name_or_path, pooling="mean"):
        """
        Creates a simple Transformer + given Pooling model and returns the modules
        """
        logger.warning(
            "No sentence-transformers model found with name {}. Creating a new one with {} pooling.".format(
                model_name_or_path, pooling
            )
        )
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(), pooling
        )
        return [transformer_model, pooling_model]

    def _eval_during_training(
        self, evaluator, output_path, save_best_model, epoch, steps, learning_rate, callback
    ):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score_list = evaluator(
                self, output_path=eval_path, epoch=epoch, steps=steps
            )
            if callback is not None:
                callback(score_list, epoch, steps, learning_rate)
            if score_list[-1][0] > self.best_score:
                self.best_score = score_list[-1][0]
                if save_best_model:
                    self.save(output_path)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []
        margins = []
        noisy_bools = []
        query_ids = []
        doc_ids = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)
            margins.append(example.margin)
            noisy_bools.append(example.noisy_bool)
            if hasattr(example, "query_id") and hasattr(example, "doc_id"):
                query_ids.append(example.query_id)
                doc_ids.append(example.doc_id)

        labels = torch.tensor(labels)
        margins = torch.tensor(margins)
        noisy_bools = torch.tensor(noisy_bools)
        query_ids = torch.tensor(query_ids)
        doc_ids = torch.tensor(doc_ids)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels, margins, noisy_bools, query_ids, doc_ids

    def fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(
                ModelCardTemplate.get_train_objective_info(dataloader, loss)
            )
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {
                "evaluator": fullname(evaluator),
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "optimizer_class": str(optimizer_class),
                "optimizer_params": optimizer_params,
                "weight_decay": weight_decay,
                "evaluation_steps": evaluation_steps,
                "max_grad_norm": max_grad_norm,
            },
            indent=4,
            sort_keys=True,
        )
        self._model_card_text = None
        self._model_card_vars["{TRAINING_SECTION}"] = (
            ModelCardTemplate.__TRAINING_SECTION__.replace(
                "{LOSS_FUNCTIONS}", info_loss_functions
            ).replace("{FIT_PARAMETERS}", info_fit_parameters)
        )

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = optimizer_class(
                optimizer_grouped_parameters, **optimizer_params
            )
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            learning_rate = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(
                steps_per_epoch,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels, margins, noisy_bools, query_ids, doc_ids = data
                    labels = labels.to(self._target_device)
                    margins = margins.to(self._target_device)
                    noisy_bools = noisy_bools.to(self._target_device)
                    query_ids = query_ids.to(self._target_device)
                    doc_ids = doc_ids.to(self._target_device)
                    features = list(
                        map(
                            lambda batch: batch_to_device(batch, self._target_device),
                            features,
                        )
                    )

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(
                                features,
                                labels,
                                margins,
                                noisy_bools,
                                query_ids,
                                doc_ids,
                            )

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(
                            features, labels, margins, noisy_bools, query_ids, doc_ids
                        )
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    # Get learning rate from first objective
                    # learning_rate = optimizers[0].param_groups[0]["lr"]
                    learning_rate = schedulers[0].get_last_lr()[0]

                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps,
                        learning_rate,
                        callback,
                    )

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(
                        checkpoint_path, checkpoint_save_total_limit, global_step
                    )

            # Get learning rate from first objective
            # learning_rate = optimizers[0].param_groups[0]["lr"]
            learning_rate = schedulers[0].get_last_lr()[0]
            self._eval_during_training(
                evaluator, output_path, save_best_model, epoch, global_step, learning_rate, callback
            )

        if (
            evaluator is None and output_path is not None
        ):  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(
                checkpoint_path, checkpoint_save_total_limit, global_step
            )

    def evaluate(
        self,
        evaluator: SentenceEvaluator,
        output_path: str = None,
        model_2: SentenceTransformer = None,
    ):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        :param model_2:
            the second model for query document similarity tasks
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, model_2, output_path)


## UPDATE 22/07/2024
## Added functions from new SBERT version
## https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CachedMultipleNegativesRankingLoss.py


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(
            sentence_features, loss_obj.cache, loss_obj.random_states
        ):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of MultipleNegativesRankingLoss (https://arxiv.org/pdf/1705.00652.pdf) by GradCache (https://arxiv.org/pdf/2101.06983.pdf).

        Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
        Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
        the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
        It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
        As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

        In detail:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
        requires a lot memory when batch size large. One drawback is about the speed. GradCache will sacrifice around 20% computation time according to the paper.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`MultipleNegativesRankingLoss`

        Relations:
            - Equivalent to :class:`MultipleNegativesRankingLoss`, but with caching that allows for much higher batch sizes
            (and thus better performance) without extra memory usage. This loss also trains roughly 2x to 2.4x slower than
            :class:`MultipleNegativesRankingLoss`.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedGISTEmbedLoss(model, mini_batch_size=64)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            k: v[begin:end] for k, v in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = (
                    RandContext(*sentence_feature_minibatch.values())
                    if copy_random_state
                    else None
                )
                reps = self.model(sentence_feature_minibatch)[
                    "sentence_embedding"
                ]  # (mbsz, hdim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def multi_label_cross_entropy_loss(self, scores: Tensor, labels: Tensor):
        log_probs = F.log_softmax(scores, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        return loss

    def create_multi_label_tensor(self, query_ids: Tensor, document_ids: Tensor):
        """Create a multi-label tensor based on query and document IDs."""
        # Create a binary mask where entries are 1 if the IDs match, 0 otherwise
        query_mask = query_ids.unsqueeze(1) == query_ids.unsqueeze(0)
        document_mask = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)

        # print(query_ids)
        # print(document_ids)
        # print(query_ids.size())
        # print(document_ids.size())

        # Combine the masks to form the multi-label tensor
        multi_label_tensor = (query_mask | document_mask).float()
        return multi_label_tensor

    def calculate_loss_and_cache_gradients(
        self, reps: list[list[Tensor]], query_ids: Tensor, document_ids: Tensor
    ) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat(
            [torch.cat(r) for r in reps[1:]]
        )  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        # Deprecated: Use multi-label tensor instead of labels
        # labels = torch.tensor(
        #     range(batch_size), dtype=torch.long, device=embeddings_a.device
        # )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]

        # Create multi-value labels instead for repeating query and document ids
        # repeated_query_ids = query_ids.repeat_interleave(
        #     1 + len(document_ids) // len(query_ids)
        # )

        # labels = self.create_multi_label_tensor(repeated_query_ids, document_ids)
        labels = self.create_multi_label_tensor(query_ids, document_ids)

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = (
                self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            )
            loss_mbatch: torch.Tensor = (
                # self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
                self.multi_label_cross_entropy_loss(scores, labels[b:e, :])
                * len(scores)
                / batch_size
            )
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses).requires_grad_()

        self.cache = [
            [r.grad for r in rs] for rs in reps
        ]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat(
            [torch.cat(r) for r in reps[1:]]
        )  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = (
                self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            )
            loss_mbatch: torch.Tensor = (
                self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            )
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
        margins: Tensor,
        noisy_bools: Tensor,
        query_ids: Tensor,
        document_ids: Tensor,
    ) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = (
            []
        )  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(
                reps, query_ids, document_ids
            )

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(
                partial(
                    _backward_hook, sentence_features=sentence_features, loss_obj=self
                )
            )
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, query_ids, document_ids)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
