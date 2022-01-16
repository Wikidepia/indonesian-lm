#!/usr/bin/env python3
import logging
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm
import torch
import flax
import jax
import jax.numpy as jnp
import librosa
import optax
from glob import glob
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    FlaxWav2Vec2ForPreTraining,
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    is_tensorboard_available,
)
from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices, _sample_negative_indices


logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    max_gumbel_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Maximum temperature for gumbel softmax."}
    )
    min_gumbel_temperature: Optional[float] = field(
        default=0.1, metadata={"help": "Minimum temperature for gumbel softmax."}
    )
    gumbel_temperature_decay: Optional[float] = field(
        default=0.999995, metadata={"help": "Decay of gumbel temperature during training."}
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0, metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"}
    )
    pad_to_multiple_of: Optional[int] = field(
        default=1024,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is important to avoid triggering recompilations on TPU"
        },
    )


@flax.struct.dataclass
class FlaxDataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.FlaxWav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: FlaxWav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="np",
        )
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])

        # sample randomly masked indices
        batch["mask_time_indices"] = _compute_mask_indices(
            (batch["input_values"].shape[0], mask_indices_seq_length),
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            min_masks=2,
        )

        # sample indices to take for negative vectors
        batch["sampled_negative_indices"] = _sample_negative_indices(
            (batch["mask_time_indices"].shape + (self.model.config.proj_codevector_dim,)),
            self.model.config.num_negatives,
        )

        return batch

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, speech_files, feature_extractor):
        self.speech_files = speech_files
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, index):
        speech_file = self.dataframe.iloc[index]
        speech, _ = librosa.load(speech_file, sr=self.feature_extractor.sampling_rate)
        return self.feature_extractor(speech, sampling_rate=self.feature_extractor.sampling_rate)

def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    logger.setLevel(logging_level)


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    return np.split(samples_idx, sections_split)


def compute_contrastive_loss(
    quantized_features, transformer_features, negative_indices, mask_time_indices, logits_temp, num_negatives
):
    batch_size, sequence_length, hidden_size = quantized_features.shape

    # take negative vectors from sampled indices
    quantized_negatives = quantized_features.reshape(-1, hidden_size)[negative_indices.reshape(-1)]
    quantized_negatives = quantized_negatives.reshape(
        batch_size, sequence_length, num_negatives, hidden_size
    ).transpose(2, 0, 1, 3)

    target_features = jnp.concatenate([quantized_features[None, :], quantized_negatives], axis=0)
    loss_logits = optax.cosine_similarity(transformer_features, target_features)
    loss_logits = loss_logits / logits_temp

    neg_is_pos = (quantized_features == quantized_negatives).all(-1)
    neg_is_pos = jnp.concatenate([jnp.full((1,) + loss_logits.shape[1:], False), neg_is_pos], axis=0)

    # make sure incorrectly sampled vectors don't contribute to loss
    loss_logits = jnp.where(neg_is_pos, -1e9, loss_logits)

    predictions = loss_logits.transpose(2, 1, 0).reshape(-1, loss_logits.shape[0])
    targets = ((1 - mask_time_indices) * -100).transpose(1, 0).flatten()

    target_mask = jnp.where(targets >= 0, 1.0, 0.0)
    contrastive_loss = optax.softmax_cross_entropy(predictions, onehot(targets, predictions.shape[-1])) * target_mask

    contrastive_loss = contrastive_loss.sum()

    return contrastive_loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    # only normalized-inputs-training is supported
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, do_normalize=True
    )

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=model_args.gradient_checkpointing,
    )

    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and ``config.feat_extract_norm='layer'"
        )

    model = FlaxWav2Vec2ForPreTraining(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))

    data_collator = FlaxDataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of=data_args.pad_to_multiple_of
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    gumbel_rngs = jax.random.split(rng, jax.local_device_count())

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    speech_files = glob("podcast/*.ogg")
    dataset = AudioDataset(speech_files, feature_extractor=feature_extractor)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=train_batch_size, shuffle=False, num_workers=48, pin_memory=True)
    num_train_steps = len(speech_files) // train_batch_size * num_epochs

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup train state and define training hyper-parameters
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)
    num_negatives = model.config.num_negatives
    contrastive_logits_temperature = model.config.contrastive_logits_temperature
    num_codevectors = model.config.num_codevectors_per_group * model.config.num_codevector_groups
    diversity_loss_weight = model.config.diversity_loss_weight

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng, gumbel_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        gumbel_rng, new_gumbel_rng = jax.random.split(gumbel_rng)

        def loss_fn(params):
            negative_indices = batch.pop("sampled_negative_indices")

            gumbel_temperature = jnp.clip(
                model_args.max_gumbel_temperature * model_args.gumbel_temperature_decay ** state.step,
                a_min=model_args.min_gumbel_temperature,
            )

            outputs = state.apply_fn(
                **batch,
                gumbel_temperature=gumbel_temperature,
                params=params,
                dropout_rng=dropout_rng,
                gumbel_rng=gumbel_rng,
                train=True,
            )

            contrastive_loss = compute_contrastive_loss(
                outputs.projected_quantized_states,
                outputs.projected_states,
                negative_indices,
                batch["mask_time_indices"],
                contrastive_logits_temperature,
                num_negatives,
            )

            diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors
            return contrastive_loss + diversity_loss_weight * diversity_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng, new_gumbel_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()        
        train_pbar = tqdm(len(speech_files) // train_batch_size, desc="Training...", position=1)
        for step, samples in enumerate(dataloader):
            model_inputs = data_collator(samples)
            model_inputs = shard(model_inputs.data)

            # Model forward
            state, train_metric, dropout_rngs, gumbel_rngs = p_train_step(
                state, model_inputs, dropout_rngs, gumbel_rngs
            )
            train_metrics.append(train_metric)

            cur_step = epoch * (len(speech_files) // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
                )

                train_metrics = []
            train_pbar.update()

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(training_args.output_dir, params=params, push_to_hub=training_args.push_to_hub)


if __name__ == "__main__":
    main()