import argparse
import functools
import os

import gin
import seqio
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import t5
from t5.data import preprocessors

parser = argparse.ArgumentParser()
parser.add_argument("-ms", "--model_size", type=str, required=True, help="Model Size")
args = parser.parse_args()

# Experiment specific parameters
MODEL_SIZE = args.model_size.lower()
BASE_DIR = f"gs://wikidepia/t5/"
MODELS_DIR = os.path.join(BASE_DIR, f"models/{MODEL_SIZE}-c4")
DATA_PATH = os.path.join(BASE_DIR, f"data/c4-filtered.txt")

VOCAB_PATH = os.path.join(BASE_DIR, f"sp10m.cased.t5.model")
DEFAULT_EXTRA_IDS = 100


def get_vocabulary(default=False):
    if default:
        return t5.data.get_default_vocabulary()
    return seqio.SentencePieceVocabulary(VOCAB_PATH, DEFAULT_EXTRA_IDS)


TaskRegistry = seqio.TaskRegistry
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=get_vocabulary(), add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=get_vocabulary(), add_eos=True),
}


def mc4_fn(x):
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(DATA_PATH)
    # Map each line to a {"sequence": ...} dict.
    return ds.map(lambda *ex: dict(zip(["text"], ex)))


TaskRegistry.remove("mc4_id_unsupervised")
TaskRegistry.add(
    "mc4_id_unsupervised",
    source=seqio.FileDataSource(mc4_fn, {"train": DATA_PATH}),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[],
)

# Connect to TPU
TPU_NAME = os.environ["TPU_NAME"]
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
TPU_ADDRESS = tpu.get_master()

TPU_TOPOLOGY = "2x2"
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 3),
    "base": (2, 128, 3),
    "large": (4, 64, 3),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1),
}[MODEL_SIZE]


with gin.unlock_config():
    gin.parse_config_file(
        f"gs://t5-data/pretrained_models/t5.1.1.{MODEL_SIZE}/operative_config.gin",
        skip_unknown=t5.models.mesh_transformer.DEPRECATED_GIN_REFERENCES,
    )
    gin.parse_config_file(
        "gs://wikidepia/t5/models/rsqrt_no_ramp_down.gin",
        skip_unknown=t5.models.mesh_transformer.DEPRECATED_GIN_REFERENCES,
    )

model = t5.models.MtfModel(
    model_dir=MODELS_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 1024, "targets": 1024},
    save_checkpoints_steps=5000,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=100,
)
model.train(
    mixture_or_task_name="mc4_id_unsupervised", steps=1000000, init_checkpoint=None
)
