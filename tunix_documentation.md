Programming Guide
Tunix provides a flexible, protocol-based logging system that allows you to integrate any logging service or library by creating a custom backend. This guide explains how to create a backend that conforms to the Metrax LoggingBackend protocol and how to use it with the Tunix MetricsLogger.

1. The Metrax LoggingBackend Protocol
Create a custom backend that conforms to the Metrax LoggingBackend protocol.

Note: You do not need to explicitly inherit from LoggingBackend. Because Metrax uses Python’s structural typing (duck typing), your class just needs to implement the required methods described below.

log_scalar(self, event: str, value: float | np.ndarray, **kwargs): This method is called whenever a metric is logged. It receives the metric name (event), its value, and optional keyword arguments (like step).

close(self): This method is called when logging is finished, allowing the backend to flush any buffered data and release resources.

from typing import Protocol
import numpy as np

class LoggingBackend(Protocol):
  """Defines the interface for a pluggable logging backend."""

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    """Logs a scalar value.

    Args:
      event: The name of the metric/event (e.g., "train/loss").
      value: The scalar value of the metric.
      **kwargs: Additional arguments, typically including 'step' (int).
    """
    ...

  def close(self):
    """Closes the logger and flushes any pending data."""
    ...
2. Creating a Custom Backend
To create a custom backend, define a class that implements these two methods.

Example: Creating a CLU Backend
Let’s create a backend for Google’s Common Loop Utils (CLU) metric writers.

from clu import metric_writers
import jax
import numpy as np

class CluBackend:
  """A custom backend for CLU metric writers."""

  def __init__(self):
    # Only initialize the writer on the main process
    if jax.process_index() == 0:
      self._writer = metric_writers.create_default_writer(logdir="custom_path")
    else:
      self._writer = None

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    # If we are not on the main process, do nothing.
    if self._writer is None:
      return

    # Extract the step, defaulting to 0 if it's not provided.
    step = int(kwargs.get("step", 0))

    # CLU's write_scalars takes a step and a dictionary of {name: value}.
    self._writer.write_scalars(step, {event: value})

  def close(self):
    if self._writer:
      self._writer.close()
Step 1: Define the Class and __init__
Initialize your specific logger. For CLU, this involves creating a MetricWriter. It’s best practice to only initialize writers on the main process (process index 0) to avoid duplicate logging in multi-process environments.

Step 2: Implement log_scalar
Map the generic log_scalar call to your logger’s specific API.

Step 3: Implement close
Ensure your logger is properly closed to flush data to disk.

3. Using Your Custom Backend
To ensure your MetricsLoggerOptions configuration is safe to copy (required for advanced workflows like RL), Tunix requires you to pass factories (callables that return a backend instance) rather than live instances.

Case A: Simple Backend (No Arguments)
If your backend class requires no arguments in its __init__, you can simply pass the class itself.

class SimplePrintBackend:
    def log_scalar(self, event, value, **kwargs):
        print(f"{event}: {value}")
    def close(self): pass

# 1. Pass the class directly. It acts as its own factory.
options = MetricsLoggerOptions(
    log_dir="/tmp/logs",
    backend_factories=[SimplePrintBackend]
)

# 2. Initialize logger. It will instantiate SimplePrintBackend() for you.
logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
logger.log("train/loss", 0.5, mode="train", step=1)
logger.close()
Case B: Backend with Arguments (Using lambda)
If your backend requires arguments (like our CluBackend needing logdir), use a lambda function to create a simple factory.

from tunix.sft import metrics_logger

# 1. Create a factory using a lambda.
#    This defers the creation of the backend until the logger is initialized.
my_clu_factory = lambda: CluBackend(logdir="/tmp/my_experiment_logs")

# 2. Create options and include your factory in the 'backends' list.
options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/default_logs",
    backend_factories=[my_clu_factory]
)

# 3. Initialize the logger. It will call your factory to create the live backend.
logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
logger.log("train/loss", 0.5, mode="train", step=1)
logger.close()




Knowledge Distillation with Tunix: Gemma 7B to Gemma 2B
Run in Kaggle	View source on GitHub
Install necessary libraries
This notebook demonstrates how to use the Tunix library to perform knowledge distillation. Specifically, we will use logit-based distillation to transfer knowledge from a larger, more capable teacher model (Gemma 7B) to a smaller, more efficient student model (Gemma 2B).

What is Knowledge Distillation?
Knowledge distillation is a model compression technique where a smaller “student” model is trained to mimic the behavior of a larger, pre-trained “teacher” model. Instead of training the student solely on the ground-truth labels, we also train it to replicate the teacher’s outputs.

Logit-Based Distillation
In this specific strategy, the student model learns to match the teacher’s logits (the raw, unnormalized outputs before the final softmax layer). By doing so, the student learns the nuanced probability distribution that the teacher model has learned, which is often more informative than the hard labels alone.

The core components we’ll use are:

Teacher Model: Gemma 7B

Student Model: Gemma 2B

Distillation Strategy: tunix.distillation.strategies.LogitStrategy

Trainer: tunix.distillation.DistillationTrainer

In this tutorial, we use a v5e-8 TPU. Let’s get started!

!pip install -q kagglehub

!pip install -q tensorflow
!pip install -q tensorboardX
!pip install -q grain-nightly
!pip install -q datasets
!pip install -q git+https://github.com/google/tunix
!pip install -q git+https://github.com/google/qwix

!pip uninstall -q -y flax
!pip install -q git+https://github.com/google/flax.git
import gc
import os

from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from tunix.distillation import distillation_trainer
from tunix.distillation import strategies
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.examples.data import translation_dataset as data_lib
Utility Function to check HBM
import functools
import humanize
from tunix.sft import utils

show_hbm_usage = utils.show_hbm_usage
show_hbm_usage()
# --- Data ---
BATCH_SIZE = 4
MAX_TARGET_LENGTH = 128
NUM_TRAIN_EPOCHS = 1

# --- Model ---
MESH = [(1, 8), ("fsdp", "tp")]

# --- Training ---
MAX_STEPS = 200
EVAL_EVERY_N_STEPS = 50
LEARNING_RATE = 1e-4

# --- Distillation ---
TEMPERATURE = 2.0  # Softens the teacher's probabilities
ALPHA = 0.7  # Balances distillation loss and student's own task loss

# --- Checkpointing ---
TEACHER_CKPT_DIR = "/content/intermediate_ckpt/teacher/"
STUDENT_CKPT_DIR = "/content/intermediate_ckpt/student/"
First, we need to load our teacher and student models. We’ll use Gemma 7B as the teacher and Gemma 2B as the student.

Important: You must have a Kaggle account and agree to the Gemma license to download the models. The first time you run this, you will be prompted to log in to Kaggle.

# Log in to Kaggle
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  kagglehub.login()


def load_and_save_model(model_handle, version, ckpt_dir):
  """Loads a model from Kaggle, saves it locally, and cleans up memory."""
  print(f"Loading {model_handle}...")
  kaggle_ckpt_path = kagglehub.model_download(model_handle)
  ckpt_version = "2b-it"
  if "7b" in version:
    ckpt_version = "7b-it"
  # Temporarily set the default device to CPU for loading the full model
  with jax.default_device(jax.devices("cpu")[0]):
    params = params_lib.load_and_format_params(
        os.path.join(kaggle_ckpt_path, ckpt_version)
    )
    gemma = gemma_lib.Transformer.from_params(params, version=version)

  print(f"Saving checkpoint to {ckpt_dir}...")
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(gemma)
  checkpointer.save(os.path.join(ckpt_dir, "state"), state)
  checkpointer.wait_until_finished()
  # Clean up to save memory
  del params
  del gemma
  del state
  gc.collect()
  print(f"Finished processing {model_handle}.")


# Load Teacher Model (Gemma 7B)
load_and_save_model(
    "google/gemma/flax/1.1-7b-it", "1.1-7b-it", TEACHER_CKPT_DIR
)

# Load Student Model (Gemma 2B)
load_and_save_model(
    "google/gemma/flax/1.1-2b-it", "1.1-2b-it", STUDENT_CKPT_DIR
)
Now that we have the checkpoints saved locally, we can load them into sharded models. Sharding is essential for training large models efficiently on TPUs by distributing the model’s weights and the computation across multiple devices.

def get_sharded_model(ckpt_path, model_config, mesh):
  """Loads a checkpoint into a sharded model."""
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma


mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

# Create Teacher Model
print("Creating sharded teacher model (Gemma 7B)...")
teacher_config = gemma_lib.ModelConfig.gemma_7b()
teacher_model = get_sharded_model(
    os.path.join(TEACHER_CKPT_DIR, "state"), teacher_config, mesh
)
print("Teacher model created.")
# nnx.display(teacher_model) # Optional: view model structure

# Create Student Model
print("\nCreating sharded student model (Gemma 2B)...")
student_config = gemma_lib.ModelConfig.gemma_2b()
student_model = get_sharded_model(
    os.path.join(STUDENT_CKPT_DIR, "state"), student_config, mesh
)
print("Student model created.")
# nnx.display(student_model) # Optional: view model structure

show_hbm_usage()
print("Loading tokenizer...")
gemma_tokenizer_path = os.path.join(
    kagglehub.model_download("google/gemma/flax/1.1-2b-it"), "tokenizer.model"
)
gemma_tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_type='sentencepiece', 
    tokenizer_path=gemma_tokenizer_path
)
print("Tokenizer loaded.")

print("\nCreating datasets...")
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name="mtnt/en-fr",
    global_batch_size=BATCH_SIZE,
    max_target_length=MAX_TARGET_LENGTH,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    tokenizer=gemma_tokenizer,
    instruct_tuned=True,
)
print("Datasets created.")
The LogitStrategy requires three key functions:

model_forward_fn: A function that performs a forward pass for a given model and returns its logits. Since both our models are from the Gemma family, we can use the same function for both.

labels_fn: A function that creates the ground-truth labels from the input data for the standard cross-entropy loss.

gen_model_input_fn: A helper function to format each batch from the data loader into the dictionary format expected by the model.

VOCAB_SIZE = student_config.num_embed


def model_forward_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
):
  """Performs a forward pass and returns the logits."""
  logits, _ = model(
      input_tokens,
      positions,
      None,
      attention_mask,
  )
  # Exclude the last step as it does not appear in the targets.
  return logits[:, :-1, :]


def labels_fn(
    input_tokens: jax.Array,
    input_mask: jax.Array,
    **kwargs,
):
  """Creates one-hot encoded labels for the next-token prediction task."""
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]
  labels = jax.nn.one_hot(target_tokens, VOCAB_SIZE)
  # Mask out the padding tokens from the loss calculation.
  return labels * target_mask.astype(labels.dtype)[..., None]


def gen_model_input_fn(x: distillation_trainer.TrainingInput):
  """Formats a batch from the data loader into the model's expected input format."""
  pad_mask = x.input_tokens != gemma_tokenizer.pad_id()
  positions = utils.build_positions_from_mask(pad_mask)
  attention_mask = utils.make_causal_attn_mask(pad_mask)
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': positions,
      'attention_mask': attention_mask,
  }
Now we can assemble all the components. We’ll instantiate the LogitStrategy, configure the DistillationTrainer, and start the training process. The trainer will handle the distributed training loop across the available TPU cores.

# 1. Setup the distillation strategy
strategy = strategies.LogitStrategy(
    student_forward_fn=model_forward_fn,
    teacher_forward_fn=model_forward_fn,
    labels_fn=labels_fn,
    temperature=TEMPERATURE,
    alpha=ALPHA,
)

# 2. Setup the training configuration
config = distillation_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
)

# 3. Setup the optimizer
optimizer = optax.adamw(LEARNING_RATE)


# Set teacher model in eval mode
teacher_model.eval()
# Set student model in train mode
student_model.train()
# 4. Instantiate the trainer
trainer = distillation_trainer.DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    strategy=strategy,
    optimizer=optimizer,
    training_config=config,
).with_gen_model_input_fn(gen_model_input_fn)

# 5. Run training within the mesh context, the first couple of training step might take up to 5 minutes to finish. Please be patient. If you experience long training steps, e.g. >10 minutes per, please open a bug. Really appreciated!
print("Starting distillation training...")
with mesh:
  trainer.train(train_ds, validation_ds)
print("Training complete.")
After training, the student model should have improved its ability to perform the translation task by learning from the teacher. Let’s test it with a few sample prompts.

print("Setting up sampler for evaluation...")
sampler = sampler_lib.Sampler(
    transformer=student_model,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_TARGET_LENGTH + 64,
        num_layers=student_config.num_layers,
        num_kv_heads=student_config.num_kv_heads,
        head_dim=student_config.head_dim,
    ),
)
input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
    "Translate this into French:\nI am a student.\n",
]

print("Generating translations with the distilled student model...")
with mesh:
  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=20,
  )

print("\n--- Evaluation Results ---")
for input_string, out_string in zip(input_batch, out_data.text):
  print(f"----------------------")
  print(f"Prompt:\n{input_string}")
  print(f"Distilled Student's Output:\n{out_string}")



uning
Fine-tuning examples using Google Tunix.

Notebooks
The following notebooks provide comprehensive examples of different fine-tuning techniques:

qlora_gemma.ipynb - LoRA and QLoRA fine-tuning with Gemma models. Demonstrates parameter-efficient fine-tuning techniques using low-rank adaptation.

grpo_gemma.ipynb - GRPO (Group Relative Policy Optimization) reinforcement learning. Shows how to fine-tune models using policy optimization for improved response generation.

dpo_gemma.ipynb - DPO (Direct Preference Optimization). Demonstrates preference-based fine-tuning to align model outputs with desired behaviors.

logit_distillation.ipynb - Knowledge distillation from larger models. Shows how to transfer knowledge from a teacher model to a student model.

Subdirectories
deepscaler/
Contains scripts for training and evaluating models with DeepScaler:

train_deepscaler_nb.py - Training script for DeepScaler models

math_eval_nb.py - Mathematical reasoning evaluation utilities

model_load/
Examples for loading models from different formats:

from_safetensor_load/ - Contains notebooks for loading Gemma2 and Gemma3 models from safetensors format

gemma2_model_load.ipynb

gemma3_model_load.ipynb

rl/
Reinforcement learning examples and hardware resource requirements:

grpo/gsm8k/ - GRPO implementation scripts for GSM8K mathematical reasoning tasks

Launch scripts for various models (Gemma 7b, Gemma2 2b, Llama3.2 1b/8b)

README.md - Detailed hardware resource requirements and configuration recommendations for RL training

sft/
Supervised fine-tuning examples:

mtnt/ - MTNT translation task examples with launch scripts for multiple models

Launch scripts for Gemma 2b, Gemma2 2b, Gemma3 4b, Llama3.2 3b, Qwen2.5 0.5b

README.md - Hardware resource requirements for SFT training

GCE VM Setup for Fine-Tuning
1. Create TPU VM
Create a v5litepod-8 TPU VM in GCE:

SW version: v2-alpha-tpuv5-lite

Name: v5-8

Reference: TPU Runtime Versions

2. Configure VM
SSH into the VM using the supplied gcloud command, then run:

# Create .env file with required credentials
vim .env

# Download and install Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh  # always input "yes"/enter
source ~/.bashrc

# Create conda environment (Python 3.12 - MUST BE 12, NOT 11!)
conda create -n colab python=3.12 -y
conda activate colab

# Install dependencies
pip install 'ipykernel<7' jupyterlab
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu
Reference: Run JAX on TPU

Exit the SSH session after setup is complete.

3. Connect from Local Machine
From your local machine, run the following to connect to Jupyter Lab:

gcloud compute tpus tpu-vm ssh v5-8 --zone=us-west1-c \
  -- -L 8080:localhost:8080 -L 6006:localhost:6006 \
  "source \$HOME/anaconda3/etc/profile.d/conda.sh && \
   conda activate colab && \
   jupyter lab \
     --ServerApp.allow_origin='https://colab.research.google.com' \
     --port=8080 \
     --no-browser \
     --ServerApp.port_retries=0 \
     --ServerApp.allow_credentials=True"
Reference: Local Runtimes in Colab

4. Environment Variables
Example .env file:

HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
Loading Saved Safetensors Models
To load a saved safetensors model back into JAX (with a given local_path):

import os
import jax
import jax.numpy as jnp
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib


local_path = '[PLACEHOLDER]'
MESH = [(1, 1), ("fsdp", "tp")]

mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
with mesh:
  model = params_safetensors_lib.create_model_from_safe_tensors(
      os.path.abspath(local_path), (model_config), mesh, dtype=jnp.bfloat16
  )


Supervised fine-tuning (SFT)
PeftTrainer(model, optimizer, training_config)

PEFT trainer for LoRA.

TrainingConfig(*, eval_every_n_steps[, ...])

Configuration for the trainer.

DPOTrainer(model, ref_model, optimizer, ...)

Direct Preference Optimization (DPO) and ORPO trainer.

DPOTrainingConfig(*, eval_every_n_steps[, ...])

DPO/ORPO Training Config.

MetricsLogger([metrics_logger_options])

Simple Metrics logger.

MetricsLoggerOptions(log_dir[, ...])

Metrics Logger options.

class tunix.PeftTrainer(model: Module, optimizer: GradientTransformation, training_config: TrainingConfig, metrics_logger: MetricsLogger | None = None)
PEFT trainer for LoRA. Only LoRA parameters are updated.

model
The model to train.

config
The training config.

optimizer
The optimizer to use. To monitor the learning rate at each step, use optax.schedules.inject_hyperparams to inject learning rate as a hyperparameter. For example: optimizer = optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)

loss_fn
The loss function to use.

eval_loss_fn
The loss function to use for evaluation.

gen_model_input_fn
The function to generate model input from training input.

checkpoint_manager
The checkpoint manager to use.

metrics_logger
The metrics logger to use.

is_managed_externally
Whether the trainer is managed externally.

training_hooks
The training hooks to use.

data_hooks
The data hooks to use.

clear_jit_cache()
Clears the JIT cache of the train and eval step functions.

This function should be called when the trainer is being reused after overiding the training related states, for example, the loss function.

close()
Closes the trainer and its associated resources.

This includes writing any buffered metrics, saving the last checkpoint, and closing the checkpoint manager and metrics logger.

create_eval_step_fn() → Callable[[...], Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]
Creates the eval step function.

create_train_step_fn() → Callable[[...], Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]
Creates the train step function.

custom_checkpoint_metadata() → dict[str, Any]
Override this function to return the custom metadata for the checkpoint manager.

property iter_steps: int
Returns the number of iterator steps taken.

jit_train_and_eval_step(skip_jit: bool = False)
Creates and returns the train and eval step functions.

This function will return the cached ones if available.

Parameters
:
skip_jit – If True, the train and eval step functions will not be JITed.

Returns
:
A tuple of train and eval step functions.

train(train_ds: Iterable[Any], eval_ds: Iterable[Any] | None = None, skip_jit: bool = False) → None
Training loop.

property train_steps: int
Returns the number of train steps taken.

with_data_hooks(data_hooks: DataHooks)
with_gen_model_input_fn(gen_model_input_fn: Callable[[Any], Dict[str, Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]])
Generates model input from training input.

NB: output of this function will be passed to the loss function, so the args should match what loss function expects.

Parameters
:
gen_model_input_fn – A function that generates model input from training input.

Returns
:
PeftTrainer.

with_loss_fn(loss_fn: Callable[[Concatenate[Module, P]], Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray, Any]], has_aux: bool = False)
with_training_hooks(training_hooks: TrainingHooks)
class tunix.TrainingConfig(*, eval_every_n_steps: int, max_steps: int | None = None, gradient_accumulation_steps: int | None = None, checkpoint_root_directory: str | None = None, checkpointing_options: CheckpointManagerOptions | None = None, metrics_logging_options: MetricsLoggerOptions | None = None, profiler_options: ProfilerOptions | None = None, data_sharding_axis: Tuple[str, ...] = ('fsdp',), max_inflight_computations: int = 2, metrics_prefix: str = '', pbar_description: str | None = 'Training')
Configuration for the trainer.

checkpoint_root_directory: str | None
checkpointing_options: CheckpointManagerOptions | None
data_sharding_axis: Tuple[str, ...]
eval_every_n_steps: int
get_with_default(key: str, default: Any) → Any
gradient_accumulation_steps: int | None
max_inflight_computations: int
max_steps: int | None
metrics_logging_options: MetricsLoggerOptions | None
metrics_prefix: str
pbar_description: str | None
profiler_options: ProfilerOptions | None
class tunix.DPOTrainer(model: Module, ref_model: Module | None, optimizer: GradientTransformation, training_config: DPOTrainingConfig, tokenizer: Any | None = None)
Direct Preference Optimization (DPO) and ORPO trainer.

DPO is a preference tuning method for aligning large language models with human or AI preferences. It is a more efficient, performant alternative to RLHF.

DPO is simpler because it eliminates the need for text generation in the training loop. Moreover, DPO bypasses the reward modeling step entirely, i.e., we do not need to train a separate reward model. It uses a dataset of preferences (pairs of “chosen” and “rejected responses) to directly optimize the policy model by using a classification-style loss.

ORPO (Odds Ratio Preference Optimization) is a memory-efficient variant that combines supervised fine-tuning with preference alignment without requiring a separate reference model, making it approximately 50% more memory-efficient.

References: - DPO: https://arxiv.org/abs/2305.18290 - ORPO: https://arxiv.org/abs/2403.07691

class tunix.DPOTrainingConfig(*, eval_every_n_steps: int, max_steps: int | None = None, gradient_accumulation_steps: int | None = None, checkpoint_root_directory: str | None = None, checkpointing_options: CheckpointManagerOptions | None = None, metrics_logging_options: MetricsLoggerOptions | None = None, profiler_options: ProfilerOptions | None = None, data_sharding_axis: Tuple[str, ...] = ('fsdp',), max_inflight_computations: int = 2, metrics_prefix: str = '', pbar_description: str | None = 'Training', algorithm: str = 'dpo', beta: float = 0.1, lambda_orpo: float = 0.1, label_smoothing: float = 0.0, max_prompt_length: int | None = None, max_response_length: int | None = None)
DPO/ORPO Training Config.

algorithm: str
beta: float
label_smoothing: float
lambda_orpo: float
max_prompt_length: int | None
max_response_length: int | None
class tunix.MetricsLogger(metrics_logger_options: MetricsLoggerOptions | None = None)
Simple Metrics logger.

close()
Closes all registered logging backends.

get_metric(metrics_prefix, metric_name: str, mode: Mode | str)
Returns the mean metric value for the given metric name and mode.

get_metric_history(metrics_prefix, metric_name: str, mode: Mode | str)
Returns all past metric values for the given metric name and mode.

log(metrics_prefix: str, metric_name: str, scalar_value: float | ndarray, mode: Mode | str, step: int)
Logs the scalar metric value to local history and via jax.monitoring.

metric_exists(metrics_prefix, metric_name: str, mode: Mode | str) → bool
Checks if the metric exists for the given metric name and mode.

class tunix.MetricsLoggerOptions(log_dir: str, flush_every_n_steps: int = 100, backend_factories: list[Callable[[], LoggingBackend]] | None = None)
Metrics Logger options.

backend_factories: list[Callable[[], LoggingBackend]] | None = None
create_backends() → list[LoggingBackend]
Factory method to create a fresh set of live backends.

flush_every_n_steps: int = 100
log_dir: str


Reinforcement learning (RL)
GRPOConfig(*[, algo_variant, ...])

Configuration for GRPO algorithms.

GRPOLearner(rl_cluster, algo_config, reward_fns)

GRPO (Group Relative Policy Optimization) learner.

RewardFn

alias of Callable[[...], List[float]]

PPOConfig(*[, algo_variant, ...])

Configuration for PPO learner.

PPOLearner(rl_cluster, ppo_config[, ...])

PPO (Proximal Policy Optimization) learner.

ClusterConfig(*, role_to_mesh[, ...])

Cluster config.

RLCluster(*, actor[, critic, reference, ...])

RLCluster.

RLTrainingConfig(*, eval_every_n_steps[, ...])

RLTraining config.

Role(*values)

Role of the model.

RolloutConfig([max_tokens_to_generate, ...])

Configuration for the rollout worker.

class tunix.GRPOConfig(*, algo_variant: str = 'grpo', advantage_estimator: str = 'grpo', policy_loss_fn: str = 'grpo', loss_agg_mode: str = 'sequence-mean-token-mean', loss_algo: str = 'grpo', num_generations: int = 2, num_iterations: int = 1, beta: float = 0.04, epsilon: float = 0.2)
Configuration for GRPO algorithms.

algo_variant
The core algorithm variant to use.

Type
:
str

advantage_estimator
The advantage estimator to use.

Type
:
str

policy_loss_fn
The policy loss function to use.

Type
:
str

loss_agg_mode
The aggregation mode for the loss function.

Type
:
str

loss_algo
The loss algorithm to use. To be deprecated.

Type
:
str

num_generations
The number of times the policy generates multiple responses for a given prompt within a single training step. This corresponds to ‘G’ in Algorithm 1 in the paper. A higher value means more samples are used to compute relative advantages.

Type
:
int

num_iterations
The number of iterations per batch (𝜇 in GRPO algo 1).

Type
:
int

beta
The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function. This term prevents policy updates from deviating too far from the reference model. A value of 0.0 means no KL penalty is applied.

Type
:
float

epsilon
Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, it ensures stable updates.

Type
:
float

epsilon_high
Epsilon value for upper bound clipping.

loss_algo
use GRPO or GSPO for loss computation. GRPO loss is per-batch normalized instead of per-response normalized as mentioned in the paper. For GSPO, we use gspo-token loss which is more flexible.

Type
:
str

References
GRPO: https://arxiv.org/abs/2402.03300 - GSPO:

https://www.arxiv.org/pdf/2507.18071

beta: float
epsilon: float
loss_agg_mode: str
loss_algo: str
num_generations: int
num_iterations: int
class tunix.GRPOLearner(rl_cluster: RLCluster, algo_config: TGrpoConfig, reward_fns: Callable[[...], List[float]] | List[Callable[[...], List[float]]], metric_fns: Sequence[Callable[[...], Dict[str, Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | str, Callable[[Array], Array] | None]]]] | None = None, data_shuffle_seed: int | None = None)
GRPO (Group Relative Policy Optimization) learner.

GRPO is a reinforcement learning algorithm designed to enhance the reasoning abilities of large language models, like mathematical problem-solving. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group’s performance to update the policy.

References

https://arxiv.org/abs/2402.03300

train(train_ds: Iterable[Dict[str, List[str] | Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]], eval_ds: Iterable[Dict[str, List[str] | Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]] | None = None, skip_jit: bool = False) → None
GRPO training loop.

Algorithm as below: extract from https://arxiv.org/abs/2402.03300

Input:
    initial policy model πθinit;
    reward models rφ;
    task prompts D;
    hyperparameters ε, β, μ

policy model πθ ← πθinit
for iteration = 1, ..., I do
  reference model πref ← πθ
  for step = 1, ..., M do
    Sample a batch D♭ from D
    Update the old policy model πθold ← πθ
    Sample G outputs {oi}G_i=1 ~ πθold(· | q) for each question q ∈ D♭
    Compute rewards {ri}G_i=1 for each sampled output oi by running rφ
    Compute Âi,t for the t-th token of oi through group relative
    advantage estimation.
    for GRPO iteration = 1, ..., μ do
      Update the policy model πθ by maximizing the GRPO objective
      (Equation 21)
  Update rφ through continuous training using a replay mechanism.
Output πθ
Note

The outer loop (I) is ignored for now because we never update the reference model for now.

Currently sample and train hold the same referece to the model. So we also omit the step to update the sampler model.

Parameters
:
train_ds – An iterable of training input data, where each element is a dictionary containing the key ‘prompts’.

eval_ds – An iterable of evaluation input data, where each element is a dictionary containing the key ‘prompts’.

skip_jit – Whether to skip JIT compilation of the training loop.

tunix.RewardFn
alias of Callable[[…], List[float]]

class tunix.PPOConfig(*, algo_variant: str = 'ppo', advantage_estimator: str = 'gae', policy_loss_fn: str = 'ppo', num_iterations: int = 1, gamma: float = 1.0, gae_lambda: float = 0.95, beta: float = 0.04, epsilon: float = 0.2, epsilon_low: float | None = None, epsilon_high: float | None = None, epsilon_c: float | None = None, entropy_coef: float | None = None, clip_range_value: float = 0.2, kl_method: str = 'low_var_kl')
Configuration for PPO learner.

num_iterations
The number of optimization epochs per batch of rollouts. This corresponds to the number of times the policy updates its weights for a given batch of rollouts.

Type
:
int

mini_batch_size
The batch size on which the actual model updates happen. The rollout phase (generate_and_compute_advantages) happen on a larger batch, which is then split into “mini-batches”.

gamma
The discount factor for future rewards in GAE.

Type
:
float

gae_lambda
The lambda parameter for Generalized Advantage Estimation (GAE).

Type
:
float

beta
The coefficient for the KL divergence penalty.

Type
:
float

epsilon
Epsilon value for clipping the ratio for the policy objective.

Type
:
float

epsilon_low
Lower bound for clipping the ratio for the policy objective. Set to epsilon if not provided.

Type
:
float | None

epsilon_high
Upper bound for clipping the ratio for the policy objective. Set to epsilon if not provided.

Type
:
float | None

epsilon_c
Lower bound for clipping for dual-clip PPO. If not provided, we don’t do dual-clip PPO. Reference: https://arxiv.org/abs/1912.09729.

Type
:
float | None

entropy_coef
Entropy coefficient for the policy loss. Set to None or 0.0 to disable entropy regularization.

Type
:
float | None

clip_range_value
The range for clipping the value function loss.

Type
:
float

kl_method
The method for computing KL divergence. Must be one of ["low_var_kl", "kl", "mse_kl"].

Type
:
str

beta: float
clip_range_value: float
entropy_coef: float | None
epsilon: float
epsilon_c: float | None
epsilon_high: float | None
epsilon_low: float | None
gae_lambda: float
gamma: float
kl_method: str
num_iterations: int
class tunix.PPOLearner(rl_cluster: RLCluster, ppo_config: PPOConfig, reward_fns: Callable[[...], List[float]] | List[Callable[[...], List[float]]] | None = None, metric_fns: Sequence[Callable[[...], Dict[str, Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | str, Callable[[Array], Array] | None]]]] | None = None, data_shuffle_seed: int | None = None)
PPO (Proximal Policy Optimization) learner.

PPO is a reinforcement learning algorithm that fine-tunes models using an actor-critic architecture. It optimizes a clipped surrogate objective function to ensure stable policy updates, preventing large, destructive changes. The actor (policy model) learns what actions to take, while the critic (value model) estimates the value of states to help calculate advantages. This approach balances exploration and exploitation, making it a robust choice for a wide range of RL tasks.

References: - https://arxiv.org/abs/1707.06347

train(train_ds: Iterable[Dict[str, List[str] | Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]], eval_ds: Iterable[Dict[str, List[str] | Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]] | None = None, skip_jit: bool = False) → None
PPO training loop.

class tunix.ClusterConfig(*, role_to_mesh: dict[Role, Mesh], rollout_engine: str | type[BaseRollout] = 'vanilla', offload_to_cpu: bool = False, training_config: RLTrainingConfig, rollout_config: dict[Mode, RolloutConfig] | RolloutConfig)
Cluster config.

role_to_mesh
Mapping from model role to mesh. Key config for colocated vs disaggregated setup.

Type
:
dict[tunix.rl.rl_cluster.Role, jax._src.mesh.Mesh]

rollout_engine
Rollout engine to use. E.g. “vanilla”, “vllm”, “sglang_jax”. Alternatively, if a subclass of base_rollout.BaseRollout is provided, it will be used as the rollout engine.

Type
:
str | type[tunix.rl.rollout.base_rollout.BaseRollout]

offload_to_cpu
Whether to offload models to CPU at each step..

Type
:
bool

training_config
RL training config.

Type
:
tunix.rl.rl_cluster.RLTrainingConfig

rollout_config
Rollout config. It may be different for different modes, e.g. TRAIN vs EVAL.

Type
:
dict[tunix.rl.rl_cluster.Mode, tunix.rl.rollout.base_rollout.RolloutConfig] | tunix.rl.rollout.base_rollout.RolloutConfig

rollout_vllm_model_version
Model version for vllm rollout engine.

rollout_vllm_lora_config
LoRA config for vllm rollout engine.

rollout_vllm_hbm_utilization
The percentage of TPU/GPU HBM allocated the vllm rollout engine.

rollout_vllm_init_with_random_weights
Init the vllm TPU backend model with random weights instead of loading from the given path.

rollout_vllm_tpu_backend_type
The TPU Jax backend type for vllm rollout engine, E.g. “jax”, “torchax” or “pytorch_xla”.

rollout_vllm_swap_space_size_gb
The swap space size (in GiB) for vllm rollout engine. This is the amount of CPU memory (RAM) to allocate for swapping KV cache blocks from the TPU/GPU memory (HBM). A larger value allows for larger batch sizes and longer sequences, potentially at the cost of increased latency if swapping occurs.

offload_to_cpu: bool = False
role_to_mesh: dict[Role, Mesh]
rollout_config: dict[Mode, RolloutConfig] | RolloutConfig
rollout_engine: str | type[BaseRollout] = 'vanilla'
training_config: RLTrainingConfig
class tunix.RLCluster(*, actor: Module | str, critic: Module | str | None = None, reference: Module | str | None = None, reward: Module | str | None = None, tokenizer: Any | None, cluster_config: ClusterConfig, perf_config: PerfMetricsConfig | None = None)
RLCluster.

property actor_trainer: Trainer
buffer_metrics(metrics: Dict[str, Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | str, Callable[[Array], Array] | None]], mode: Mode = Mode.TRAIN) → None
Buffers rl metrics to be logged.

Actual logging will happen when global steps are incremented.

Parameters
:
metrics – A dictionary mapping metric names to a tuple containing the metric value and an optional aggregation function.

mode – The mode of the workload, either TRAIN or EVAL.

buffer_metrics_async(metrics: Dict[str, Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | str, Callable[[Array], Array] | None]], mode: Mode = Mode.TRAIN, step: int = 0) → None
Buffers rl metrics to be logged for async training.

Actual logging will happen when global steps are incremented.

Parameters
:
metrics – A dictionary mapping metric names to a tuple containing the metric value and an optional aggregation function.

mode – The mode of the workload, either TRAIN or EVAL.

step – The step number for the metrics. Only used in TRAIN mode.

close()
property critic_trainer: Trainer
generate(prompts: list[str] | list[list[dict[str, str]]], apply_chat_template: bool = False, mode: Mode = Mode.TRAIN, micro_batch_size: int | None = None) → RolloutOutput
Generates text from the given prompts.

Parameters
:
prompts – A list of prompts to generate text from. If apply_chat_template is True, this should be a list of conversations (each a list of dictionaries with ‘role’ and ‘content’). Otherwise, it should be a list of strings.

apply_chat_template – Whether to apply chat template to the prompts.

mode – The mode of rollout, either TRAIN or EVAL.

micro_batch_size – The micro-batch size for generation. If None, no micro-batching is performed.

Returns
:
A RolloutOutput object containing the generated text and other info.

get_old_per_token_logps(prompt_tokens: Array, completion_tokens: Array, micro_batch_size: int | None = None, completion_mask: Array | None = None) → Array
Gets the per-token logps of the current policy model.

get_ref_per_token_logps(prompt_tokens: Array, completion_tokens: Array, pad_id: int, eos_id: int, micro_batch_size: int | None = None, completion_mask: Array | None = None) → Array
Gets the per-token logps of the reference model.

get_rewards(prompt_tokens: Array, completion_tokens: Array, pad_id: int, eos_id: int) → Array
get_values(prompt_tokens: Array, completion_tokens: Array, pad_id: int, eos_id: int, completion_mask: Array | None = None) → Array
property inference_worker: InferenceWorker
property perf: PerfTracer | NoopTracer
property rollout: BaseRollout
sync_weights()
Syncs the weights of between the sampler model and trainer model.

update_actor(train_ds, eval_ds, skip_jit=False)
update_critic(train_ds, eval_ds, skip_jit=False)
with_external_metrics_logger(external_metrics_logger: Callable[[MetricsBuffer], None])
class tunix.RLTrainingConfig(*, eval_every_n_steps: int, max_steps: int | None = None, gradient_accumulation_steps: int | None = None, checkpoint_root_directory: str | None = None, checkpointing_options: CheckpointManagerOptions | None = None, metrics_logging_options: MetricsLoggerOptions | None = None, profiler_options: ProfilerOptions | None = None, data_sharding_axis: Tuple[str, ...] = ('fsdp',), max_inflight_computations: int = 2, metrics_prefix: str = '', pbar_description: str | None = 'Training', actor_optimizer: GradientTransformation, critic_optimizer: GradientTransformation | None = None, mini_batch_size: int | None = None, train_micro_batch_size: int | None = None, rollout_micro_batch_size: int | None = None, compute_logps_micro_batch_size: int | None = None)
RLTraining config.

actor_optimizer
Optimizer for the actor model.

Type
:
optax._src.base.GradientTransformation

critic_optimizer
Optimizer for the critic model. If None, the critic model will be trained in the same optimizer as the actor model.

Type
:
optax._src.base.GradientTransformation | None

mini_batch_size
The mini-batch size used for policy weight updates. One mini-batch corresponds to one optimizer update. mini_batch_size must be divisible by the global batch size.

Type
:
int | None

train_micro_batch_size
The micro-batch size used for gradient accumulation at training time. train_micro_batch_size must be divisible by mini_batch_size.

Type
:
int | None

rollout_micro_batch_size
The micro-batch size used for model rollouts.

Type
:
int | None

compute_logps_micro_batch_size
The micro-batch size used for computing log probabilities (e.g. for reference and old policy models).

Type
:
int | None

actor_optimizer: GradientTransformation
compute_logps_micro_batch_size: int | None
critic_optimizer: GradientTransformation | None
mini_batch_size: int | None
rollout_micro_batch_size: int | None
train_micro_batch_size: int | None
class tunix.Role(*values)
Role of the model.

ACTOR = 'actor'
CRITIC = 'critic'
REFERENCE = 'reference'
REWARD = 'reward'
ROLLOUT = 'rollout'
class tunix.RolloutConfig(max_tokens_to_generate: int = 64, temperature: float = 0.9, top_p: float | None = 1.0, top_k: int | None = None, seed: Array | None = None, max_prompt_length: int = 64, kv_cache_size: int = 1024, data_type: dtype | None = None, eos_tokens: list[int] | None = None, rollout_mapping_config: MappingConfig | None = None, tensor_parallel_size: int = -1, data_parallel_size: int = -1, rollout_vllm_server_mode: bool = False, rollout_vllm_model_version: str = '', rollout_vllm_lora_config: dict[str, Any] | None = None, rollout_vllm_hbm_utilization: float = 0.2, rollout_vllm_init_with_random_weights: bool = True, rollout_vllm_tpu_backend_type: str | None = None, rollout_vllm_swap_space_size_gb: float = 4.0, rollout_vllm_async_scheduling: bool = False, rollout_sglang_jax_model_version: str = '', rollout_sglang_jax_context_length: int = 8192, rollout_sglang_jax_mem_fraction_static: float = 0.2, rollout_sglang_jax_init_with_random_weights: bool = True, rollout_sglang_jax_disable_radix_cache: bool = True, rollout_sglang_jax_enable_deterministic_sampling: bool = False)
Configuration for the rollout worker.

Fields should be mapped to a subset of vLLM sampling knobs https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html

data_parallel_size: int = -1
data_type: dtype | None = None
eos_tokens: list[int] | None = None
kv_cache_size: int = 1024
max_prompt_length: int = 64
max_tokens_to_generate: int = 64
rollout_mapping_config: MappingConfig | None = None
rollout_sglang_jax_context_length: int = 8192
rollout_sglang_jax_disable_radix_cache: bool = True
rollout_sglang_jax_enable_deterministic_sampling: bool = False
rollout_sglang_jax_init_with_random_weights: bool = True
rollout_sglang_jax_mem_fraction_static: float = 0.2
rollout_sglang_jax_model_version: str = ''
rollout_vllm_async_scheduling: bool = False
rollout_vllm_hbm_utilization: float = 0.2
rollout_vllm_init_with_random_weights: bool = True
rollout_vllm_lora_config: dict[str, Any] | None = None
rollout_vllm_model_version: str = ''
rollout_vllm_server_mode: bool = False
rollout_vllm_swap_space_size_gb: float = 4.0
rollout_vllm_tpu_backend_type: str | None = None
seed: Array | None = None
temperature: float = 0.9
tensor_parallel_size: int = -1
top_k: int | None = None
top_p: float | None = 1.0



Distillation
DistillationTrainer(student_model, ...)

Distillation trainer.

DistillationTrainingConfig

alias of TrainingConfig

class tunix.DistillationTrainer(student_model: Module, teacher_model: Module, strategy: BaseStrategy, optimizer: GradientTransformation, training_config: TrainingConfig)
Distillation trainer.

close()
Closes the trainer and its associated resources.

This includes writing any buffered metrics, saving the last checkpoint, and closing the checkpoint manager and metrics logger.

get_eval_loss(model: Module, teacher_output: Any, inputs: dict[str, Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]) → Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray
get_train_loss(model: Module, teacher_output: Any, inputs: dict[str, Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]) → Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray
with_gen_model_input_fn(gen_model_input_fn: Callable[[Any], dict[str, Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray]]) → DistillationTrainer
Generates model input from training input.

NB: output of this function will be passed to the loss function, so the args should match what loss function expects.

Parameters
:
gen_model_input_fn – A function that generates model input from training input.

Returns
:
PeftTrainer.

with_loss_fn(loss_fn: Callable[[...], Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray | Tuple[Array | ndarray | bool | number | bool | int | float | complex | TypedNdArray, Any]], has_aux: bool = False) → DistillationTrainer
tunix.DistillationTrainingConfig
alias of TrainingConfig


Generation
Sampler(transformer, tokenizer, cache_config)

Sampler for transformer model.

CacheConfig(cache_size, num_layers, ...)

Configuration for the KV cache.

class tunix.Sampler(transformer: Module, tokenizer: Any, cache_config: CacheConfig)
Sampler for transformer model.

property dtype: dtype
init_sample_state(all_input_ids: Array, total_sampling_steps: int, include_logits: bool, forbidden_token_ids: Sequence[int] | None, temperature: float, top_p: float | None, top_k: int | None, seed: Array, beam_size: int | None) → _SamplingState
Initializes the sampling state given input prompts.

tokenize(input_string: str) → Array | list[int]
Tokenizes the input string.

property transformer: Module
Returns the transformer module used by the sampler.

property transformer_state: State
Returns the transformer state used by the sampler.

class tunix.CacheConfig(cache_size: int, num_layers: int, num_kv_heads: int, head_dim: int)
Configuration for the KV cache.

cache_size: int
head_dim: int
num_kv_heads: int
num_layers: int


Readme:

Tunix: A JAX-native LLM Post-Training Library


Tunix(Tune-in-JAX) is a JAX based library designed to streamline the post-training of Large Language Models. It provides efficient and scalable supports for:

Supervised Fine-Tuning
Reinforcement Learning (RL)
Knowledge Distillation
Tunix leverages the power of JAX for accelerated computation and seamless integration with JAX-based modeling framework Flax NNX.

Current Status: Early Development

Tunix is in early development. We're actively working to expand its capabilities, usability and improve its performance. Stay tuned for upcoming updates and new features!

Key Features & Highlights
Tunix is still under development, here's a glimpse of the current features:

Supervised Fine-Tuning:
Full Weights Fine-Tuning
Parameter-Efficient Fine-Tuning (PEFT) with LoRA/Q-LoRA Layers
Reinforcement Learning (RL):
Proximal Policy Optimization (PPO)
Group Relative Policy Optimization (GRPO)
Token-level Group Sequence Policy Optimization (GSPO-token)
Preference Fine-Tuning:
Preference alignments with Direct Preference Optimization (DPO)
Knowledge Distillation:
Logit Strategy: A classic approach where the student learns to match the teacher's output probability distribution.
Attention Transfer & Projection Strategies: Methods to align the attention mechanisms between the student and teacher models.
Feature Pooling & Projection Strategies: General techniques for matching intermediate feature representations, even between models of different architectures.
Modularity:
Components are designed to be reusable and composable
Easy to customize and extend
Efficiency:
Native support of common model sharding strategies such as DP, FSDP and TP
Designed for distributed training on accelerators (TPU)
Upcoming
Agentic RL Training:
Async Rollout
Multi-turn & multi-step support
Tool usage
Advanced Algorithms:
Addtional state-of-the-art RL and distillation algorithms
Scalability:
Multi-host distributed training
Optimized rollout with vLLM or SGLang-Jax
User Guides:
More advanced RL recipe
Installation
You can install Tunix in several ways:
From PyPI (recommended):
pip install "google-tunix[prod]"
Directly from GitHub (latest main branch)
pip install git+https://github.com/google/tunix
From source (editable install) If you plan to modify the codebase and run it in development mode. If you'd like to install vllm, the tpu-inference supported version is not released yet, please follow the instructions to install manually (https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/) or download the docker image (vllm/vllm-tpu:v0.11.1) then pip install tpu-inference for TPU backend:
git clone https://github.com/google/tunix.git
cd tunix
pip install -e ".[dev]"

# Then install vLLM and tpu-inference
Using tunix with SGLang-Jax rollout
Install tunix using above ways
Then install SGLang-Jax
git clone git@github.com:sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
Getting Started
To get started, we have a bunch of detailed examples and tutorials.

PEFT Gemma with QLoRA
Training Gemma on grade school Math problems using GRPO
Logit Distillation using Gemma models
Training Llama3 or Qwen2 using GRPO and SGLang-Jax rollout
To setup Jupyter notebook on single host GCP TPU VM, please refer to the setup script.

We plan to provide clear, concise documentation and more examples in the near future.

Contributing and Feedbacks
We welcome contributions! As Tunix is in early development, the contribution process is still being formalized. A rough draft of the contribution process is present here. In the meantime, you can make feature requests, report issues and ask questions in our Tunix GitHub discussion forum.

Collaborations and Partnership
GRL (Game Reinforcement Learning), developed by Hao AI Lab from UCSD, is an open-source framework for post-training large language models through multi-turn RL on challenging games. In collaboration with Tunix, GRL integrates seamless TPU support—letting users quickly run scalable, reproducible RL experiments (like PPO rollouts on Qwen2.5-0.5B-Instruct) on TPU v4 meshes with minimal setup. This partnership empowers the community to push LLM capabilities further, combining Tunix’s optimized TPU runtime with GRL’s flexible game RL pipeline for cutting-edge research and easy reproducibility.

Stay Tuned!