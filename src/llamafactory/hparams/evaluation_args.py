# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""Arguments pertaining to specify the evaluation parameters."""

    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task_dir: str = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    lang: Literal["en", "zh"] = field(
        default="en",
        metadata={"help": "Language used at evaluation."},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."},
    )
    # --- SAR
    do_sfs_fi_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run Semantic Focus Score (SFS) and Focus Instability (FI) evaluation."},
    )
    gd_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the Grounding DINO model config file."},
    )
    gd_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the Grounding DINO model checkpoint file."},
    )
    sam_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the Segment Anything Model (SAM) checkpoint file."},
    )
    sam_model_type: str = field(
        default="vit_h",
        metadata={"help": "The type of SAM model to load (e.g., 'vit_h', 'vit_l')."},
    )
    do_layerwise_probe: bool = field(
        default=False,
        metadata={"help": "Whether to run layer-wise performance probing analysis."},
    )
    layerwise_probe_task: str = field(
        default="grounding",
        metadata={"help": "The task for layer-wise probing (currently supports 'grounding')."},
    )
    probe_lr: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for training the linear probe."},
    )
    probe_epochs: int = field(
        default=5,
        metadata={"help": "Number of epochs to train each probe."},
    )
    probe_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for probe training."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the output results."},
    )








    # download_mode: DownloadMode = field(
    #     default=DownloadMode.REUSE_DATASET_IF_EXISTS,
    #     metadata={"help": "Download mode used for the evaluation datasets."},
    # )

    # def __post_init__(self):
    #     if self.save_dir is not None and os.path.exists(self.save_dir):
    #         raise ValueError("`save_dir` already exists, use another one.")
