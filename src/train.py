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
from llamafactory.train.tuner import run_exp
def main():
    run_exp()
    if finetuning_args.custom_trainer == "control_sft":
    from llamafactory.train.sft.control_trainer import ControlSFTTrainer
    from llamafactory.train.sft import workflow
    workflow.CustomSFTTrainer = ControlSFTTrainer # Monkey-patch the trainer class
    logger.info("Using ControlSFTTrainer for visual supervision experiment.")
run_train() # The original function call in train.py that starts the process

def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
