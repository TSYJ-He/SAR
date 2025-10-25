
# 🚀 SAR: How Text-Only Supervision Diminishes Semantic Focus in MLLMs and How to Restore It

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Code-Official%20Implementation-blue" alt="Code"></a>
  <a href="https://github.com/huggingface/llama-factory"><img src="https://img.shields.io/badge/Built%20With-LLaMA--Factory-green" alt="LLaMA-Factory"></a>
</p>

This repository contains the official codebase for the paper **"SAR: How Text-Only Supervision Diminishes Semantic Focus in MLLMs and How to Restore It"**. Our goal is to provide a robust and reproducible framework for all experiments and analyses presented in the paper.

---

## 📖 Overview

This project is built upon the excellent [LLaMA-Factory](https://github.com/huggingface/llama-factory) framework. We assume you have a working knowledge of its core concepts. This document will focus on the specific additions and modifications we have made to support our research.

Our philosophy is to keep the SAR-specific logic modular and cleanly integrated. All new analysis tools and the core SAR algorithm are located in dedicated files, which are then called by the main LLaMA-Factory scripts.

---

## 🏗️ Codebase Structure

Here is an overview of the key additions `[NEW]` and modifications `[MODIFIED]` to the LLaMA-Factory structure.

```

sar-project/  
│  
├── examples/                   \# [NEW] All YAML configs topaper experiments  
└── src/
├── evaluate.py             \# [MODIFIED] To launch our custom analysis tasks
└── llamafactory/
│
├── hparams/
│   └── finetuning\_args.py  \# [MODIFIED] Added CLI arguments for SAR
│
├── analysis/               \# [NEW] Directory for all new analysis modules
│   ├── gradient\_analyzer.py\# [NEW] Callback for gradient sparsity analysis (Sec 3)
│   ├── focus\_metrics.py    \# [NEW] Evaluator for SFS and FI metrics (Sec 4)
│   └── layerwise\_probe.py  \# [NEW] Tool for layer-wise performance probing (Sec 6)
│
├── model/
│   ├── patcher.py          \# [MODIFIED] Injects the SAR logic into models at load time
│   └── sar\_core.py         \# [NEW] Core implementation of the SAR algorithm (SAV, re-weighting)
│
└── train/sft/
└── control\_trainer.py  \# [NEW] Custom trainer for the visual supervision control group (Sec 3)

````

---

## 🔑 Key Component Explanations

* `src/llamafactory/model/sar_core.py`
    * This is the heart of our proposed method. It contains a model-agnostic implementation of the **Semantic Alignment Variance (SAV)** calculation and the **adaptive soft re-weighting** logic.

* `src/llamafactory/model/patcher.py`
    * We've modified this file to include a `patch_model_for_sar` function. When you use the `--use_sar` flag, this function dynamically replaces the standard attention forward pass of the loaded model with our SAR-enhanced version, making SAR a **plug-and-play module**.

* `src/llamafactory/analysis/`
    * This new directory houses all the tools required for our paper's diagnostic experiments.
        * **`gradient_analyzer.py`**: Implements a `TrainerCallback` that hooks into the training loop to capture and log gradient statistics to Weights & Biases.
        * **`focus_metrics.py`**: Implements the `FocusMetricsEvaluator` class, which uses external grounding models (Grounding DINO + SAM) to compute the **SFS** and **FI** scores.
        * **`layerwise_probe.py`**: Implements the `LayerwiseProbe` for training and evaluating linear heads on the outputs of each decoder layer.

* `src/llamafactory/train/sft/control_trainer.py`
    * This new trainer inherits from the standard SFT trainer but overrides the loss computation to include an auxiliary CLIP-style contrastive loss, enabling our crucial **control group experiment**.

* `examples/sar_paper/`
    * This directory is your primary entry point for running experiments. Each YAML file is named according to the paper section it corresponds to and contains all necessary configurations.

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd sar-project
    ```

2.  **Create a virtual environment and install base dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install dependencies for Analysis Tools:**
    ⚠️ The SFS/FI evaluation requires **Grounding DINO**, **Segment Anything**, and **scikit-image**. Please follow their official installation instructions. A typical installation might look like this:
    ```bash
    pip install git+[https://github.com/IDEA-Research/GroundingDINO.git](https://github.com/IDEA-Research/GroundingDINO.git)
    pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
    pip install scikit-image
    # and so on
    ```
    You will also need to download the model checkpoints for Grounding DINO and SAM. Please place them in a designated folder and update the paths in the respective YAML configuration files.

---

## 🧪For Paper Experiments

All experiments are designed to be (and you are suggested to do it as well) launched using the main `train.py` and `evaluate.py` scripts, controlled by the YAML files in `examples/sar_paper/`.

### Experiment for Section 3: Validating Gradient Sparsity

🎯 **Goal**: To produce the analysis showing that text-only supervision leads to gradient sparsity.

#### 1. Standard SFT (Text-Only Group)
This experiment runs a standard SFT process while enabling the gradient analysis callback.

-   **Run Command:**
    ```bash
    python src/train.py --model_name_or_path <path_to_base_model> \
          --do_train \
          --dataset llava_instruct_50k \
          --finetuning_type full \
          --output_dir results/sft_grad_analysis \
          --overwrite_output_dir \
          --per_device_train_batch_size 2 \
          --gradient_accumulation_steps 8 \
          --lr_scheduler_type cosine \
          --logging_steps 10 \
          --save_steps 1000 \
          --learning_rate 5e-6 \
          --num_train_epochs 1.0 \
          --plot_loss \
          --bf16 \
          --report_to wandb \
          --run_name sft_gradient_analysis
    ```
-   **Configuration File**: `examples/sar_paper/exp_sec3.2_grad_sparsity_sft.yaml`
-   **Expected Output**: This will start a training run and log metrics to Weights & Biases. You will find logs under `gradient_analysis/*` and `attention_grad_sparsity_*`, which can be used to generate the plots for **Figure 3 and 4** in the paper.

#### 2. Visually-Grounded SFT (Control Group)
This experiment uses our custom `ControlSFTTrainer` to add an auxiliary contrastive loss.

-   **Run Command:**
    ```bash
    python src/train.py --model_name_or_path <path_to_base_model> \
          --do_train \
          --dataset llava_instruct_50k \
          --finetuning_type full \
          --custom_trainer control_sft \
          --visual_contrastive_loss_weight 0.1 \
          --output_dir results/control_grad_analysis \
          ... # (rest of the args are same as above)
    ```
-   **Configuration File**: `examples/sar_paper/exp_sec3.3_grad_sparsity_control.yaml`
-   **Expected Output**: A separate W&B run will be created. Comparing the gradient sparsity logs from this run with the text-only run will allow you to generate **Figure 4 (Right)**. You will also see `loss_contrastive` being logged.

### Experiment for Section 4: Measuring Semantic Focus Loss

🎯 **Goal**: To compute the SFS and FI scores for a fine-tuned model.

-   **Run Command:**
    ```bash
    python src/evaluate.py --model_name_or_path <path_to_your_sft_model> \
          --do_sfs_fi_eval \
          --dataset refcoco_val_sar \
          --output_dir results/sfs_fi_eval \
          --per_device_eval_batch_size 4 \
          --predict_with_generate \
          --bf16 \
          --gd_config_path <path_to_groundingdino_config> \
          --gd_checkpoint_path <path_to_groundingdino_checkpoint> \
          --sam_checkpoint_path <path_to_sam_checkpoint>
    ```
-   **Configuration File**: `examples/sar_paper/analysis_sec4.3_sfs_fi_eval.yaml`
-   **Expected Output**: The script will run the evaluation and print the **SFS** and **FI** scores to the console. A `focus_metrics_results.json` file will be saved in the output directory.

### Experiment for Section 6.2: Main Evaluation Table

🎯 **Goal**: To generate the main results table by evaluating **Base**, **SFT**, and **SFT+SAR** models.

This is a three-step process for each backbone:

1.  **Evaluate Base Model**:
    ```bash
    python src/evaluate.py --model_name_or_path <path_to_base_model> --task mme --output_dir results/eval_base/mme ...
    ```
2.  **Evaluate SFT Model**:
    ```bash
    python src/evaluate.py --model_name_or_path <path_to_sft_model> --task mme --output_dir results/eval_sft/mme ...
    ```
3.  **Evaluate SFT+SAR Model**:
    Simply add the `--use_sar` flag and its parameters to the SFT evaluation command.
    ```bash
    python src/evaluate.py --model_name_or_path <path_to_sft_model> \
          --use_sar \
          --sar_activation_layer_k 10 \
          --sar_beta 0.5 \
          --task mme \
          --output_dir results/eval_sars/mme \
          ...
    ```
-   **Configuration Files**: Use `eval_sec6.2_main_llava.yaml`, etc., and set the `--model_name_or_path` and `--task` accordingly for each benchmark.
-   **Expected Output**: For each run, a `all_results.json` will be saved in the output directory, containing the scores for the specified benchmark.

### Experiment for Section 6.3: In-depth Analysis & Ablations

#### 1. Layer-wise Performance Analysis
-   **Run Command:**
    ```bash
    python src/evaluate.py --model_name_or_path <path_to_model_to_probe> \
          --do_layerwise_probe \
          --dataset refcoco_val_sar \
          --layerwise_probe_task grounding \
          --output_dir results/layerwise_probe \
          ...
    ```
-   **Configuration File**: `examples/sar_paper/analysis_sec6.3_layerwise_probe.yaml`
-   **Expected Output**: This script trains and evaluates a probe for each layer. The final output is a `layerwise_probe_results.json` file mapping layer index to performance, which can be used to plot **Figure 5**.

#### 2. Ablation on Activation Depth K
This involves running evaluation multiple times with different values for `sar_activation_layer_k`.

-   **Run Commands:**
    ```bash
    # Example for K=5
    python src/evaluate.py --model_name_or_path <path_to_sft_model> \
          --use_sar --sar_activation_layer_k 5 --task mme \
          --output_dir results/ablation_k_5 ...

    # Example for K=15
    python src/evaluate.py --model_name_or_path <path_to_sft_model> \
          --use_sar --sar_activation_layer_k 15 --task mme \
          --output_dir results/ablation_k_15 ...
    ```
-   **Configuration File**: `examples/sar_paper/eval_sec6.3_ablation_k.yaml` can be used as a template.
-   **Expected Output**: A series of result files, each corresponding to a different value of K. You can then aggregate these results to plot **Figure 6**.



## 🎉 Let's Go SAR!
