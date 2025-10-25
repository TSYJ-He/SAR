# For Section 6.3. It needs to be carefully checked cause it is newly written.

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from ..extras import logging
if TYPE_CHECKING:
    from datasets import Dataset
logger = logging.get_logger(__name__)

_captured_hidden_states: Dict[int, torch.Tensor] = {}


def _forward_hook(layer_idx: int):
    """Factory to create a forward hook for a specific layer."""
    def hook(module, input, output):
        # For most Transformer decoders, the output is a tuple (hidden_states, ...)
        # We store the main hidden states (output[0])
        _captured_hidden_states[layer_idx] = output[0].detach().cpu()
    return hook
class LayerwiseProbe:
    """
    Performs layer-wise performance probing on a given MLLM.

    This class freezes the MLLM and trains a lightweight linear probe on top of the
    hidden states of each decoder layer to evaluate that layer's representational
    quality for a specific downstream task. This is used for the analysis in Section 6.3.
    """
    def __init__(
            self,
            model: "PreTrainedModel",
            tokenizer: "PreTrainedTokenizerBase",
            task_type: str = "grounding",
            probe_lr: float = 1e-3,
            probe_epochs: int = 5,
            probe_batch_size: int = 32,
    ):
        """
        Initializes the LayerwiseProbe.
        There are many Args:
            model (PreTrainedModel): The MLLM to be probed.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for the MLLM.
            task_type (str): The type of probing task. Currently supports 'grounding'.
            probe_lr (float): Learning rate for training the linear probe.
            probe_epochs (int): Number of epochs to train each probe.
            probe_batch_size (int): Batch size for probe training.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.probe_lr = probe_lr
        self.probe_epochs = probe_epochs
        self.probe_batch_size = probe_batch_size
        self.device = model.device
        self.hidden_size = model.config.hidden_size

        self.decoder_layers = self._get_decoder_layers()
        if not self.decoder_layers:
            raise ValueError("Could not automatically determine decoder layers for this model architecture.")

        self.num_layers = len(self.decoder_layers)
        logger.info_rank0(f"Found {self.num_layers} decoder layers for probing.")

    def _get_decoder_layers(self) -> Optional[nn.ModuleList]:
        """Robustly gets the list of decoder layers from various model architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        return None

    def _create_probe_head(self) -> nn.Module:
        """Creates a task-specific linear probe head."""
        if self.task_type == "grounding":
            # Predicts 4 values for the bounding box [x_center, y_center, width, height]
            return nn.Linear(self.hidden_size, 4).to(self.device)
        else:
            raise NotImplementedError(f"Probing for task type '{self.task_type}' is not implemented.")

    def _prepare_dataset_for_grounding(
            self,
            dataset: "Dataset"
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        """
        Pre-processes the RefCOCOg dataset to create inputs and labels for the probe.
        """
        all_inputs = []
        all_labels = []
        for sample in tqdm(dataset, desc="Preprocessing grounding dataset"):
            image = sample["image"]
            query = sample["query"]
            bbox = sample["bbox"]  # Assuming bbox is [x1, y1, x2, y2] in absolute pixel values
            inputs = self.tokenizer(text=query, images=image, return_tensors="pt")
            all_inputs.append({k: v.squeeze(0) for k, v in inputs.items()})
            # Normalize bbox to [0, 1] and convert to [x_center, y_center, width, height]
            w, h = image.size
            x1, y1, x2, y2 = bbox
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            all_labels.append(torch.tensor([xc, yc, width, height], dtype=torch.float32))
        return all_inputs, all_labels

    def _extract_features(self, layer_idx: int, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Runs a forward pass and extracts hidden states from the specified layer.
        """
        global _captured_hidden_states
        _captured_hidden_states.clear()

        handle = self.decoder_layers[layer_idx].register_forward_hook(_forward_hook(layer_idx))

        batch = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**batch)

        handle.remove()

        hidden_state = _captured_hidden_states.get(layer_idx)
        if hidden_state is None:
            raise RuntimeError(f"Failed to capture hidden state for layer {layer_idx}.")
        # We probe the representation of the last text token
        return hidden_state[0, -1, :].to(self.device)  # (hidden_size,)

    def _train_and_evaluate_probe(
            self,
            train_features: torch.Tensor,
            train_labels: torch.Tensor,
            eval_features: torch.Tensor,
            eval_labels: torch.Tensor,
    ) -> float:
        """
        Trains and evaluates a linear probe for a single layer.
        """
        probe_head = self._create_probe_head()
        optimizer = AdamW(probe_head.parameters(), lr=self.probe_lr)
        loss_fn = nn.L1Loss()  # L1 loss is robust for bounding box regression

        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.probe_batch_size, shuffle=True)

        probe_head.train()
        for _ in range(self.probe_epochs):
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                predictions = probe_head(features)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
        # Evaluation
        probe_head.eval()
        with torch.no_grad():
            eval_preds = probe_head(eval_features.to(self.device))
        # For grounding, calculate Accuracy @ 0.5 IoU
        if self.task_type == "grounding":
            iou_scores = self._calculate_iou(eval_preds.cpu(), eval_labels)
            accuracy = torch.mean((iou_scores >= 0.5).float()).item()
            return accuracy
        return 0.0

    @staticmethod
    def _calculate_iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates IoU for batches of bounding boxes in (xc, yc, w, h) format."""
        # Convert to (x1, y1, x2, y2)
        preds_x1 = preds[:, 0] - preds[:, 2] / 2
        preds_y1 = preds[:, 1] - preds[:, 3] / 2
        preds_x2 = preds[:, 0] + preds[:, 2] / 2
        preds_y2 = preds[:, 1] + preds[:, 3] / 2

        targets_x1 = targets[:, 0] - targets[:, 2] / 2
        targets_y1 = targets[:, 1] - targets[:, 3] / 2
        targets_x2 = targets[:, 0] + targets[:, 2] / 2
        targets_y2 = targets[:, 1] + targets[:, 3] / 2

        # Intersection area
        inter_x1 = torch.max(preds_x1, targets_x1)
        inter_y1 = torch.max(preds_y1, targets_y1)
        inter_x2 = torch.min(preds_x2, targets_x2)
        inter_y2 = torch.min(preds_y2, targets_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, 0) * torch.clamp(inter_y2 - inter_y1, 0)
        # Union area
        preds_area = preds[:, 2] * preds[:, 3]
        targets_area = targets[:, 2] * targets[:, 3]
        union_area = preds_area + targets_area - inter_area

        return inter_area / (union_area + 1e-6)

    def run_probing(
            self,
            dataset: "Dataset",
            train_test_split_ratio: float = 0.8
    ) -> Dict[int, float]:
        """
        Orchestrates the entire layer-wise probing process.
        Args:
            dataset (Dataset): The dataset for the probing task (e.g., RefCOCOg).
            train_test_split_ratio (float): The ratio of data to use for training the probe.
        Returns:
            Dict[int, float]: A dictionary mapping layer index to its performance score.
        """
        if self.task_type == "grounding":
            inputs, labels = self._prepare_dataset_for_grounding(dataset)
        else:
            raise NotImplementedError()

        split_idx = int(len(inputs) * train_test_split_ratio)
        train_inputs, eval_inputs = inputs[:split_idx], inputs[split_idx:]
        train_labels, eval_labels = labels[:split_idx], labels[split_idx:]
        eval_labels_tensor = torch.stack(eval_labels)
        layer_performance = {}

        for layer_idx in tqdm(range(self.num_layers), desc="Probing Layers"):
            # Extract features for this layer for the entire dataset
            train_features_list = [self._extract_features(layer_idx, inp) for inp in train_inputs]
            eval_features_list = [self._extract_features(layer_idx, inp) for inp in eval_inputs]

            train_features_tensor = torch.stack(train_features_list)
            eval_features_tensor = torch.stack(eval_features_list)
            train_labels_tensor = torch.stack(train_labels)
            # Train and evaluate the probe for this layer
            performance = self._train_and_evaluate_probe(
                train_features_tensor,
                train_labels_tensor,
                eval_features_tensor,
                eval_labels_tensor,
            )
            layer_performance[layer_idx] = performance
            logger.info_rank0(f"Layer {layer_idx} | Performance: {performance:.4f}")
        return layer_performance