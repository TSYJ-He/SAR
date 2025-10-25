# Semantic Focus Score (SFS) and Focus Instability (FI)
import torch
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from ..extras import logging
if TYPE_CHECKING:
    from datasets import Dataset, tqdm
try:
    from groundingdino.util.inference import load_model as load_gd_model, predict as predict_gd
    _GROUNDINGDINO_AVAILABLE = True
except ImportError:
    _GROUNDINGDINO_AVAILABLE = False
try:
    from segment_anything import SamPredictor, sam_model_registry

    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as ssim
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
logger = logging.get_logger(__name__)


class FocusMetricsEvaluator:
    """
    A class to evaluate the Visual Semantic Focus Loss of MLLMs using
    the Semantic Focus Score (SFS) and Focus Instability (FI) metrics, as
    defined in Section 4.2 of the paper.
    """
    def __init__(
            self,
            gd_config_path: str,
            gd_checkpoint_path: str,
            sam_checkpoint_path: str,
            sam_model_type: str = "vit_h",
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
    ):
        if not _GROUNDINGDINO_AVAILABLE:
            raise ImportError(
                "Please install GroundingDINO (`pip install GroundingDINO`) to use FocusMetricsEvaluator.")
        if not _SAM_AVAILABLE:
            raise ImportError(
                "Please install segment-anything-py (`pip install segment-anything`) to use FocusMetricsEvaluator.")
        if not _SKIMAGE_AVAILABLE:
            raise ImportError(
                "Please install scikit-image (`pip install scikit-image`) to compute SSIM for the FI metric.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.gd_config_path = gd_config_path
        self.gd_checkpoint_path = gd_checkpoint_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_model_type = sam_model_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Lazy load models to save resources until they are first needed
        self._grounding_dino: Optional[PreTrainedModel] = None
        self._sam_predictor: Optional["SamPredictor"] = None

    def _load_grounding_model(self):
        """Loads the Grounding DINO and Segment Anything models on demand."""
        logger.info_rank0("Lazy loading Grounding DINO and SAM models...")
        try:
            self._grounding_dino = load_gd_model(self.gd_config_path, self.gd_checkpoint_path, device=self.device)
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
            sam.to(device=self.device)
            self._sam_predictor = SamPredictor(sam)
            logger.info_rank0("Grounding models loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Grounding DINO or SAM models. Please check checkpoint paths. Error: {e}")

    @property
    def grounding_dino(self):
        if self._grounding_dino is None:
            self._load_grounding_model()
        return self._grounding_dino

    @property
    def sam_predictor(self):
        if self._sam_predictor is None:
            self._load_grounding_model()
        return self._sam_predictor

    @torch.no_grad()
    def _get_ground_truth_mask(self, image: Image.Image, text_prompt: str) -> Optional[np.ndarray]:
        """
        Generates a high-quality ground-truth binary mask for a given text prompt in an image
        by combining Grounding DINO and SAM.
        """
        try:
            # GroundingDINO expects a tensor image
            image_tensor = torch.from_numpy(np.array(image)).to(self.device)
            boxes_filt, _, _ = predict_gd(
                model=self.grounding_dino,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )

            if boxes_filt.size(0) == 0:
                logger.warning_rank0(f"Grounding model did not find any object for prompt: '{text_prompt}'")
                return None

            self.sam_predictor.set_image(np.array(image))

            # Use the box with the highest confidence as the prompt for SAM
            top_box = boxes_filt[0:1].cpu().numpy()

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                box=top_box,
                multimask_output=False,  # We want the single best mask
            )
            return (masks[0] > 0).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error during ground truth mask generation: {e}", exc_info=True)
            return None

    @staticmethod
    def _extract_and_process_attention_map(
            attentions: Tuple[torch.Tensor, ...],
            input_ids: torch.LongTensor,
            image_token_id: int,
            image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Extracts, averages, and resizes the cross-modal attention map from the last decoder layer.
        This is a robust implementation that dynamically finds visual token positions.
        """
        last_layer_attention = attentions[-1].detach()  # Shape: (batch, heads, query_len, key_len)
        # We are interested in the attention from the last generated token (or a representative text token)
        # to all visual tokens. For simplicity and consistency, we use the attention from the last query token.
        last_query_attention = last_layer_attention[0, :, -1, :]  # Shape: (heads, key_len)

        # Dynamically find visual token indices
        is_visual_token = (input_ids[0] == image_token_id)
        if not torch.any(is_visual_token):
            return None

        visual_indices = is_visual_token.nonzero(as_tuple=True)[0]
        num_vision_tokens = len(visual_indices)

        # Ensure the number of vision tokens corresponds to a square grid
        patch_dim = int(np.sqrt(num_vision_tokens))
        if patch_dim * patch_dim != num_vision_tokens:
            logger.warning_rank0(
                f"Number of vision tokens ({num_vision_tokens}) is not a perfect square. Cannot reshape attention map.")
            return None
        # Filter attention to only visual tokens and average across heads
        cross_attention_to_visual = last_query_attention[:, visual_indices]  # Shape: (heads, num_vision_tokens)
        avg_attention = torch.mean(cross_attention_to_visual, dim=0).cpu()  # Shape: (num_vision_tokens,)
        # Reshape to 2D grid
        attention_map = avg_attention.view(patch_dim, patch_dim).numpy()

        # Resize to original image dimensions using high-quality interpolation
        map_img = Image.fromarray(attention_map)
        resized_map = map_img.resize(image_size, Image.Resampling.BICUBIC)
        resized_map_np = np.array(resized_map)

        # Normalize to ensure the map is a probability distribution
        normalized_map = resized_map_np / (resized_map_np.sum() + 1e-9)
        return normalized_map

    @staticmethod
    def _generate_paraphrases(base_query: str, object_prompt: str) -> List[str]:
        """
        Generates a predefined set of semantic paraphrases using a template-based approach
        for consistency and reproducibility.
        """
        templates = [
            "the {} in the image",
            "a photo of the {}",
            "the location of the {}",
            "focus on the {}",
            "where is the {} in this picture?",
        ]
        # This implementation assumes the dataset provides the core object of the query
        return [t.format(object_prompt) for t in templates]

    def evaluate(
            self,
            model: "PreTrainedModel",
            tokenizer: "PreTrainedTokenizerBase",
            dataset: "Dataset",
    ) -> Dict[str, float]:
        """
        Evaluates the given MLLM on the dataset to compute SFS and FI scores.
        The dataset is expected to have 'image', 'query', and 'object_prompt' columns.
        """
        sfs_scores, fi_scores = [], []
        image_token_id = getattr(model.config, "image_token_index", None)
        if image_token_id is None:
            raise ValueError("Model config must have an 'image_token_index' attribute.")

        for sample in tqdm(dataset, desc="Evaluating SFS and FI"):
            image: Image.Image = sample["image"]
            query: str = sample["query"]
            object_prompt: str = sample["object_prompt"]  # The core object for grounding and paraphrasing
            # --- Compute SFS ---
            gt_mask = self._get_ground_truth_mask(image, object_prompt)
            if gt_mask is None:
                continue

            inputs = tokenizer(text=query, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # We need to run generation to get attention scores during autoregressive decoding
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2,  # Generate just enough to get cross-attention
                    output_attentions=True,
                    return_dict_in_generate=True
                )

            # The attentions are a tuple of tuples, one for each generated token
            attentions = outputs.attentions[0]  # Attentions for the first generated token
            attention_map = self._extract_and_process_attention_map(attentions, inputs.input_ids, image_token_id,
                                                                    image.size)

            if attention_map is not None:
                sfs = np.sum(attention_map * gt_mask)
                sfs_scores.append(sfs)
                # --- Compute FI ---
                paraphrases = self._generate_paraphrases(query, object_prompt)
                if not paraphrases:
                    continue
                attention_maps_paraphrased = []
                for para in paraphrases:
                    inputs_para = tokenizer(text=para, images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs_para = model.generate(
                            **inputs_para, max_new_tokens=2, output_attentions=True, return_dict_in_generate=True
                        )

                    attentions_para = outputs_para.attentions[0]
                    map_para = self._extract_and_process_attention_map(attentions_para, inputs_para.input_ids,
                                                                       image_token_id, image.size)
                    if map_para is not None:
                        attention_maps_paraphrased.append(map_para)

                if attention_maps_paraphrased:
                    ssim_scores = [
                        ssim(attention_map, map_para, data_range=attention_map.max() - attention_map.min())
                        for map_para in attention_maps_paraphrased
                    ]
                    fi = 1.0 - np.mean(ssim_scores)
                    fi_scores.append(fi)

        results = {
            "semantic_focus_score": np.mean(sfs_scores) if sfs_scores else 0.0,
            "focus_instability": np.mean(fi_scores) if fi_scores else 0.0,
            "num_samples_evaluated": len(sfs_scores)
        }

        return results