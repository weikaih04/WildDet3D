"""WildDet3D Loss Module.

This module implements the loss function for WildDet3D, combining:
1. SAM3-style 2D losses (IABCEMdetr for classification, L1+GIoU for boxes)
2. 3D-MOOD-style 3D losses (delta_center, depth, dimensions, rotation)

Key Design Decisions:
- Uses SAM3's Hungarian matcher for assignment (already computed in model)
- Follows SAM3's loss normalization (global/local/none)
- Adds 3D regression losses on top of 2D losses
- Supports deep supervision on auxiliary decoder outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vis4d.common.distributed import reduce_mean
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss

from wilddet3d.head.coder_3d import Det3DCoder
from sam3.model.box_ops import fast_diag_box_iou, fast_diag_generalized_box_iou
from sam3.train.matcher import BinaryOneToManyMatcher
from sam3.train.loss.loss_fns import (
    IABCEMdetr,
    Boxes as SAM3Boxes,
    sigmoid_focal_loss,
)


def _packed_to_padded(boxes_packed: Tensor, num_boxes: Tensor, fill_value: float = 0.0) -> Tensor:
    """Convert packed tensor to padded tensor.

    This function converts a packed (concatenated) tensor of bounding boxes
    to a batch-wise padded tensor, following SAM3's collator implementation.

    Args:
        boxes_packed: Packed boxes tensor of shape (N_total, 4) where
                     N_total = N_1 + N_2 + ... + N_B
        num_boxes: Number of boxes per image, shape (B,)
        fill_value: Value to use for padding (default: 0.0)

    Returns:
        Padded boxes tensor of shape (B, max_N, 4) where max_N = max(num_boxes)

    Example:
        >>> boxes = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        >>> num_boxes = torch.tensor([1, 2])
        >>> padded = _packed_to_padded(boxes, num_boxes)
        >>> padded.shape
        torch.Size([2, 2, 4])
    """
    B = num_boxes.shape[0]
    Ns = num_boxes.tolist()
    max_N = max(Ns)

    # Create padded tensor
    boxes_padded = boxes_packed.new_full((B, max_N, *boxes_packed.shape[1:]), fill_value)

    # Fill in actual boxes
    prev_idx = 0
    for i in range(B):
        next_idx = prev_idx + Ns[i]
        boxes_padded[i, :Ns[i]] = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx

    return boxes_padded


@dataclass
class WildDet3DLossConfig:
    """Configuration for WildDet3D loss.

    Follows SAM3's loss configuration style with additional 3D loss weights.
    """
    # ========== Global Scale Factors ==========
    # These allow adjusting the balance between 2D, 3D, and geometry losses
    # Default 1.0, can be adjusted in training config to tune 2D:3D:Geom ratio
    loss_2d_scale: float = 1.0  # Scale for 2D losses (cls, bbox, giou)
    loss_3d_scale: float = 1.0  # Scale for 3D losses (delta, depth, dim, rot)
    loss_geom_scale: float = 10.0  # Scale for geometry backend losses (SILog, SSI, camera angles)

    # ========== O2M (One-to-Many) Matcher Configuration ==========
    # Note: O2O matcher is configured in wilddet3d.py (self.sam3.matcher)
    use_o2m: bool = True  # Enable O2M matching
    o2m_loss_clip: float = 150.0  # Clip O2M loss to prevent gradient explosion
    o2m_alpha: float = 0.3  # Alpha for O2M cost computation
    o2m_threshold: float = 0.4  # IoU threshold for O2M matching
    o2m_topk: int = 4  # Top-k predictions per GT (SAM3 original: topk: 4)
    o2m_loss_weight: float = 2.0  # Weight for O2M loss (SAM3 original: o2m_weight: 2.0)

    # ========== 2D Loss Weights (SAM3 style) ==========
    # Classification loss (IABCEMdetr style)
    loss_cls_weight: float = 20.0  # SAM3 original
    pos_weight: float = 5.0  # SAM3 original (was incorrectly 10.0)
    gamma: float = 2.0  # SAM3 original focal (was incorrectly 0.0)
    alpha: float = 0.25  # IoU-aware alpha

    # IABCEMdetr advanced features
    use_weak_loss: bool = False  # Enable weak supervision (SAM3 original: weak_loss: False)
    weak_loss_weight: float = 1.0  # Weight for weak loss (only used if use_weak_loss=True)
    use_presence: bool = True  # Enable presence loss (per-category presence detection)
    presence_loss_weight: float = 20.0  # Weight for presence loss (SAM3 original: presence_weight: 20.0)
    presence_alpha: float = 0.5  # SAM3 original presence focal loss alpha
    presence_gamma: float = 0.0  # SAM3 original (gamma=0 = plain BCE, no focal weighting)

    # Box regression loss
    loss_bbox_weight: float = 5.0  # L1 loss weight
    loss_giou_weight: float = 2.0  # GIoU loss weight
    
    # ========== 3D Loss Weights (3D-MOOD style) ==========
    loss_delta_2d_weight: float = 1.0  # Delta 2D center
    loss_depth_weight: float = 1.0  # Log depth
    loss_dim_weight: float = 1.0  # Log dimensions
    loss_rot_weight: float = 1.0  # 6D rotation
    
    # ========== Geometry Backend Loss Weights ==========
    loss_silog_weight: float = 1.0  # SILog depth loss
    loss_phi_weight: float = 0.1  # Phi angle loss
    loss_theta_weight: float = 0.1  # Theta angle loss
    loss_opt_ssi_weight: float = 0.5  # SSI loss weight (UniDepthV2)
    
    # ========== Normalization ==========
    normalization: Literal["global", "local", "none"] = "global"
    
    # ========== Auxiliary Loss ==========
    aux_loss_weight: float = 1.0  # Weight for auxiliary decoder outputs
    
    # ========== Mask Loss (optional) ==========
    loss_mask_weight: float = 0.0  # Set > 0 to enable mask loss
    loss_dice_weight: float = 0.0  # Set > 0 to enable dice loss

    # ========== 3D Confidence Head ==========
    # Positive: soft target = quality (iou_3d + depth). Negative: push to 0.
    # Inference: final_score = 2d_score + conf_3d_inference_weight * 3d_score
    use_3d_conf: bool = False  # Enable 3D confidence head loss
    loss_3d_conf_weight: float = 20.0  # Weight for 3D confidence loss (same as 2D loss_cls_weight)
    conf_depth_weight: float = 0.7  # Weight for depth quality in quality target
    conf_iou_3d_weight: float = 0.3  # Weight for 3D IoU in quality target

    # ========== Ignore Box Negative Loss Suppression ==========
    # Suppress negative classification loss for predictions that overlap
    # with ignore-annotated objects (truncated, occluded, etc.).
    # This aligns training with eval, where such detections are neutral.
    use_ignore_suppress: bool = False
    ignore_iou_threshold: float = 0.5  # 2D IoU threshold for suppression


class WildDet3DLoss(nn.Module):
    """Loss function for WildDet3D.
    
    Combines SAM3-style 2D losses with 3D-MOOD-style 3D losses.
    
    Loss Components:
    1. Classification: IABCEMdetr (IoU-aware BCE with soft targets)
    2. 2D Box: L1 + GIoU
    3. 3D Box: L1 for (delta_center, log_depth, log_dims, rot_6d)
    4. Geometry: SILog depth + phi/theta angles (from geometry backend)
    """
    
    def __init__(
        self,
        config: WildDet3DLossConfig | None = None,
        box_coder: Det3DCoder | None = None,
    ) -> None:
        """Initialize WildDet3D loss.

        Args:
            config: Loss configuration
            box_coder: 3D box encoder/decoder for target encoding
        """
        super().__init__()
        self.config = config or WildDet3DLossConfig()
        self.box_coder = box_coder or Det3DCoder()
        self.reg_dims = self.box_coder.reg_dims

        # SAM3's 2D loss classes (directly imported from sam3.train.loss.loss_fns)
        # weak_loss=False follows SAM3's own training configs — all unmatched
        # predictions receive negative loss regardless of is_exhaustive.
        self.cls_loss = IABCEMdetr(
            pos_weight=self.config.pos_weight,
            gamma=self.config.gamma,
            alpha=self.config.alpha,
            weak_loss=False,
            use_presence=self.config.use_presence,
            presence_alpha=self.config.presence_alpha,
            presence_gamma=self.config.presence_gamma,
        )
        self.box_loss = SAM3Boxes()

        # O2M matcher for DAC one-to-many loss
        if self.config.use_o2m:
            self.o2m_matcher = BinaryOneToManyMatcher(
                alpha=self.config.o2m_alpha,
                threshold=self.config.o2m_threshold,
                topk=self.config.o2m_topk,
            )
        else:
            self.o2m_matcher = None

    def _compute_ignore_neg_mask(
        self,
        pred_boxes: Tensor,
        ignore_boxes: Tensor,
        num_ignores: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Compute mask for predictions overlapping ignore boxes.

        Args:
            pred_boxes: (B, S, 4) normalized xyxy predicted boxes.
            ignore_boxes: (B, max_ignore, 4) normalized xyxy ignore boxes.
            num_ignores: (B,) number of valid ignore boxes per prompt.
            threshold: 2D IoU threshold above which to suppress.

        Returns:
            mask: (B, S) float. 1.0 = suppress negative loss, 0.0 = normal.
        """
        import torchvision.ops

        B, S, _ = pred_boxes.shape
        device = pred_boxes.device
        mask = torch.zeros(B, S, device=device)

        for b in range(B):
            n_ign = num_ignores[b].item()
            if n_ign == 0:
                continue
            iou = torchvision.ops.box_iou(
                pred_boxes[b],
                ignore_boxes[b, :n_ign],
            )  # (S, n_ign)
            mask[b] = (iou.max(dim=1).values > threshold).float()

        return mask

    def _build_targets_from_batch(
        self, batch: "WildDet3DInput"
    ) -> dict[str, Tensor]:
        """Build targets dict from WildDet3DInput.

        WildDet3D uses per-category queries with multi-instance targets.
        The collator produces:
        - gt_boxes2d: (N_prompts, max_gt, 4) - multiple GTs per query
        - gt_boxes3d: (N_prompts, max_gt, 12) - multiple GTs per query (if available)
        - num_gts: (N_prompts,) - number of valid GTs per query (can be > 1)

        We convert this to the packed format expected by loss computation.

        Args:
            batch: WildDet3DInput containing GT boxes

        Returns:
            targets dict with:
            - boxes_xyxy: (N_total, 4) GT boxes in xyxy format (packed)
            - boxes_3d: (N_total, 12) 3D GT boxes (packed)
            - num_boxes: (N_prompts,) number of GTs per query
            - intrinsics: (N_prompts, 3, 3) camera intrinsics per prompt
        """
        device = batch.images.device
        N_prompts = batch.img_ids.shape[0]

        # Extract GT from batch
        gt_boxes2d = batch.gt_boxes2d  # (N_prompts, max_gt, 4) or (N_prompts, 4)
        gt_boxes3d = batch.gt_boxes3d  # (N_prompts, max_gt, 12) or None
        num_gts = batch.num_gts  # (N_prompts,) number of valid GTs per query

        if gt_boxes2d is None:
            # No GT available
            return {
                "boxes_xyxy": torch.zeros(0, 4, device=device),
                "boxes_3d": torch.zeros(0, 12, device=device),
                "classes": torch.zeros(0, dtype=torch.long, device=device),
                "num_boxes": torch.zeros(N_prompts, dtype=torch.long, device=device),
                "intrinsics": batch.intrinsics[batch.img_ids],
            }

        # Handle both old (N_prompts, 4) and new (N_prompts, max_gt, 4) formats
        if gt_boxes2d.dim() == 2:
            # Old format: (N_prompts, 4) - single GT per prompt
            boxes_xyxy = gt_boxes2d
            if num_gts is None:
                num_gts = torch.ones(N_prompts, dtype=torch.long, device=device)

            if gt_boxes3d is not None and gt_boxes3d.dim() == 2:
                boxes_3d = gt_boxes3d
            else:
                boxes_3d = torch.zeros(N_prompts, 12, device=device)
        else:
            # New format: (N_prompts, max_gt, 4) - multi-instance targets
            # Pack valid boxes into a flat tensor
            if num_gts is None:
                # Fallback: assume all boxes are valid
                num_gts = torch.tensor([gt_boxes2d.shape[1]] * N_prompts, dtype=torch.long, device=device)

            # Pack boxes into (N_total, 4)
            boxes_list = []
            boxes_3d_list = []
            for i in range(N_prompts):
                n_gt = num_gts[i].item()
                boxes_list.append(gt_boxes2d[i, :n_gt])  # (n_gt, 4)
                if gt_boxes3d is not None:
                    boxes_3d_list.append(gt_boxes3d[i, :n_gt])  # (n_gt, 12)

            if boxes_list:
                boxes_xyxy = torch.cat(boxes_list, dim=0)  # (N_total, 4)
            else:
                boxes_xyxy = torch.zeros(0, 4, device=device)

            if boxes_3d_list:
                boxes_3d = torch.cat(boxes_3d_list, dim=0)  # (N_total, 12)
            else:
                box3d_dim = gt_boxes3d.shape[-1] if gt_boxes3d is not None else 12
                boxes_3d = torch.zeros(boxes_xyxy.shape[0], box3d_dim, device=device)

        # SAM3 uses binary detection (all targets are class 1)
        N_total = boxes_xyxy.shape[0]
        classes = torch.ones(N_total, dtype=torch.long, device=device)

        # Get per-prompt intrinsics
        intrinsics = batch.intrinsics[batch.img_ids]  # (N_prompts, 3, 3)

        # SAM3's IABCEMdetr and Boxes loss classes need additional formats:
        # - boxes (cxcywh packed) for L1 loss
        # - boxes_padded (cxcywh padded) for presence keep_loss
        # - object_ids_padded for presence keep_loss
        # - is_exhaustive for weak loss masking
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)

        # Padded format (B, max_N, 4) for presence loss keep_loss computation
        boxes_padded = _packed_to_padded(boxes_cxcywh, num_gts)
        max_N = boxes_padded.shape[1]

        # Object IDs: sequential within each prompt's targets
        object_ids_padded = torch.full(
            (N_prompts, max_N), -1, dtype=torch.long, device=device
        )
        offset = 0
        for i in range(N_prompts):
            n = int(num_gts[i].item())
            if n > 0:
                object_ids_padded[i, :n] = torch.arange(
                    offset, offset + n, device=device
                )
                offset += n

        # is_exhaustive: multi-target queries are exhaustive, single-target are not
        # query_types: 0=TEXT, 1=VISUAL, 3=VISUAL+LABEL → exhaustive (True)
        # query_types: 2=GEOMETRY, 4=GEOMETRY+LABEL → not exhaustive (False)
        if batch.query_types is not None:
            qt = batch.query_types.to(device)
            is_exhaustive = (qt == 0) | (qt == 1) | (qt == 3)
        else:
            is_exhaustive = torch.ones(N_prompts, dtype=torch.bool, device=device)

        return {
            "boxes_xyxy": boxes_xyxy,
            "boxes": boxes_cxcywh,
            "boxes_padded": boxes_padded,
            "boxes_3d": boxes_3d,
            "classes": classes,
            "num_boxes": num_gts,
            "intrinsics": intrinsics,
            "object_ids_padded": object_ids_padded,
            "is_exhaustive": is_exhaustive,
        }

    def forward(
        self,
        out: "WildDet3DOutput",
        batch: "WildDet3DInput",
    ) -> dict[str, Tensor]:
        """Compute all losses.

        vis4d LossModule interface: expects either Tensor, dict, or namedtuple.
        We return a dict of tensors, and LossModule will sum them automatically.

        Following SAM3 and GDino3D's design, we compute 2D box L1 loss in normalized
        cxcywh space and GIoU loss in pixel xyxy space for consistent loss weights.

        Args:
            out: Model output (WildDet3DOutput dataclass)
            batch: Input batch (WildDet3DInput dataclass)

        Returns:
            Dict of loss tensors (vis4d LossModule will sum them)
        """
        import time
        import os
        import torch
        _PROFILE_LOSS = os.environ.get("PROFILE_WILDDET3D", "0") == "1"
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_start = time.perf_counter()
        # Unpack model outputs
        pred_logits = out.pred_logits
        pred_boxes_2d = out.pred_boxes_2d
        pred_boxes_3d = out.pred_boxes_3d
        aux_outputs = out.aux_outputs
        geom_losses = out.geom_losses

        # Build targets from batch
        # Get per-prompt intrinsics by indexing into batch intrinsics
        B_images = batch.images.shape[0]
        N_prompts = batch.img_ids.shape[0]
        intrinsics = batch.intrinsics[batch.img_ids]  # (N_prompts, 3, 3)

        # Image size from batch
        image_size = (batch.images.shape[2], batch.images.shape[3])  # (H, W)

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t_targets = time.perf_counter()

        targets = self._build_targets_from_batch(batch)
        losses = {}

        # Normalize targets to [0, 1] range (for matching and computation)
        normalized_targets = self._normalize_targets(targets)

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_targets_time = (time.perf_counter() - _t_targets) * 1000

        # Store image_size for pixel coordinate conversion
        if image_size is None and "image_size" in targets:
            image_size = targets["image_size"]

        # Get matching indices from SAM3's internal matching
        # SAM3's forward_grounding computes indices via _compute_matching when find_target is provided
        # Handle empty batch (N_prompts=0) case - return zero loss with grad
        if out.indices is None:
            device = pred_logits.device
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[WildDet3D Loss] Empty batch detected on rank {rank}, returning zero loss")

            # CRITICAL: Must still participate in all_reduce to prevent DDP deadlock
            # Other ranks may have non-empty batches and will call all_reduce
            if self.config.normalization == "global" and torch.distributed.is_initialized():
                dummy_num_boxes = torch.tensor(0.0, device=device)
                torch.distributed.all_reduce(dummy_num_boxes)

            # Use pred_logits.sum() * 0 to maintain computation graph for DDP
            zero_loss = pred_logits.sum() * 0
            return {
                "loss_cls": zero_loss,  # Keep grad for DDP
                "loss_bbox": zero_loss.clone(),
                "loss_giou": zero_loss.clone(),
            }

        batch_idx, src_idx, tgt_idx = out.indices

        # Move indices to the same device as predictions
        batch_idx = batch_idx.to(pred_logits.device)
        src_idx = src_idx.to(pred_logits.device)
        tgt_idx = tgt_idx.to(pred_logits.device) if tgt_idx is not None else None

        indices = (batch_idx, src_idx, tgt_idx)

        # Get number of boxes for normalization
        num_boxes = self._get_num_boxes(normalized_targets)
        
        # ========== 2D Losses via SAM3's loss classes (scaled by loss_2d_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # Build SAM3-format outputs dict for loss classes
        sam3_outputs = {
            "pred_logits": pred_logits,
            "pred_boxes_xyxy": pred_boxes_2d,
            "pred_boxes": out.pred_boxes_2d_cxcywh,
        }
        if out.presence_logits is not None:
            sam3_outputs["presence_logit_dec"] = out.presence_logits

        # Compute ignore negative loss suppression mask
        if (
            self.config.use_ignore_suppress
            and batch.ignore_boxes2d is not None
            and batch.num_ignores is not None
        ):
            normalized_targets["_ignore_boxes2d"] = batch.ignore_boxes2d
            normalized_targets["_num_ignores"] = batch.num_ignores
            normalized_targets["ignore_neg_mask"] = (
                self._compute_ignore_neg_mask(
                    pred_boxes_2d,
                    batch.ignore_boxes2d,
                    batch.num_ignores,
                    threshold=self.config.ignore_iou_threshold,
                )
            )

        # Classification + presence via SAM3's IABCEMdetr
        cls_losses = self.cls_loss.get_loss(
            sam3_outputs, normalized_targets, indices, num_boxes
        )
        losses["loss_cls"] = (
            self.config.loss_2d_scale * cls_losses["loss_ce"] * self.config.loss_cls_weight
        )
        # Metrics from SAM3's IABCEMdetr (not losses, just for wandb logging)
        if "ce_f1" in cls_losses:
            losses["metric_ce_f1"] = cls_losses["ce_f1"].detach()
        # Presence loss (computed inside IABCEMdetr when use_presence=True)
        presence_val = cls_losses.get("presence_loss")
        if presence_val is not None and isinstance(presence_val, Tensor):
            losses["loss_presence"] = (
                self.config.loss_2d_scale * presence_val
                * self.config.presence_loss_weight
            )
            if "presence_dec_acc" in cls_losses:
                losses["metric_presence_acc"] = cls_losses["presence_dec_acc"].detach()

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_cls_time = (time.perf_counter() - _t0) * 1000

        # 2D box losses (L1 + GIoU) via SAM3's Boxes class
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        box_losses = self.box_loss.get_loss(
            sam3_outputs, normalized_targets, indices, num_boxes
        )
        losses["loss_bbox"] = (
            self.config.loss_2d_scale * box_losses["loss_bbox"] * self.config.loss_bbox_weight
        )
        losses["loss_giou"] = (
            self.config.loss_2d_scale * box_losses["loss_giou"] * self.config.loss_giou_weight
        )

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_2d_box_time = (time.perf_counter() - _t1) * 1000

        # ========== O2M Loss (2D scaled by loss_2d_scale, 3D scaled by loss_3d_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t_o2m = time.perf_counter()
            _loss_o2m_time = 0

        # Use real O2M outputs from SAM3 DAC mechanism (not O2O outputs)
        if self.config.use_o2m and self.o2m_matcher is not None and out.pred_logits_o2m is not None:
            o2m_losses = self._loss_o2m(
                pred_logits=out.pred_logits_o2m,
                pred_boxes_2d=out.pred_boxes_2d_o2m,
                pred_boxes_2d_cxcywh=out.pred_boxes_2d_cxcywh_o2m,
                pred_boxes_3d=out.pred_boxes_3d_o2m,
                targets=normalized_targets,
                num_boxes=num_boxes,
                intrinsics=intrinsics,
                image_size=image_size,
                pred_conf_3d=out.pred_conf_3d_o2m,
            )
            # Apply appropriate scale and loss weights (following SAM3 original)
            # SAM3 original: loss = loss_value * o2m_weight * loss_weight
            # We need to apply the individual loss weights, not just o2m_loss_weight
            o2m_weight_map = {
                "loss_cls": self.config.loss_cls_weight,
                "loss_bbox": self.config.loss_bbox_weight,
                "loss_giou": self.config.loss_giou_weight,
                "loss_delta_2d": self.config.loss_delta_2d_weight,
                "loss_depth": self.config.loss_depth_weight,
                "loss_dim": self.config.loss_dim_weight,
                "loss_rot": self.config.loss_rot_weight,
                "loss_3d_cls": self.config.loss_3d_conf_weight,
            }
            for key, value in o2m_losses.items():
                loss_weight = o2m_weight_map.get(key, 1.0)
                if key in ("loss_delta_2d", "loss_depth", "loss_dim", "loss_rot"):
                    # 3D losses use loss_3d_scale
                    o2m_loss_val = (
                        self.config.loss_3d_scale * value * loss_weight * self.config.o2m_loss_weight
                    )
                elif key == "loss_3d_cls":
                    # 3D confidence loss: weight * o2m_weight (no extra scale)
                    o2m_loss_val = value * loss_weight * self.config.o2m_loss_weight
                else:
                    # 2D losses (loss_cls, loss_bbox, loss_giou) use loss_2d_scale
                    o2m_loss_val = (
                        self.config.loss_2d_scale * value * loss_weight * self.config.o2m_loss_weight
                    )
                # Clip O2M loss to prevent gradient explosion
                losses[f"o2m_{key}"] = torch.clamp(o2m_loss_val, max=self.config.o2m_loss_clip)

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_o2m_time = (time.perf_counter() - _t_o2m) * 1000

        # ========== 3D Losses (scaled by loss_3d_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            _loss_3d_time = 0

        if pred_boxes_3d is not None and intrinsics is not None:
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d, pred_boxes_3d, indices, normalized_targets,
                intrinsics, num_boxes, image_size=image_size
            )
            # Apply loss_3d_scale to all 3D losses
            for key, value in loss_3d.items():
                losses[key] = self.config.loss_3d_scale * value

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_3d_time = (time.perf_counter() - _t2) * 1000

        # ========== 3D Confidence Loss (positive samples only) ==========
        if (self.config.use_3d_conf
                and out.pred_conf_3d is not None
                and pred_boxes_3d is not None
                and intrinsics is not None):
            loss_3d_cls = self._loss_3d_classification(
                out.pred_conf_3d, pred_boxes_2d, pred_boxes_3d,
                indices, normalized_targets, intrinsics, num_boxes, image_size,
            )
            losses["loss_3d_cls"] = self.config.loss_3d_conf_weight * loss_3d_cls

        # ========== Geometry Backend Losses (scaled by loss_geom_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t_geom = time.perf_counter()
            _loss_geom_time = 0

        if geom_losses is not None:
            for key, value in geom_losses.items():
                if key.startswith("metric_"):
                    # Monitoring-only: log raw value, no scaling
                    losses[key] = value.detach()
                else:
                    weight = getattr(
                        self.config, f"loss_{key}_weight", 1.0
                    )
                    losses[f"loss_{key}"] = (
                        self.config.loss_geom_scale * value * weight
                    )

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_geom_time = (time.perf_counter() - _t_geom) * 1000

        # ========== Auxiliary Losses (Deep Supervision) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()
            _loss_aux_time = 0

        _num_aux_layers = 0
        if aux_outputs is not None:
            _num_aux_layers = len(aux_outputs)
            for i, aux_out in enumerate(aux_outputs):
                aux_losses = self._compute_aux_loss(
                    aux_out, indices, normalized_targets, num_boxes, intrinsics, image_size
                )
                for key, value in aux_losses.items():
                    losses[f"d{i}.{key}"] = value * self.config.aux_loss_weight

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_aux_time = (time.perf_counter() - _t3) * 1000
            _loss_total_time = (time.perf_counter() - _loss_start) * 1000

            # Print loss timing summary (every N steps via profiler)
            from wilddet3d.ops.profiler import profiler
            p = profiler()
            p.current_step_timings["loss_total"] = _loss_total_time / 1000
            p.current_step_timings["  loss_targets"] = _loss_targets_time / 1000
            p.current_step_timings["  loss_cls"] = _loss_cls_time / 1000
            p.current_step_timings["  loss_2d_box"] = _loss_2d_box_time / 1000
            p.current_step_timings["  loss_o2m"] = _loss_o2m_time / 1000
            p.current_step_timings["  loss_3d"] = _loss_3d_time / 1000
            p.current_step_timings["  loss_geom"] = _loss_geom_time / 1000
            p.current_step_timings["  loss_aux"] = _loss_aux_time / 1000
            p.current_step_timings["  loss_aux_layers"] = _num_aux_layers

        # ========== Ensure all losses are tensors ==========
        # vis4d LossModule expects dict of tensors
        for k, v in list(losses.items()):
            if not isinstance(v, Tensor):
                losses[k] = torch.tensor(v, device=pred_logits.device)

        # vis4d LossModule will sum all losses in the dict automatically
        return losses

    def _get_num_boxes(self, targets: dict) -> Tensor:
        """Get number of boxes for loss normalization."""
        num_boxes = targets["num_boxes"].sum().float()

        if self.config.normalization == "global":
            # Handle non-distributed case
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(num_boxes)
                world_size = torch.distributed.get_world_size()
                num_boxes = torch.clamp(num_boxes / world_size, min=1)
            else:
                # Non-distributed: just clamp
                num_boxes = torch.clamp(num_boxes, min=1)
        elif self.config.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        else:  # "none"
            num_boxes = torch.ones_like(num_boxes)

        return num_boxes

    # 2D classification and box losses are now handled by SAM3's
    # IABCEMdetr (self.cls_loss) and Boxes (self.box_loss) classes.

    def _loss_o2m(
        self,
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        pred_boxes_2d_cxcywh: Tensor | None,  # (B, S, 4) normalized cxcywh
        pred_boxes_3d: Tensor | None,  # (B, S, reg_dims)
        targets: dict,
        num_boxes: Tensor,
        intrinsics: Tensor | None = None,  # (B, 3, 3)
        image_size: tuple[int, int] | None = None,
        pred_conf_3d: Tensor | None = None,  # (B, S, 1) 3D confidence
    ) -> dict[str, Tensor]:
        """Compute O2M (One-to-Many) auxiliary loss.

        Uses SAM3's IABCEMdetr and Boxes classes for 2D losses,
        plus our own 3D loss for matched predictions.
        """
        losses = {}
        device = pred_logits.device
        B, S = pred_logits.shape[:2]

        # Prepare targets in padded format for O2M matcher
        num_boxes_per_image = targets.get(
            "num_boxes",
            torch.tensor([len(targets["boxes_xyxy"])], device=device),
        )
        boxes_padded = targets.get("boxes_padded")
        if boxes_padded is None:
            boxes_cxcywh = self._xyxy_to_cxcywh(targets["boxes_xyxy"])
            boxes_padded = _packed_to_padded(boxes_cxcywh, num_boxes_per_image)

        max_N = boxes_padded.shape[1]
        target_is_valid_padded = torch.zeros(
            B, max_N, dtype=torch.bool, device=device
        )
        for i in range(B):
            target_is_valid_padded[i, :num_boxes_per_image[i]] = True

        # O2M matching
        if pred_boxes_2d_cxcywh is None:
            pred_boxes_2d_cxcywh = self._xyxy_to_cxcywh(pred_boxes_2d)

        outputs_dict = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes_2d_cxcywh,
        }
        targets_dict = {
            "boxes_padded": boxes_padded,
            "labels": targets["classes"],
            "num_boxes": num_boxes_per_image,
        }
        batch_idx, src_idx, tgt_idx = self.o2m_matcher(
            outputs_dict,
            targets_dict,
            target_is_valid_padded=target_is_valid_padded,
        )

        if batch_idx.numel() == 0:
            zero_losses = {
                "loss_cls": torch.tensor(0.0, device=device),
                "loss_bbox": torch.tensor(0.0, device=device),
                "loss_giou": torch.tensor(0.0, device=device),
            }
            if pred_boxes_3d is not None and intrinsics is not None:
                zero_losses.update({
                    "loss_delta_2d": torch.tensor(0.0, device=device),
                    "loss_depth": torch.tensor(0.0, device=device),
                    "loss_dim": torch.tensor(0.0, device=device),
                    "loss_rot": torch.tensor(0.0, device=device),
                })
            return zero_losses

        o2m_indices = (batch_idx, src_idx, tgt_idx)

        # Recompute ignore mask for O2M predictions (different pred boxes)
        if "_ignore_boxes2d" in targets:
            targets = targets.copy()
            targets["ignore_neg_mask"] = self._compute_ignore_neg_mask(
                pred_boxes_2d,
                targets["_ignore_boxes2d"],
                targets["_num_ignores"],
                threshold=self.config.ignore_iou_threshold,
            )

        # 2D losses via SAM3 classes
        o2m_outputs = {
            "pred_logits": pred_logits,
            "pred_boxes_xyxy": pred_boxes_2d,
            "pred_boxes": pred_boxes_2d_cxcywh,
        }
        cls_losses = self.cls_loss.get_loss(
            o2m_outputs, targets, o2m_indices, num_boxes
        )
        losses["loss_cls"] = cls_losses["loss_ce"]

        box_losses = self.box_loss.get_loss(
            o2m_outputs, targets, o2m_indices, num_boxes
        )
        losses["loss_bbox"] = box_losses["loss_bbox"]
        losses["loss_giou"] = box_losses["loss_giou"]

        # 3D losses (our own, not in SAM3)
        if (pred_boxes_3d is not None and intrinsics is not None
                and "boxes_3d" in targets):
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d=pred_boxes_2d,
                pred_boxes_3d=pred_boxes_3d,
                indices=o2m_indices,
                targets=targets,
                intrinsics=intrinsics,
                num_boxes=num_boxes,
                image_size=image_size,
            )
            losses.update(loss_3d)

        # 3D confidence loss (O2M branch)
        if (self.config.use_3d_conf
                and pred_conf_3d is not None
                and pred_boxes_3d is not None
                and intrinsics is not None):
            loss_3d_cls = self._loss_3d_classification(
                pred_conf_3d, pred_boxes_2d, pred_boxes_3d,
                o2m_indices, targets, intrinsics, num_boxes, image_size,
            )
            losses["loss_3d_cls"] = loss_3d_cls

        return losses

    # _loss_boxes_2d replaced by SAM3's Boxes class (self.box_loss).

    def _loss_boxes_3d(
        self,
        pred_boxes_2d: Tensor,  # (B, S, 4)
        pred_boxes_3d: Tensor,  # (B, S, reg_dims)
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        intrinsics: Tensor,
        num_boxes: Tensor,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Compute 3D box regression losses.

        Args:
            pred_boxes_2d: Predicted 2D boxes in normalized xyxy [0,1]. Shape (B, S, 4).
            pred_boxes_3d: Predicted 3D box parameters. Shape (B, S, reg_dims).
            indices: Matching indices (batch_idx, src_idx, tgt_idx).
            targets: Target dict containing boxes_3d.
            intrinsics: Camera intrinsics. Shape (B, 3, 3).
            num_boxes: Number of matched boxes for normalization.
            image_size: (H, W) tuple for converting normalized to pixel coords.
                Required for correct box_coder.encode() which expects pixel coords.
        """
        batch_idx, src_idx, tgt_idx = indices

        # Get matched predictions (for loss computation)
        src_boxes_3d = pred_boxes_3d[(batch_idx, src_idx)]

        # Get matched GT 2D boxes (for box_coder.encode target computation)
        # IMPORTANT: Use GT 2D boxes, NOT predicted boxes!
        # This matches GDino3D's design where encode() uses GT 2D boxes to compute
        # stable targets, while decode() at inference uses predicted 2D boxes.
        target_boxes_2d = (
            targets["boxes_xyxy"][tgt_idx] if tgt_idx is not None
            else targets["boxes_xyxy"]
        )

        # Get matched GT 3D boxes
        target_boxes_3d = (
            targets["boxes_3d"][tgt_idx] if tgt_idx is not None
            else targets["boxes_3d"]
        )

        # Get intrinsics for matched samples
        # Note: intrinsics is (B, 3, 3), need to index by batch_idx
        # Since box_coder.encode() expects single intrinsics (3, 3),
        # we need to process each matched box individually
        if len(batch_idx) == 0:
            # No matches, return zero losses
            return {
                "loss_delta_2d": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_depth": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_dim": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_rot": torch.tensor(0.0, device=pred_boxes_2d.device),
            }

        target_boxes_3d_encoded_list = []
        weights_3d_list = []

        # Validate image_size is provided - required for correct box_coder.encode()
        if image_size is None:
            raise ValueError(
                "image_size is required for _loss_boxes_3d. "
                "box_coder.encode() expects pixel coordinates because "
                "project_points() returns pixel coords and "
                "delta_center = projected_3d_center - 2d_box_center (both in pixels)."
            )

        H, W = image_size
        factors = target_boxes_2d.new_tensor([W, H, W, H])

        for i in range(len(batch_idx)):
            single_box_3d = target_boxes_3d[i:i+1]

            # Skip entries with invalid (all-zero) 3D boxes: set weight=0
            # so they don't contribute to 3D loss. This handles the case
            # where GT has a valid 2D box but no 3D annotation.
            if single_box_3d.abs().sum() < 1e-6:
                reg_dims = pred_boxes_3d.shape[-1]
                target_boxes_3d_encoded_list.append(
                    torch.zeros(1, reg_dims, device=pred_boxes_3d.device)
                )
                weights_3d_list.append(
                    torch.zeros(1, reg_dims, device=pred_boxes_3d.device)
                )
                continue

            # Use GT 2D box (normalized xyxy) and convert to pixel
            single_gt_box_2d = target_boxes_2d[i:i+1]
            single_gt_box_2d_pixel = single_gt_box_2d * factors

            single_intrinsic = intrinsics[batch_idx[i]]  # (3, 3)

            encoded, weights = self.box_coder.encode(
                single_gt_box_2d_pixel, single_box_3d, single_intrinsic,
            )
            target_boxes_3d_encoded_list.append(encoded)
            weights_3d_list.append(weights)

        target_boxes_3d_encoded = torch.cat(target_boxes_3d_encoded_list, dim=0)
        weights_3d = torch.cat(weights_3d_list, dim=0)

        losses = {}

        # Delta 2D center loss
        loss_delta_2d = l1_loss(
            src_boxes_3d[:, :2],
            target_boxes_3d_encoded[:, :2],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, :2], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_delta_2d"] = loss_delta_2d * self.config.loss_delta_2d_weight

        # Depth loss
        loss_depth = l1_loss(
            src_boxes_3d[:, 2],
            target_boxes_3d_encoded[:, 2],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 2], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_depth"] = loss_depth * self.config.loss_depth_weight

        # Dimension loss
        loss_dim = l1_loss(
            src_boxes_3d[:, 3:6],
            target_boxes_3d_encoded[:, 3:6],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 3:6], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_dim"] = loss_dim * self.config.loss_dim_weight

        # Rotation loss
        loss_rot = l1_loss(
            src_boxes_3d[:, 6:],
            target_boxes_3d_encoded[:, 6:],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 6:], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_rot"] = loss_rot * self.config.loss_rot_weight

        return losses

    def _loss_3d_classification(
        self,
        pred_conf_3d: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        pred_boxes_3d: Tensor,  # (B, S, 12) encoded
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        intrinsics: Tensor,  # (N_prompts, 3, 3)
        num_boxes: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Compute 3D confidence loss (positive + negative).

        Positive: soft target = quality (0.7*iou_3d + 0.3*depth)
        Negative: target = 0, with focal weighting
        Same structure as 2D cls loss (IABCEMdetr).

        At inference: final_score = 2d_score + 0.5 * 3d_score
        """
        batch_idx, src_idx, tgt_idx = indices
        B, S, _ = pred_conf_3d.shape
        device = pred_conf_3d.device
        M = len(batch_idx)

        if M == 0:
            return pred_conf_3d.sum() * 0.0

        prob = pred_conf_3d.sigmoid()
        target_classes = torch.zeros(B, S, 1, device=device)
        target_classes[(batch_idx, src_idx)] = 1.0

        with torch.no_grad():
            # 1. Depth quality - directly from encoded params, no decode needed
            src_boxes_3d = pred_boxes_3d[(batch_idx, src_idx)]
            target_boxes_3d_raw = (
                targets["boxes_3d"][tgt_idx] if tgt_idx is not None
                else targets["boxes_3d"]
            )
            depth_scale = self.box_coder.depth_scale
            pred_log_z = src_boxes_3d[:, 2] / depth_scale  # = log(pred_z)
            gt_z = target_boxes_3d_raw[:, 2].clamp(min=0.1)
            gt_log_z = torch.log(gt_z)
            depth_quality = torch.exp(-torch.abs(pred_log_z - gt_log_z))
            depth_quality = torch.nan_to_num(depth_quality, nan=0.0, posinf=1.0, neginf=0.0)

            # 2. 3D IoU using safe shapely-based implementation
            #    (CPU, full rotation support, never crashes)
            from wilddet3d.ops.iou_3d_safe import batch_box3d_iou

            H, W = image_size
            factors = pred_boxes_2d.new_tensor([[W, H, W, H]])
            src_boxes_2d_pixel = pred_boxes_2d[(batch_idx, src_idx)] * factors

            pred_decoded_list = []
            for i in range(M):
                single_decoded = self.box_coder.decode(
                    src_boxes_2d_pixel[i:i+1],
                    src_boxes_3d[i:i+1],
                    intrinsics[batch_idx[i]],
                )
                pred_decoded_list.append(single_decoded)
            pred_decoded = torch.cat(pred_decoded_list, dim=0)  # (M, 10)

            iou_3d = batch_box3d_iou(pred_decoded, target_boxes_3d_raw[:, :10])

            # 3. Combined quality
            quality = (
                self.config.conf_depth_weight * depth_quality
                + self.config.conf_iou_3d_weight * iou_3d
            )
            quality = torch.nan_to_num(quality, nan=0.0).clamp(0.0, 1.0)

            # 4. Build soft target (same as 2D IABCEMdetr pattern)
            t = (
                prob[(batch_idx, src_idx)].squeeze(-1) ** self.config.alpha
                * quality ** (1 - self.config.alpha)
            )
            t = t.clamp(min=0.01).detach()

            positive_target = target_classes.clone()
            positive_target[(batch_idx, src_idx)] = t.unsqueeze(-1)

        # Positive loss with soft quality target
        loss_pos = F.binary_cross_entropy_with_logits(
            pred_conf_3d, positive_target, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.pos_weight

        # Negative loss with focal weighting (push unmatched queries toward 0)
        loss_neg = F.binary_cross_entropy_with_logits(
            pred_conf_3d, target_classes, reduction="none"
        )
        loss_neg = loss_neg * (1 - target_classes) * (prob ** self.config.gamma)

        # Suppress negative loss for predictions overlapping ignore boxes
        if "ignore_neg_mask" in targets:
            neg_suppress = targets["ignore_neg_mask"].unsqueeze(-1)
            loss_neg = loss_neg * (1 - neg_suppress)

        loss_bce = loss_pos + loss_neg

        # Apply presence mask (zero out loss for prompts with no GT)
        if self.config.use_presence:
            num_gts = targets.get(
                "num_boxes", torch.zeros(B, dtype=torch.long, device=device)
            )
            keep_loss = (num_gts > 0).float().view(B, 1, 1)  # (B, 1, 1) for (B, S, 1) broadcasting
            loss_bce = loss_bce * keep_loss

        return loss_bce.mean()

    def _compute_aux_loss(
        self,
        aux_out: dict,
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        num_boxes: Tensor,
        intrinsics: Tensor | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses for auxiliary decoder outputs.

        Following GDino3D's design, we compute all losses (2D + 3D) for auxiliary outputs
        to enable full deep supervision across all decoder layers.

        Args:
            aux_out: Auxiliary output dictionary containing pred_logits, pred_boxes_2d, pred_boxes_3d
            indices: Matching indices from matcher
            targets: Ground truth targets
            num_boxes: Number of boxes for normalization
            intrinsics: Camera intrinsics for 3D loss computation
            image_size: (H, W) for pixel coordinate conversion

        Returns:
            Dictionary of auxiliary losses
        """
        losses = {}

        # Build SAM3-format outputs for aux layer
        sam3_aux = {
            "pred_logits": aux_out.get("pred_logits"),
            "pred_boxes_xyxy": aux_out.get(
                "pred_boxes_xyxy", aux_out.get("pred_boxes_2d")
            ),
            "pred_boxes": aux_out.get("pred_boxes"),
        }
        # If pred_boxes (cxcywh) not available, convert from xyxy
        if sam3_aux["pred_boxes"] is None and sam3_aux["pred_boxes_xyxy"] is not None:
            sam3_aux["pred_boxes"] = self._xyxy_to_cxcywh(
                sam3_aux["pred_boxes_xyxy"]
            )

        # Recompute ignore mask for this aux layer's predicted boxes
        if "_ignore_boxes2d" in targets and sam3_aux["pred_boxes_xyxy"] is not None:
            targets = targets.copy()
            targets["ignore_neg_mask"] = self._compute_ignore_neg_mask(
                sam3_aux["pred_boxes_xyxy"],
                targets["_ignore_boxes2d"],
                targets["_num_ignores"],
                threshold=self.config.ignore_iou_threshold,
            )

        # Classification loss via SAM3's IABCEMdetr (scaled by loss_2d_scale)
        if sam3_aux["pred_logits"] is not None:
            cls_losses = self.cls_loss.get_loss(
                sam3_aux, targets, indices, num_boxes
            )
            losses["loss_cls"] = (
                self.config.loss_2d_scale
                * cls_losses["loss_ce"]
                * self.config.loss_cls_weight
            )

        # 2D box losses via SAM3's Boxes class (scaled by loss_2d_scale)
        if sam3_aux["pred_boxes"] is not None:
            box_losses = self.box_loss.get_loss(
                sam3_aux, targets, indices, num_boxes
            )
            losses["loss_bbox"] = (
                self.config.loss_2d_scale
                * box_losses["loss_bbox"]
                * self.config.loss_bbox_weight
            )
            losses["loss_giou"] = (
                self.config.loss_2d_scale
                * box_losses["loss_giou"]
                * self.config.loss_giou_weight
            )

        # 3D box loss (our own, scaled by loss_3d_scale)
        pred_boxes_2d_aux = aux_out.get(
            "pred_boxes_2d", aux_out.get("pred_boxes_xyxy")
        )
        if "pred_boxes_3d" in aux_out and intrinsics is not None:
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d_aux,
                aux_out["pred_boxes_3d"],
                indices,
                targets,
                intrinsics,
                num_boxes,
                image_size=image_size,
            )
            for key, value in loss_3d.items():
                losses[key] = self.config.loss_3d_scale * value

        # 3D confidence loss (deep supervision)
        if (self.config.use_3d_conf
                and "pred_conf_3d" in aux_out
                and "pred_boxes_3d" in aux_out
                and intrinsics is not None):
            loss_3d_cls = self._loss_3d_classification(
                aux_out["pred_conf_3d"],
                pred_boxes_2d_aux,
                aux_out["pred_boxes_3d"],
                indices, targets, intrinsics, num_boxes, image_size,
            )
            losses["loss_3d_cls"] = self.config.loss_3d_conf_weight * loss_3d_cls

        return losses

    def _normalize_targets(self, targets: dict) -> dict:
        """Ensure targets are in expected format for loss computation.

        Note: WildDet3D collator always outputs GT boxes in normalized [0, 1] xyxy format.
        This function simply ensures the classes tensor exists (for binary classification).

        Args:
            targets: Dictionary containing ground truth data
                - boxes_xyxy: (N, 4) boxes in normalized xyxy [0, 1] format
                - classes: (N,) class labels (all ones for SAM3)
                - num_boxes: (N,) number of boxes per prompt (always 1)
                - boxes_3d: (N, 12) 3D boxes (optional)

        Returns:
            Targets dict with classes tensor guaranteed to exist
        """
        normalized = targets.copy()
        boxes_xyxy = targets["boxes_xyxy"]

        # Ensure classes tensor exists (all ones for binary classification)
        if "classes" not in normalized:
            num_boxes = boxes_xyxy.shape[0]
            normalized["classes"] = torch.ones(
                num_boxes, dtype=torch.long, device=boxes_xyxy.device
            )

        return normalized

    def _xyxy_to_cxcywh(self, boxes_xyxy: Tensor) -> Tensor:
        """Convert boxes from xyxy to cxcywh format.

        Args:
            boxes_xyxy: (N, 4) boxes in xyxy format

        Returns:
            boxes_cxcywh: (N, 4) boxes in cxcywh format
        """
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

