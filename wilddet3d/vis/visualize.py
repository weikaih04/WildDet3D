"""WildDet3D visualization utilities.

Anti-aliased 3D bounding boxes with Manrope font score labels.
Uses vis4d's preprocess_boxes3d for correct 3D corner projection,
cv2 LINE_AA for smooth lines, PIL + Manrope for text rendering.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from vis4d.common.array import array_to_numpy
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners
from vis4d.vis.util import generate_color_map

_FONT_DIR = Path(__file__).parent / "fonts"

# vis4d edge order (from PillowCanvasBackend.draw_box_3d)
# Front face: 0-1-5-4, Back face: 2-3-7-6, Sides: 0-2, 1-3, 4-6, 5-7
_EDGES = [
    # Front
    (0, 1), (1, 5), (5, 4), (4, 0),
    # Sides
    (0, 2), (1, 3), (4, 6), (5, 7),
    # Back
    (2, 3), (3, 7), (7, 6), (6, 2),
]


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Get Manrope Bold font with fallbacks."""
    for path in [
        _FONT_DIR / "Manrope-Bold.ttf",
        _FONT_DIR / "Manrope-SemiBold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _project_pt_simple(pt_3d, K_np):
    """Project single 3D point to 2D using intrinsics (no torch overhead)."""
    x, y, z = pt_3d
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return float(u), float(v)


def _clip_to_near(p1, p2, near=0.15):
    """Clip line to near plane, return clipped point."""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    k_up = abs(z1 - near)
    k_down = abs(z1 - z2)
    k = min(k_up / k_down, 1.0) if k_down > 0 else 1.0
    return ((1 - k) * x1 + k * x2, (1 - k) * y1 + k * y2, near)


def draw_3d_boxes(
    image: np.ndarray,
    boxes3d: Tensor | np.ndarray,
    intrinsics: np.ndarray,
    scores_2d: Tensor | np.ndarray | None = None,
    scores_3d: Tensor | np.ndarray | None = None,
    class_ids: Tensor | np.ndarray | None = None,
    class_names: list[str] | None = None,
    line_width: int = 2,
    font_size: int = 13,
    n_colors: int = 50,
    score_format: str = "{name} 2D:{s2d:.2f} 3D:{s3d:.2f}",
    near_clip: float = 0.15,
    score_2d_threshold: float = 0.3,
    score_3d_threshold: float = 0.1,
    save_path: str | None = None,
    boxes_2d: Tensor | np.ndarray | None = None,
    draw_predicted_2d_boxes: bool = False,
    input_boxes: list[list[float]] | None = None,
    input_points: list[list[tuple[float, float, int]]] | None = None,
    draw_prompt: bool = False,
) -> Image.Image:
    """Draw anti-aliased 3D bounding boxes with 2D/3D score labels.

    Args:
        image: RGB image (H, W, 3) uint8.
        boxes3d: 3D boxes (N, 10) in OPENCV camera coordinates.
        intrinsics: Camera intrinsics (3, 3).
        scores_2d: 2D confidence scores (N,).
        scores_3d: 3D confidence scores (N,).
        class_ids: Class indices (N,).
        class_names: List of class names.
        line_width: Width of 3D box edges.
        font_size: Font size for labels.
        n_colors: Number of colors in palette.
        score_format: Format string. Available: {name}, {s2d}, {s3d}.
        near_clip: Camera near clipping plane.
        score_2d_threshold: Only draw boxes with 2D score >= this
            (default 0.3; set 0.0 to disable). Requires scores_2d.
        score_3d_threshold: Only draw boxes with 3D score >= this
            (default 0.1; set 0.0 to disable). Requires scores_3d.
        save_path: If provided, save the result.
        boxes_2d: Model-predicted 2D boxes (N, 4) in pixel xyxy; drawn
            in green when draw_predicted_2d_boxes=True.
        draw_predicted_2d_boxes: Overlay predicted 2D boxes. Default False.
        input_boxes: User prompt boxes (list of [x1,y1,x2,y2] in
            original-image pixels); drawn in red when draw_prompt=True.
        input_points: User prompt points (list of list of
            (x, y, label)); drawn in red (positive) / gray (negative)
            when draw_prompt=True.
        draw_prompt: Overlay user prompt boxes / points. Default False.

    Returns:
        PIL Image with drawn boxes and score labels.
    """
    if isinstance(image, Tensor):
        image = image.cpu().numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if isinstance(boxes3d, Tensor):
        boxes3d_t = boxes3d.cpu().float()
    else:
        boxes3d_t = torch.tensor(boxes3d, dtype=torch.float32)
    if isinstance(scores_2d, Tensor):
        scores_2d = scores_2d.cpu().numpy()
    if isinstance(scores_3d, Tensor):
        scores_3d = scores_3d.cpu().numpy()
    if isinstance(class_ids, Tensor):
        class_ids = class_ids.cpu().numpy()
    boxes_2d_np = None
    if boxes_2d is not None:
        boxes_2d_np = (
            boxes_2d.cpu().numpy()
            if isinstance(boxes_2d, Tensor)
            else np.asarray(boxes_2d)
        )

    # Drop boxes below the 2D / 3D score floors.
    keep = np.ones(len(boxes3d_t), dtype=bool)
    if scores_2d is not None and score_2d_threshold > 0:
        keep &= np.asarray(scores_2d) >= score_2d_threshold
    if scores_3d is not None and score_3d_threshold > 0:
        keep &= np.asarray(scores_3d) >= score_3d_threshold
    if not keep.all():
        keep_t = torch.from_numpy(keep)
        boxes3d_t = boxes3d_t[keep_t]
        if scores_2d is not None:
            scores_2d = np.asarray(scores_2d)[keep]
        if scores_3d is not None:
            scores_3d = np.asarray(scores_3d)[keep]
        if class_ids is not None:
            class_ids = np.asarray(class_ids)[keep]
        if boxes_2d_np is not None:
            boxes_2d_np = boxes_2d_np[keep]

    N = len(boxes3d_t)
    H, W = image.shape[:2]
    K_np = intrinsics.astype(np.float32)

    if N == 0:
        pil_img = Image.fromarray(image)
        if save_path:
            pil_img.save(save_path, quality=95)
        return pil_img

    # Get 3D corners (N, 8, 3) using vis4d's OPENCV convention
    corners_3d = boxes3d_to_corners(boxes3d_t, AxisMode.OPENCV).numpy()

    color_map = generate_color_map(n_colors)

    # --- Draw lines with cv2 (anti-aliased) ---
    canvas = image.copy()
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for i in range(N):
        cid = int(class_ids[i]) if class_ids is not None else i
        color_rgb = color_map[cid % len(color_map)]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        corners = corners_3d[i]  # (8, 3)

        for e0, e1 in _EDGES:
            p1 = tuple(corners[e0].tolist())
            p2 = tuple(corners[e1].tolist())

            # Near-plane clipping
            if p1[2] < near_clip and p2[2] < near_clip:
                continue
            if p1[2] < near_clip:
                p1 = _clip_to_near(p1, p2, near_clip)
            elif p2[2] < near_clip:
                p2 = _clip_to_near(p2, p1, near_clip)

            # Project to 2D
            u1, v1 = _project_pt_simple(p1, K_np)
            u2, v2 = _project_pt_simple(p2, K_np)

            # Skip if way outside image
            margin = max(W, H)
            if (abs(u1) > margin * 2 or abs(v1) > margin * 2 or
                abs(u2) > margin * 2 or abs(v2) > margin * 2):
                continue

            cv2.line(
                canvas_bgr,
                (int(round(u1)), int(round(v1))),
                (int(round(u2)), int(round(v2))),
                color_bgr,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    # --- Draw text labels with PIL (Manrope font) ---
    # Use RGBA for rounded rectangle with alpha
    pil_img = Image.fromarray(canvas_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)

    for i in range(N):
        cid = int(class_ids[i]) if class_ids is not None else 0
        color = color_map[cid % len(color_map)]

        # Project center to 2D
        center_3d = boxes3d_t[i, :3].numpy()
        if center_3d[2] < near_clip:
            continue
        cx, cy = _project_pt_simple(tuple(center_3d.tolist()), K_np)
        if cx < -50 or cx >= W + 50 or cy < -50 or cy >= H + 50:
            continue

        name = class_names[cid] if class_names is not None else str(cid)
        s2d = float(scores_2d[i]) if scores_2d is not None else 0.0
        s3d = float(scores_3d[i]) if scores_3d is not None else 0.0
        label = score_format.format(name=name, s2d=s2d, s3d=s3d)

        # Measure text size (textbbox returns actual glyph bounds)
        left, top, right, bottom = draw_main.textbbox((0, 0), label, font=font)
        tw = right - left
        th = bottom - top
        y_offset = top  # font ascent offset (glyphs don't start at y=0)

        # Position: inside the box, near the projected center
        pad_x, pad_y = 6, 4
        radius = 5

        # Place label centered at projected center
        rx0 = cx - tw / 2 - pad_x
        ry0 = cy - th / 2 - pad_y
        rx1 = cx + tw / 2 + pad_x
        ry1 = cy + th / 2 + pad_y

        # Clamp to image bounds
        if rx0 < 2:
            shift = 2 - rx0
            rx0 += shift
            rx1 += shift
        if rx1 > W - 2:
            shift = rx1 - (W - 2)
            rx0 -= shift
            rx1 -= shift
        if ry0 < 2:
            shift = 2 - ry0
            ry0 += shift
            ry1 += shift
        if ry1 > H - 2:
            shift = ry1 - (H - 2)
            ry0 -= shift
            ry1 -= shift

        # Draw rounded rectangle on overlay (semi-transparent)
        fill_color = tuple(color) + (210,)
        draw_overlay.rounded_rectangle(
            [rx0, ry0, rx1, ry1],
            radius=radius,
            fill=fill_color,
        )

        # Text centered in the rounded rect (compensate font ascent offset)
        text_x = rx0 + pad_x - left
        text_y = ry0 + pad_y - y_offset
        draw_overlay.text(
            (text_x, text_y),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    # Composite overlay onto main image
    pil_img = Image.alpha_composite(pil_img, overlay).convert("RGB")

    # Optional overlays.
    if (draw_predicted_2d_boxes and boxes_2d_np is not None) or (
        draw_prompt and (input_boxes is not None or input_points is not None)
    ):
        annot = ImageDraw.Draw(pil_img)
        if draw_predicted_2d_boxes and boxes_2d_np is not None:
            for box in boxes_2d_np.tolist():
                annot.rectangle(list(box), outline=(0, 200, 0), width=2)
        if draw_prompt and input_boxes is not None:
            for box in input_boxes:
                annot.rectangle(list(box), outline=(255, 0, 0), width=3)
        if draw_prompt and input_points is not None:
            for pts in input_points:
                for (px, py, lbl) in pts:
                    color = (255, 0, 0) if lbl == 1 else (128, 128, 128)
                    annot.ellipse(
                        [px - 6, py - 6, px + 6, py + 6],
                        outline=color, fill=color, width=2,
                    )

    if save_path:
        pil_img.save(save_path, quality=95)

    return pil_img
