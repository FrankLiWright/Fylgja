"""BlazeFace（face_detector.tflite）解码：SSD 锚框 + TensorsToDetections + 加权 NMS。

参数来自 MediaPipe 源 face_detection.pbtxt（short-range 档）：
  SsdAnchorsCalculator: num_layers=4, strides=[8,16,16,16], min_scale=0.1484375,
  max_scale=0.75, anchor_offset=0.5, aspect_ratios=[1.0], fixed_anchor_size=true,
  interpolated_scale_aspect_ratio>0（每层 2 框：插值框 + 尺度框 → 896 锚框）
  TensorsToDetections: num_classes=1, num_coords=16, box_coord_offset=0,
  keypoint_coord_offset=4, num_keypoints=6, num_values_per_keypoint=2,
  sigmoid_score=true, score_clipping_thresh=100, reverse_output_order=true
  NMS: IoU 0.3, WEIGHTED
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INPUT_SIZE = 128
NUM_LAYERS = 4
STRIDES = (8, 16, 16, 16)
MIN_SCALE = 0.1484375
MAX_SCALE = 0.75
ANCHOR_OFFSET = 0.5
FIXED_ANCHOR_SIZE = True
NUM_COORDS = 16
NUM_KEYPOINTS = 6
SCORE_CLIP = 100.0
NMS_IOU = 0.3


def generate_anchors() -> np.ndarray:
    """返回 (896, 4)：x_center, y_center, w, h（输入归一化 [0,1]）。

    严格复刻 ssd_anchors_calculator.cc GenerateAnchors：
    - 同 stride 层合并为一组（strides [8,16,16,16] → 组1: stride8/16×16图/2框，组2: stride16/8×8图/6框）
    - fixed_anchor_size=true：全部锚框 w=h=1.0（scale 只决定解码偏移基准？否——直接 1.0）
    """
    boxes = []
    layer_id = 0
    while layer_id < NUM_LAYERS:
        # 合并同 stride 的层
        last = layer_id
        scales = []
        while last < NUM_LAYERS and STRIDES[last] == STRIDES[layer_id]:
            scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * last / (NUM_LAYERS - 1)
            scales.append(scale)
            if last == NUM_LAYERS - 1:
                scale_next = 1.0
            else:
                scale_next = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (last + 1) / (NUM_LAYERS - 1)
            scales.append(float(np.sqrt(scale * scale_next)))  # 插值框
            last += 1
        stride = STRIDES[layer_id]
        fm = INPUT_SIZE // stride
        for y in range(fm):
            for x in range(fm):
                cx = (x + ANCHOR_OFFSET) / fm
                cy = (y + ANCHOR_OFFSET) / fm
                for _ in scales:  # 组内锚框 id 顺序：层序 × [尺度框, 插值框]
                    boxes.append((cx, cy, 1.0, 1.0))  # fixed_anchor_size
        layer_id = last
    out = np.asarray(boxes, dtype=np.float64)
    assert out.shape == (896, 4), out.shape
    return out


@dataclass
class Detection:
    score: float
    box: np.ndarray        # (4,) x_center, y_center, w, h（输入归一化）
    keypoints: np.ndarray  # (6, 2) x, y（输入归一化）


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -SCORE_CLIP, SCORE_CLIP)))


def decode_detections(regressors: np.ndarray, classificators: np.ndarray,
                      anchors: np.ndarray, min_score: float = 0.5,
                      x_scale: float = 128.0, wh_scale: float = 128.0) -> list[Detection]:
    """解码 (896,16)+(896,1) → Detection 列表（未 NMS）。

    reverse_output_order=true：box 为 [y_center, x_center, h, w]，
    关键点为 (x,y) 对（实测与引擎 C 标准 BlazeFace 顺序 [右眼,左眼,鼻,嘴,右耳,左耳] 吻合）。
    x_scale/wh_scale：raw → 归一化偏移的缩放（输入像素尺度 128，empiric 已校准）。
    """
    reg = np.asarray(regressors, dtype=np.float64).reshape(-1, NUM_COORDS)
    cls = np.asarray(classificators, dtype=np.float64).reshape(-1)
    scores = _sigmoid(cls)
    keep = np.nonzero(scores >= min_score)[0]

    dets: list[Detection] = []
    for i in keep:
        r = reg[i]
        a = anchors[i]
        # reverse_output_order: y, x, h, w
        yc = r[0] / wh_scale * a[3] + a[1]
        xc = r[1] / wh_scale * a[2] + a[0]
        h = r[2] / wh_scale * a[3]
        w = r[3] / wh_scale * a[2]
        kps = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
        for k in range(NUM_KEYPOINTS):
            kx = r[4 + 2 * k] / x_scale * a[2] + a[0]
            ky = r[5 + 2 * k] / x_scale * a[3] + a[1]
            kps[k] = (kx, ky)
        dets.append(Detection(float(scores[i]),
                              np.array([xc, yc, w, h]), kps))
    return dets


def nms_weighted(dets: list[Detection], iou_thresh: float = NMS_IOU) -> list[Detection]:
    """NonMaxSuppressionCalculator：WEIGHTED、INTERSECTION_OVER_UNION。"""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    suppressed = [False] * len(dets)
    out: list[Detection] = []

    def iou(a: Detection, b: Detection) -> float:
        ax1, ay1 = a.box[0] - a.box[2] / 2, a.box[1] - a.box[3] / 2
        ax2, ay2 = a.box[0] + a.box[2] / 2, a.box[1] + a.box[3] / 2
        bx1, by1 = b.box[0] - b.box[2] / 2, b.box[1] - b.box[3] / 2
        bx2, by2 = b.box[0] + b.box[2] / 2, b.box[1] + b.box[3] / 2
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = a.box[2] * a.box[3] + b.box[2] * b.box[3] - inter
        return inter / union if union > 0 else 0.0

    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        # WEIGHTED NMS：与更高分框重叠的框按分数加权并入（按分数序遍历，非原始索引）
        merged_box = dets[i].box.copy()
        merged_kps = dets[i].keypoints.copy()
        total_w = dets[i].score
        for j in order[pos + 1:]:
            if suppressed[j]:
                continue
            if iou(dets[i], dets[j]) >= iou_thresh:
                w = dets[j].score
                merged_box = (merged_box * total_w + dets[j].box * w) / (total_w + w)
                merged_kps = (merged_kps * total_w + dets[j].keypoints * w) / (total_w + w)
                total_w += w
                suppressed[j] = True
        out.append(Detection(dets[i].score, merged_box, merged_kps))
    return out
