"""模型资产：从 face_landmarker.task 解包子模型，解析几何元数据 binarypb。

方案书 4.1/4.3 第 3 条：geometry_pipeline_metadata_landmarks.binarypb 为标准 protobuf，
含规范人脸三维点（厘米单位，VERTEX_PT 顶点 5 浮点：XYZ+UV）与 procrustes 加权索引。
本模块用手写 wire 级解析器提取（不引入 protobuf 运行时依赖，方案书 11.2 降低打包面）。
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import Attribution, SelfCheckError

SUBMODELS = ("face_detector.tflite", "face_landmarks_detector.tflite",
             "geometry_pipeline_metadata_landmarks.binarypb")


@dataclass
class GeometryMetadata:
    canonical_vertices: np.ndarray   # (N, 3) 度量空间厘米
    landmark_ids: np.ndarray         # (K,) procrustes 基集的关键点 ID
    weights: np.ndarray              # (K,) 权重


def extract_assets(task_path: Path | str, dest: Path | str) -> dict[str, Path]:
    """从 .task（zip）解出子模型与几何元数据；已存在且非空则复用。"""
    task_path, dest = Path(task_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(task_path) as z:
            names = {Path(n).name: n for n in z.namelist()}
            for want in SUBMODELS:
                target = dest / want
                if not target.exists() or target.stat().st_size == 0:
                    if want not in names:
                        raise SelfCheckError(
                            Attribution.PACKAGE,
                            f"模型包缺少 {want}（安装包不完整）",
                        )
                    target.write_bytes(z.read(names[want]))
                out[want] = target
    except zipfile.BadZipFile as e:
        raise SelfCheckError(Attribution.PACKAGE, f"模型包损坏：{e}") from e
    return out


# ------------------------------------------------------------- protobuf wire 解析

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    v = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7


def _iter_fields(buf: bytes):
    """逐字段 (field_no, wire_type, payload)。payload: varint→int, ld→bytes, f32→float。"""
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field_no, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(buf, i)
            yield field_no, wire, v
        elif wire == 2:
            ln, i = _varint(buf, i)
            yield field_no, wire, buf[i:i + ln]
            i += ln
        elif wire == 5:
            yield field_no, wire, struct.unpack_from("<f", buf, i)[0]
            i += 4
        elif wire == 1:
            yield field_no, wire, struct.unpack_from("<d", buf, i)[0]
            i += 8
        else:
            raise ValueError(f"未知 wire type {wire}")


def _packed_floats(payload: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def parse_geometry_metadata(data: bytes) -> GeometryMetadata:
    """GeometryPipelineMetadata：canonical_mesh=1（Mesh3d：vertex_buffer=3，5 浮点/顶点）、
    procrustes_landmark_basis=2（WeightedLandmarkRef：landmark_id=1，weight=2）。"""
    verts: list[float] = []
    ids: list[int] = []
    weights: list[float] = []

    for fno, wire, payload in _iter_fields(data):
        if fno == 1 and wire == 2:  # canonical_mesh (Mesh3d)
            for mno, mwire, mpayload in _iter_fields(payload):
                if mno == 3:  # vertex_buffer
                    if mwire == 2:  # packed
                        verts.extend(_packed_floats(mpayload))
                    elif mwire == 5:
                        verts.append(mpayload)
        elif fno == 2 and wire == 2:  # procrustes_landmark_ref
            lid, wt = 0, 0.0
            for wno, wwire, wpayload in _iter_fields(payload):
                if wno == 1 and wwire == 0:
                    lid = wpayload
                elif wno == 2 and wwire == 5:
                    wt = wpayload
            ids.append(lid)
            weights.append(wt)

    if not verts:
        raise SelfCheckError(Attribution.PACKAGE, "几何元数据无顶点数据（包损坏）")
    arr = np.asarray(verts, dtype=np.float64)
    # VERTEX_PT：5 浮点/顶点（XYZ + UV）；若整除 5 且 UV 列可识别则取前三列
    if arr.size % 5 == 0:
        v = arr.reshape(-1, 5)[:, :3]
    elif arr.size % 3 == 0:
        v = arr.reshape(-1, 3)
    else:
        raise SelfCheckError(Attribution.PACKAGE, f"几何元数据顶点数异常：{arr.size}")
    return GeometryMetadata(
        canonical_vertices=v,
        landmark_ids=np.asarray(ids, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
    )
