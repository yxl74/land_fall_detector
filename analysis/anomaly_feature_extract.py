#!/usr/bin/env python3
"""
Extract anomaly-focused features (bytes + structural + LandFall-specific extras).

This script is separate from the supervised pipeline and is intended for
benign-only anomaly experiments (autoencoder / Deep SVDD).
"""

from __future__ import annotations

import argparse
import csv
import os
import mmap
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


TIFF_TYPES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}

TAG_WIDTH = 256
TAG_HEIGHT = 257
TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_DNG_VERSION = 50706
TAG_NEW_SUBFILE_TYPE = 254
TAG_OPCODE_LIST1 = 51008
TAG_OPCODE_LIST2 = 51009
TAG_OPCODE_LIST3 = 51022


def get_label_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "benign_data" in parts:
        return "benign"
    if "LandFall" in parts:
        return "landfall"
    if "general_mal" in parts:
        return "general_mal"
    return "unknown"


def is_whitespace(b: int) -> bool:
    return b in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20)


def strip_prefix(data: bytes) -> bytes:
    i = 0
    while i < len(data) and is_whitespace(data[i]):
        i += 1
    return data[i:]


def strip_suffix(data: bytes) -> bytes:
    i = len(data)
    while i > 0 and is_whitespace(data[i - 1]):
        i -= 1
    return data[:i]


def magika_like_bytes(
    path: str,
    beg_size: int = 1024,
    end_size: int = 1024,
    block_size: int = 4096,
    padding_token: int = 256,
) -> np.ndarray:
    file_size = os.path.getsize(path)
    if file_size == 0:
        return np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    buffer_size = min(block_size, file_size)
    with open(path, "rb") as f:
        beg_block = f.read(buffer_size)
        beg = strip_prefix(beg_block)
        if file_size >= buffer_size:
            f.seek(max(0, file_size - buffer_size))
        end_block = f.read(buffer_size)
        end = strip_suffix(end_block)

    features = np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    beg_len = min(len(beg), beg_size)
    if beg_len:
        features[:beg_len] = np.frombuffer(beg[:beg_len], dtype=np.uint8).astype(np.int16)

    end_len = min(len(end), end_size)
    if end_len:
        features[beg_size : beg_size + end_len] = np.frombuffer(
            end[-end_len:], dtype=np.uint8
        ).astype(np.int16)

    return features


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _read_u16(buf: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off + 2 > len(buf):
        return None
    fmt = "<H" if endian == "II" else ">H"
    return struct.unpack_from(fmt, buf, off)[0]


def _read_u32(buf: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off + 4 > len(buf):
        return None
    fmt = "<I" if endian == "II" else ">I"
    return struct.unpack_from(fmt, buf, off)[0]


def _read_u32be(buf: mmap.mmap, off: int) -> Optional[int]:
    if off + 4 > len(buf):
        return None
    return struct.unpack_from(">I", buf, off)[0]


def _read_values(
    buf: mmap.mmap,
    endian: str,
    file_size: int,
    type_id: int,
    count: int,
    value_or_offset: int,
) -> List[int]:
    if count <= 0:
        return []
    size_bytes = TIFF_TYPES.get(type_id, 1) * count
    if size_bytes <= 4:
        raw = value_or_offset.to_bytes(4, "little" if endian == "II" else "big")
        if type_id == 3:
            fmt = "<H" if endian == "II" else ">H"
            out = []
            for i in range(count):
                out.append(struct.unpack_from(fmt, raw, i * 2)[0])
            return out
        if type_id == 4:
            fmt = "<I" if endian == "II" else ">I"
            return [struct.unpack_from(fmt, raw, 0)[0]]
        return []
    if value_or_offset + size_bytes > file_size:
        return []
    if type_id == 3:
        fmt = ("<" if endian == "II" else ">") + ("H" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    if type_id == 4:
        fmt = ("<" if endian == "II" else ">") + ("I" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    return []


def parse_tiff_struct(path: str) -> Optional[Dict[str, int]]:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mm) < 8:
                return None
            endian = mm[0:2].decode("latin-1", errors="ignore")
            if endian not in ("II", "MM"):
                return None
            magic = _read_u16(mm, 2, endian)
            if magic != 42:
                return None
            root = _read_u32(mm, 4, endian)
            if root is None:
                return None

            widths: List[int] = []
            heights: List[int] = []
            ifd_entry_max = 0
            subifd_offsets: List[int] = []
            exif_offset = 0
            new_subfile_types: List[int] = []
            opcode_lists: Dict[int, Tuple[int, int]] = {}
            is_dng = 0

            visited = set()
            stack: List[int] = [root]

            def parse_stack() -> None:
                nonlocal ifd_entry_max, exif_offset, is_dng
                while stack:
                    off = stack.pop()
                    if off == 0 or off in visited or off >= file_size:
                        continue
                    visited.add(off)
                    count = _read_u16(mm, off, endian)
                    if count is None:
                        continue
                    ifd_entry_max = max(ifd_entry_max, count)
                    entry_base = off + 2
                    for i in range(count):
                        entry_off = entry_base + i * 12
                        if entry_off + 12 > file_size:
                            break
                        tag = _read_u16(mm, entry_off, endian)
                        type_id = _read_u16(mm, entry_off + 2, endian)
                        val_count = _read_u32(mm, entry_off + 4, endian)
                        value_or_offset = _read_u32(mm, entry_off + 8, endian)
                        if tag is None or type_id is None or val_count is None or value_or_offset is None:
                            continue
                        if tag == TAG_DNG_VERSION:
                            is_dng = 1
                        if tag in (TAG_WIDTH, TAG_HEIGHT, TAG_NEW_SUBFILE_TYPE):
                            vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                            if vals:
                                if tag == TAG_WIDTH:
                                    widths.extend(vals)
                                elif tag == TAG_HEIGHT:
                                    heights.extend(vals)
                                else:
                                    new_subfile_types.extend(vals)
                        if tag == TAG_SUBIFD:
                            size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                            if size_bytes <= 4:
                                subifd_offsets.append(value_or_offset)
                            else:
                                if value_or_offset + size_bytes <= file_size:
                                    for j in range(val_count):
                                        subifd_offsets.append(_read_u32(mm, value_or_offset + j * 4, endian) or 0)
                        if tag == TAG_EXIF_IFD:
                            exif_offset = value_or_offset
                        if tag in (TAG_OPCODE_LIST1, TAG_OPCODE_LIST2, TAG_OPCODE_LIST3):
                            size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                            opcode_lists[tag] = (value_or_offset, size_bytes)

                    next_ptr_off = entry_base + count * 12
                    next_ifd = _read_u32(mm, next_ptr_off, endian)
                    if next_ifd:
                        stack.append(next_ifd)

            parse_stack()
            for off in subifd_offsets:
                stack.append(off)
            if exif_offset:
                stack.append(exif_offset)
            parse_stack()

            min_width = min(widths) if widths else 0
            min_height = min(heights) if heights else 0
            max_width = max(widths) if widths else 0
            max_height = max(heights) if heights else 0
            total_pixels = max_width * max_height

            total_opcodes = 0
            unknown_opcodes = 0
            max_opcode_id = 0
            opcode_list1_bytes = 0
            opcode_list2_bytes = 0
            opcode_list3_bytes = 0

            for tag, (offset, size_bytes) in opcode_lists.items():
                if offset == 0 or offset + size_bytes > file_size or size_bytes < 4:
                    continue
                opcode_count = _read_u32be(mm, offset)
                if opcode_count is None:
                    continue
                pos = offset + 4
                parsed = 0
                while parsed < opcode_count and pos + 16 <= offset + size_bytes:
                    opcode_id = _read_u32be(mm, pos)
                    data_size = _read_u32be(mm, pos + 12)
                    if opcode_id is None or data_size is None:
                        break
                    pos += 16
                    if pos + data_size > offset + size_bytes:
                        break
                    parsed += 1
                    total_opcodes += 1
                    max_opcode_id = max(max_opcode_id, opcode_id)
                    if opcode_id > 14:
                        unknown_opcodes += 1
                    pos += data_size

                if tag == TAG_OPCODE_LIST1:
                    opcode_list1_bytes = size_bytes
                elif tag == TAG_OPCODE_LIST2:
                    opcode_list2_bytes = size_bytes
                elif tag == TAG_OPCODE_LIST3:
                    opcode_list3_bytes = size_bytes

            opcode_list_bytes_total = opcode_list1_bytes + opcode_list2_bytes + opcode_list3_bytes
            opcode_list_bytes_max = max(opcode_list1_bytes, opcode_list2_bytes, opcode_list3_bytes)
            opcode_list_present_count = int(opcode_list1_bytes > 0) + int(opcode_list2_bytes > 0) + int(opcode_list3_bytes > 0)
            opcode_bytes_ratio_permille = int(opcode_list_bytes_total * 1000 / file_size) if file_size > 0 else 0
            opcode_bytes_per_opcode_milli = (
                int(opcode_list_bytes_total * 1000 / total_opcodes) if total_opcodes > 0 else 0
            )
            unknown_opcode_ratio_permille = (
                int(unknown_opcodes * 1000 / total_opcodes) if total_opcodes > 0 else 0
            )

            bytes_per_pixel_milli = int(file_size * 1000 / total_pixels) if total_pixels > 0 else 0
            pixels_per_mb = int(total_pixels * 1_000_000 / file_size) if file_size > 0 else 0

            return {
                "is_tiff": 1,
                "is_dng": is_dng,
                "min_width": min_width,
                "min_height": min_height,
                "ifd_entry_max": ifd_entry_max,
                "subifd_count_sum": len(subifd_offsets),
                "new_subfile_types_unique": len(set(new_subfile_types)) if new_subfile_types else 0,
                "total_opcodes": total_opcodes,
                "unknown_opcodes": unknown_opcodes,
                "max_opcode_id": max_opcode_id,
                "opcode_list1_bytes": opcode_list1_bytes,
                "opcode_list2_bytes": opcode_list2_bytes,
                "opcode_list3_bytes": opcode_list3_bytes,
                "max_width": max_width,
                "max_height": max_height,
                "total_pixels": total_pixels,
                "file_size": file_size,
                "bytes_per_pixel_milli": bytes_per_pixel_milli,
                "pixels_per_mb": pixels_per_mb,
                "opcode_list_bytes_total": opcode_list_bytes_total,
                "opcode_list_bytes_max": opcode_list_bytes_max,
                "opcode_list_present_count": opcode_list_present_count,
                "opcode_bytes_ratio_permille": opcode_bytes_ratio_permille,
                "opcode_bytes_per_opcode_milli": opcode_bytes_per_opcode_milli,
                "unknown_opcode_ratio_permille": unknown_opcode_ratio_permille,
                "has_opcode_list1": int(opcode_list1_bytes > 0),
                "has_opcode_list2": int(opcode_list2_bytes > 0),
                "has_opcode_list3": int(opcode_list3_bytes > 0),
            }
        finally:
            mm.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--output-csv", default="outputs/anomaly_features.csv")
    args = parser.parse_args()

    paths: List[str] = []
    for sub in ("benign_data", "LandFall", "general_mal"):
        base = os.path.join(args.data_root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                paths.append(os.path.join(dirpath, fn))

    meta_rows: List[Dict[str, str]] = []
    byte_features: List[np.ndarray] = []
    struct_features: List[List[float]] = []
    labels: List[int] = []
    label_names: List[str] = []
    file_paths: List[str] = []

    struct_feature_names = [
        "is_tiff",
        "is_dng",
        "min_width",
        "min_height",
        "ifd_entry_max",
        "subifd_count_sum",
        "new_subfile_types_unique",
        "total_opcodes",
        "unknown_opcodes",
        "max_opcode_id",
        "opcode_list1_bytes",
        "opcode_list2_bytes",
        "opcode_list3_bytes",
        "max_width",
        "max_height",
        "total_pixels",
        "file_size",
        "bytes_per_pixel_milli",
        "pixels_per_mb",
        "opcode_list_bytes_total",
        "opcode_list_bytes_max",
        "opcode_list_present_count",
        "opcode_bytes_ratio_permille",
        "opcode_bytes_per_opcode_milli",
        "unknown_opcode_ratio_permille",
        "has_opcode_list1",
        "has_opcode_list2",
        "has_opcode_list3",
        "header_entropy",
        "tail_entropy",
        "overall_entropy",
        "entropy_gradient",
    ]

    for p in sorted(paths):
        label = get_label_from_path(p)
        y = 0 if label == "benign" else 1

        tiff_feat = parse_tiff_struct(p)
        if tiff_feat is None:
            tiff_feat = {name: 0 for name in struct_feature_names if name not in ("header_entropy", "tail_entropy", "overall_entropy", "entropy_gradient")}
            tiff_feat["is_tiff"] = 0
            tiff_feat["is_dng"] = 0

        file_size = os.path.getsize(p)
        header = b""
        tail = b""
        overall = b""
        with open(p, "rb") as f:
            header = f.read(min(4096, file_size))
            if file_size > 0:
                f.seek(max(0, file_size - min(4096, file_size)))
                tail = f.read(min(4096, file_size))
            f.seek(0)
            overall = f.read(min(65536, file_size))

        header_entropy = byte_entropy(header)
        tail_entropy = byte_entropy(tail)
        overall_entropy = byte_entropy(overall)
        entropy_gradient = abs(header_entropy - tail_entropy)

        struct_vec = [
            tiff_feat.get("is_tiff", 0),
            tiff_feat.get("is_dng", 0),
            tiff_feat.get("min_width", 0),
            tiff_feat.get("min_height", 0),
            tiff_feat.get("ifd_entry_max", 0),
            tiff_feat.get("subifd_count_sum", 0),
            tiff_feat.get("new_subfile_types_unique", 0),
            tiff_feat.get("total_opcodes", 0),
            tiff_feat.get("unknown_opcodes", 0),
            tiff_feat.get("max_opcode_id", 0),
            tiff_feat.get("opcode_list1_bytes", 0),
            tiff_feat.get("opcode_list2_bytes", 0),
            tiff_feat.get("opcode_list3_bytes", 0),
            tiff_feat.get("max_width", 0),
            tiff_feat.get("max_height", 0),
            tiff_feat.get("total_pixels", 0),
            tiff_feat.get("file_size", file_size),
            tiff_feat.get("bytes_per_pixel_milli", 0),
            tiff_feat.get("pixels_per_mb", 0),
            tiff_feat.get("opcode_list_bytes_total", 0),
            tiff_feat.get("opcode_list_bytes_max", 0),
            tiff_feat.get("opcode_list_present_count", 0),
            tiff_feat.get("opcode_bytes_ratio_permille", 0),
            tiff_feat.get("opcode_bytes_per_opcode_milli", 0),
            tiff_feat.get("unknown_opcode_ratio_permille", 0),
            tiff_feat.get("has_opcode_list1", 0),
            tiff_feat.get("has_opcode_list2", 0),
            tiff_feat.get("has_opcode_list3", 0),
            header_entropy,
            tail_entropy,
            overall_entropy,
            entropy_gradient,
        ]

        byte_features.append(magika_like_bytes(p))
        struct_features.append([float(x) for x in struct_vec])
        labels.append(y)
        label_names.append(label)
        file_paths.append(p)

        meta_rows.append(
            {
                "path": p,
                "label": label,
                "y": str(y),
                "file_size": str(file_size),
                "header_entropy": f"{header_entropy:.6f}",
                "tail_entropy": f"{tail_entropy:.6f}",
                "overall_entropy": f"{overall_entropy:.6f}",
                "entropy_gradient": f"{entropy_gradient:.6f}",
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        X_bytes=np.stack(byte_features, axis=0),
        X_struct=np.array(struct_features, dtype=np.float32),
        y=np.array(labels, dtype=np.int64),
        labels=np.array(label_names),
        paths=np.array(file_paths),
        struct_feature_names=np.array(struct_feature_names),
    )

    print("Wrote:", args.output_npz)
    print("Wrote:", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
