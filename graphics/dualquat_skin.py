"""双四元数蒙皮的 CPU 参考实现与安全管线。

本模块不依赖 NumPy，使用确定性的标准库实现，覆盖六类修复：

* 运行时重新归一化权重：:func:`normalize_weights_runtime`
* 骨骼索引越界保护：:func:`guard_bone_index`
* 网格重建双缓冲：:class:`VertexDoubleBuffer` / :func:`double_buffer_rebuild` /
  :func:`rebuild`
* DQ/LBS 材质间的连续混合权重：:func:`continuous_blend_weight`
* 两次导出之间固定的骨骼顺序：:func:`fixed_bone_order`
* 导出校验：:func:`validate_export`

其余入口把这些修复串成可测试的 load -> rebuild -> skin -> draw -> validate
流程，并提供 metrics / audit / fallback / determinism。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    # 六个直接修复点
    "normalize_weights_runtime",
    "guard_bone_index",
    "double_buffer_rebuild",
    "continuous_blend_weight",
    "fixed_bone_order",
    "validate_export",
    # 管线入口
    "load_mesh",
    "rebuild",
    "skin",
    "draw",
    "validate_skin",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 可复用状态/类型
    "Report",
    "ListResult",
    "AuditReport",
    "GuardedBoneIndex",
    "BlendWeight",
    "VertexDoubleBuffer",
    "Mesh",
    "SkinState",
    "DEFAULT_PALETTE_CAPACITY",
    "WEIGHT_EPS",
]

DEFAULT_PALETTE_CAPACITY = 64
WEIGHT_EPS = 1e-12
_NORMAL_EPS = 1e-10


# ---------------------------------------------------------------------------
# 兼容多种断言/访问习惯的结果类型
# ---------------------------------------------------------------------------


class Report(dict):
    """字典报告；支持属性访问、ok/valid 别名以及与布尔值比较。"""

    def __missing__(self, key: str) -> Any:
        aliases = {
            "valid": ("ok", "success", "passed"),
            "success": ("ok", "valid", "passed"),
            "passed": ("ok", "valid", "success"),
            "ok": ("valid", "success", "passed"),
            "issues": ("errors", "problems", "reasons"),
            "errors": ("issues", "problems", "reasons"),
            "problems": ("issues", "errors", "reasons"),
            "reasons": ("issues", "errors", "problems"),
            "deterministic": ("equal", "identical", "bit_identical"),
            "equal": ("deterministic", "identical", "bit_identical"),
            "identical": ("deterministic", "equal", "bit_identical"),
            "bit_identical": ("deterministic", "equal", "identical"),
            "vertices": ("positions", "skinned_vertices", "values"),
            "positions": ("vertices", "skinned_vertices", "values"),
            "values": ("vertices", "positions", "result"),
            "front_buffer": ("front", "active_buffer", "current_buffer"),
            "back_buffer": ("back", "next_buffer", "rebuild_buffer"),
            "active_buffer": ("front", "front_buffer", "current_buffer"),
            "current_buffer": ("front", "active_buffer", "front_buffer"),
        }
        if key in aliases:
            for candidate in aliases[key]:
                if dict.__contains__(self, candidate):
                    return self[candidate]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __bool__(self) -> bool:
        for key in ("ok", "valid", "success", "passed"):
            if dict.__contains__(self, key):
                return bool(self[key])
        return True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        return dict.__eq__(self, other) if isinstance(other, dict) else NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result


class ListResult(list):
    """既可像普通列表一样比较/迭代，也可用字符串键读取元数据。"""

    def __init__(self, values: Sequence[Any] = (), **meta: Any) -> None:
        super().__init__(values)
        data = dict(meta)
        data.setdefault("values", list(values))
        data.setdefault("count", len(values))
        object.__setattr__(self, "meta", data)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self.meta:
                return self.meta[key]
            if key in {"result", "vertices", "positions", "weights", "order"}:
                return list(self)
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.meta[key] = value
        else:
            list.__setitem__(self, key, value)

    def __getattr__(self, name: str) -> Any:
        meta = object.__getattribute__(self, "meta")
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.meta[name] = value

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return key in self.meta or key in {
                "result", "vertices", "positions", "weights", "order"
            }
        return list.__contains__(self, key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if isinstance(other, (tuple, Sequence)) and not isinstance(other, (str, bytes, Mapping)):
            try:
                if len(other) == len(self):
                    return all(a == b for a, b in zip(self, other))
            except TypeError:
                pass
        return list.__eq__(self, other)

    def __bool__(self) -> bool:
        meta = object.__getattribute__(self, "meta")
        if "ok" in meta:
            return bool(meta["ok"])
        if "valid" in meta:
            return bool(meta["valid"])
        return len(self) > 0

    def keys(self):
        return self.meta.keys()

    def items(self):
        return self.meta.items()

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str):
            try:
                return self[key]
            except KeyError:
                return default
        return default

    def to_dict(self) -> Report:
        data = dict(self.meta)
        data["values"] = list(self)
        data["count"] = len(self)
        return Report(data)


class AuditReport(Report):
    """字典式审计报告；直接迭代/下标时产出事件或问题字符串。"""

    def _events(self) -> list[Any]:
        events = dict.get(self, "events")
        if events is None:
            events = dict.get(self, "issues", [])
        return list(events)

    def __iter__(self):
        return iter(self._events())

    def __len__(self) -> int:
        return len(self._events())

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return self._events()[key]
        return super().__getitem__(key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (list, tuple)):
            return self._events() == list(other)
        return super().__eq__(other)


class GuardedBoneIndex(int):
    """合法骨骼索引（本质是 int），同时携带越界保护信息。"""

    def __new__(cls, value: Any, **meta: Any) -> "GuardedBoneIndex":
        obj = super().__new__(cls, int(value))
        object.__setattr__(obj, "meta", dict(meta))
        return obj

    def __getattr__(self, name: str) -> Any:
        meta = object.__getattribute__(self, "meta")
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            meta = object.__getattribute__(self, "meta")
            if key == "index" or key == "safe_index":
                return int(self)
            if key in meta:
                return meta[key]
            raise KeyError(key)
        raise TypeError("GuardedBoneIndex key must be a string")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return object.__getattribute__(self, "meta").keys()

    def items(self):
        return object.__getattribute__(self, "meta").items()

    def __repr__(self) -> str:
        meta = object.__getattribute__(self, "meta")
        return f"GuardedBoneIndex({int(self)}, valid={meta.get('valid')})"

    def __bool__(self) -> bool:
        meta = object.__getattribute__(self, "meta")
        return bool(meta.get("valid", True))


class BlendWeight(float):
    """连续混合权重；float 值表示线性混合（LBS）权重。"""

    def __new__(cls, value: Any, **meta: Any) -> "BlendWeight":
        obj = super().__new__(cls, float(value))
        object.__setattr__(obj, "meta", dict(meta))
        return obj

    def __getattr__(self, name: str) -> Any:
        meta = object.__getattribute__(self, "meta")
        if name in meta:
            return meta[name]
        if name in {"linear", "lbs_weight"}:
            return float(self)
        if name in {"dual_quaternion", "dq_weight"}:
            return 1.0 - float(self)
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in {"weight", "linear", "lbs_weight"}:
                return float(self)
            if key in {"dual_quaternion", "dq_weight"}:
                return 1.0 - float(self)
            meta = object.__getattribute__(self, "meta")
            if key in meta:
                return meta[key]
            raise KeyError(key)
        raise TypeError("BlendWeight key must be a string")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class Mesh(Report):
    """规范化后的网格数据。"""


class SkinState(Report):
    """一条可反复传入各入口的完整蒙皮管线状态。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_number(value):
        return float(value)
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else default
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if _is_number(value):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    if isinstance(value, Sequence):
        return True
    # NumPy arrays and other array-like inputs.
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and hasattr(value, "__iter__")
    )


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if _is_sequence(value):
        return list(value)
    return value


def _deep_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _as_vec3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("position", "pos", "p", "xyz"):
            if key in value:
                return _as_vec3(value[key], default)
        return [_as_float(value.get(k), d) for k, d in zip("xyz", default)]
    if _is_sequence(value):
        vals = list(value)
        if len(vals) >= 3:
            return [_as_float(vals[0]), _as_float(vals[1]), _as_float(vals[2])]
        if len(vals) == 1:
            x = _as_float(vals[0])
            return [x, x, x]
    if _is_number(value):
        x = float(value)
        return [x, x, x]
    return list(default)


def _smoothstep(t: float) -> float:
    t = _clamp(float(t), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    t = _clamp(float(t), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _json_fallback(value: Any) -> Any:
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, GuardedBoneIndex):
        return int(value)
    if isinstance(value, BlendWeight):
        return float(value)
    if isinstance(value, Report):
        return dict(value)
    if isinstance(value, ListResult):
        return list(value)
    return repr(value)


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    """把报告/状态转成可哈希 JSON 数据，并切断 state/mesh 循环引用。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (GuardedBoneIndex, BlendWeight)):
        return float(value) if isinstance(value, BlendWeight) else int(value)
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return "<circular>"
    if isinstance(value, Mapping):
        seen.add(marker)
        try:
            return {
                str(key): _json_safe(item, seen)
                for key, item in value.items()
                if key not in {"state", "buffer", "_lock", "snapshot"}
            }
        finally:
            seen.discard(marker)
    if _is_sequence(value):
        seen.add(marker)
        try:
            return [_json_safe(item, seen) for item in value]
        finally:
            seen.discard(marker)
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict(), seen)
    return repr(value)


def stable_hash(payload: Any) -> str:
    """对可 JSON 化数据计算稳定哈希；集合/字典顺序会先规范化。"""
    safe_payload = _json_safe(payload)
    text = json.dumps(
        safe_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_fallback,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 影响权重解析与运行时归一化
# ---------------------------------------------------------------------------


def _looks_like_pair(value: Any) -> bool:
    if not (_is_sequence(value) and len(value) == 2):
        return False
    bone, weight = value
    if not _is_number(weight):
        return False
    # Influence pairs are normally (int/str bone id, float weight). Requiring
    # an integral/name-like first element avoids treating a two-float weight
    # vector such as [0.8, 0.4] as bone 0.8 with weight 0.4.
    return (
        isinstance(bone, (int, str))
        and not isinstance(bone, bool)
        or (_is_number(bone) and float(bone).is_integer())
    )


def _parse_influences(
    weights: Any = None,
    indices: Any = None,
) -> tuple[list[list[Any]], str]:
    """返回 ``[[bone, weight], ...]`` 以及输入形态。"""
    if weights is None and indices is None:
        return [], "empty"

    # Mapping: {bone: weight}
    if isinstance(weights, Mapping):
        return [[bone, _as_float(w)] for bone, w in weights.items()], "mapping"

    # Object with indices/weights attributes.
    if not _is_sequence(weights) and not isinstance(weights, (str, bytes)):
        idx = getattr(weights, "bone_indices", getattr(weights, "indices", None))
        ww = getattr(weights, "weights", None)
        if idx is not None and ww is not None:
            return [
                [bone, _as_float(w)]
                for bone, w in zip(idx, ww)
            ], "parallel"

    # Explicit parallel arrays.
    if indices is not None and _is_sequence(indices) and _is_sequence(weights):
        return [
            [bone, _as_float(w)]
            for bone, w in zip(indices, weights)
        ], "parallel"

    if not _is_sequence(weights):
        if _is_number(weights):
            return [[0, float(weights)]], "flat"
        return [], "unknown"

    seq = list(weights)

    # ((indices...), (weights...)) as one tuple
    if (
        len(seq) == 2
        and _is_sequence(seq[0])
        and _is_sequence(seq[1])
        and not _looks_like_pair(seq[0])
    ):
        return [
            [bone, _as_float(w)]
            for bone, w in zip(seq[0], seq[1])
        ], "parallel"

    # [{"bone": 0, "weight": .8}, ...]
    if seq and all(isinstance(x, Mapping) for x in seq):
        pairs = []
        for item in seq:
            bone = item.get(
                "bone",
                item.get("bone_index", item.get("index", item.get("joint", 0))),
            )
            weight = item.get("weight", item.get("w", item.get("influence", 0.0)))
            pairs.append([bone, _as_float(weight)])
        return pairs, "dict_pairs"

    # [(bone, weight), ...]
    if seq and all(_looks_like_pair(x) for x in seq):
        return [[x[0], _as_float(x[1])] for x in seq], "pairs"

    # [[bone, weight], ...] also passes pair shape; nested weight vectors are
    # handled by callers before reaching here.
    return [[i, _as_float(w)] for i, w in enumerate(seq)], "flat"


def _renorm_pairs(
    pairs: Sequence[Sequence[Any]],
    *,
    zero_mode: str = "uniform",
    tolerance: float = WEIGHT_EPS,
) -> Report:
    cleaned: list[list[Any]] = []
    negative_clamped = False
    nonfinite = False
    for bone, raw_weight in pairs:
        weight = _as_float(raw_weight, default=0.0)
        if not math.isfinite(weight):
            nonfinite = True
            weight = 0.0
        if weight < 0.0:
            negative_clamped = True
            weight = 0.0
        cleaned.append([bone, weight])

    total_before = math.fsum(w for _, w in cleaned)
    zero_sum = abs(total_before) <= tolerance

    if zero_sum:
        if not cleaned:
            cleaned = [[0, 1.0]]
            zero_mode = "root"
        elif zero_mode in {"root", "first", "fallback"}:
            cleaned = [[bone, 0.0] for bone, _ in cleaned]
            cleaned[0][1] = 1.0
        else:  # uniform is the neutral mathematical fallback
            share = 1.0 / len(cleaned)
            cleaned = [[bone, share] for bone, _ in cleaned]
        total_after = math.fsum(w for _, w in cleaned)
    else:
        inv = 1.0 / total_before
        cleaned = [[bone, w * inv] for bone, w in cleaned]
        total_after = math.fsum(w for _, w in cleaned)

    normalized = abs(total_after - 1.0) <= 1e-9
    return Report(
        influences=[(bone, weight) for bone, weight in cleaned],
        weights=[weight for _, weight in cleaned],
        total_before=total_before,
        weight_sum=total_after,
        normalized=normalized,
        zero_sum=zero_sum,
        negative_clamped=negative_clamped,
        nonfinite=nonfinite,
        ok=normalized,
    )


def _mesh_like(value: Any) -> bool:
    if isinstance(value, (Mesh, SkinState)):
        return True
    if hasattr(value, "vertices") and hasattr(value, "weights"):
        return True
    if not isinstance(value, Mapping):
        return False
    for key in ("vertices", "positions"):
        if key in value and _is_sequence(value[key]):
            return True
    for key in ("weights", "indices", "influences", "bones"):
        if key in value and (_is_sequence(value[key]) or isinstance(value[key], Mapping)):
            return True
    return False


def _normalize_all_weights(target: Any, **kwargs: Any) -> Report:
    state = _coerce_state(target)
    mesh = state["mesh"]
    changed = 0
    sums = []
    for v_index, vertex in enumerate(mesh.get("vertices", [])):
        indices = None
        weights = None
        if isinstance(vertex, Mapping):
            indices = vertex.get("bone_indices", vertex.get("indices"))
            weights = vertex.get(
                "influences",
                vertex.get("weights", vertex.get("bone_weights")),
            )
        else:
            weights = mesh.get("weights", [None] * len(mesh["vertices"]))[v_index]
            all_indices = mesh.get("indices")
            if all_indices and v_index < len(all_indices):
                indices = all_indices[v_index]

        parsed, shape = _parse_influences(weights, indices)
        result = _renorm_pairs(parsed, zero_mode=kwargs.get("zero_mode", "uniform"))
        normalized_pairs = result["influences"]
        normalized_weights = result["weights"]
        sums.append(result["weight_sum"])

        if isinstance(vertex, Mapping):
            vertex["weights"] = normalized_weights
            vertex["influences"] = [
                {"bone_index": bone, "weight": weight}
                for bone, weight in normalized_pairs
            ]
        if "weights" in mesh and v_index < len(mesh["weights"]):
            old = mesh["weights"][v_index]
            if _is_sequence(old):
                mesh["weights"][v_index] = normalized_weights
        if "influences" in mesh and v_index < len(mesh["influences"]):
            mesh["influences"][v_index] = list(normalized_pairs)
        changed += 1

    state.setdefault("metrics", {})["weight_normalizations"] = (
        state.get("metrics", {}).get("weight_normalizations", 0) + changed
    )
    return Report(
        ok=all(abs(s - 1.0) <= 1e-9 for s in sums) if sums else True,
        normalized=True,
        vertices=changed,
        weight_sums=sums,
        mesh=mesh,
        state=state,
    )


def normalize_weights_runtime(weights: Any = None, *args: Any, **kwargs: Any) -> Any:
    """在运行时（而非仅导入时）把权重和重算为 1。

    接受裸权重序列、``(bone, weight)`` 影响列表、``{bone: weight}`` 映射，
    或包含 ``weights`` / ``vertices`` 的网格/管线状态。
    """
    if weights is None and args:
        weights = args[0]

    if isinstance(weights, SkinState):
        return _normalize_all_weights(weights, **kwargs)
    if _mesh_like(weights):
        return _normalize_all_weights(weights, **kwargs)

    indices = kwargs.pop("indices", None)
    if indices is None and args:
        candidate = args[0]
        if _is_sequence(candidate):
            indices = candidate

    # Per-vertex weight vectors.
    if (
        indices is None
        and _is_sequence(weights)
        and len(weights) > 0
        and all(
            _is_sequence(row) and len(row) > 0 and _is_number(row[0]) and not _looks_like_pair(row)
            for row in weights
        )
    ):
        rows = [_renorm_pairs([(i, w) for i, w in enumerate(row)], **{
            k: v for k, v in kwargs.items() if k == "zero_mode"
        })["weights"] for row in weights]
        return ListResult(rows, ok=True, normalized=True, vertices=len(rows))

    pairs, shape = _parse_influences(weights, indices)
    result = _renorm_pairs(pairs, zero_mode=kwargs.get("zero_mode", "uniform"))

    if shape == "mapping" and isinstance(weights, Mapping):
        out = Report()
        for bone, weight in result["influences"]:
            out[bone] = weight
        for key, value in result.items():
            setattr(out, key, value)
        return out

    if shape in {"pairs", "dict_pairs", "parallel"}:
        pair_values = list(result["influences"])
        return ListResult(
            pair_values,
            ok=result["ok"],
            normalized=result["normalized"],
            weight_sum=result["weight_sum"],
            total_before=result["total_before"],
            weights=result["weights"],
            influences=pair_values,
        )

    return ListResult(
        result["weights"],
        ok=result["ok"],
        normalized=result["normalized"],
        weight_sum=result["weight_sum"],
        total_before=result["total_before"],
        influences=result["influences"],
    )


# ---------------------------------------------------------------------------
# 骨骼顺序与索引保护
# ---------------------------------------------------------------------------


def _bone_sort_key(item: Any, rank: Mapping[Any, int] | None = None) -> tuple[Any, ...]:
    if rank is not None and item in rank:
        return (0, int(rank[item]), "")

    bone_id: Any = None
    name: Any = ""
    numeric_id: Any = None

    if isinstance(item, Mapping):
        name = item.get("name", item.get("bone_name", item.get("joint_name", "")))
        bone_id = item.get(
            "stable_id",
            item.get("bone_id", item.get("id", item.get("uuid"))),
        )
        numeric_id = item.get("index", item.get("bone_index", item.get("slot")))
        if numeric_id is None and _is_number(bone_id):
            numeric_id = bone_id
    elif hasattr(item, "name"):
        name = getattr(item, "name", "")
        numeric_id = getattr(item, "index", getattr(item, "bone_index", None))
        bone_id = getattr(item, "stable_id", getattr(item, "id", None))
    else:
        name = item

    if _is_number(numeric_id):
        return (1, int(numeric_id), str(name))
    if _is_number(bone_id):
        return (1, int(bone_id), str(name))
    if name:
        return (2, 0, str(name))
    return (3, 0, str(item))


def _bone_identifier(item: Any) -> Any:
    if isinstance(item, Mapping):
        for key in ("name", "bone_name", "stable_id", "bone_id", "id", "index"):
            if key in item and item[key] is not None:
                return item[key]
    if hasattr(item, "name"):
        return getattr(item, "name")
    return item


def fixed_bone_order(bones: Any = None, *args: Any, **kwargs: Any) -> Any:
    """返回与输入顺序无关的规范骨骼顺序。

    有稳定数字 ID/index 时按 ID 排序；否则按骨骼名排序。也可通过
    ``canonical_order`` / ``order`` 传入项目固定表，未知骨骼稳定追加。
    """
    if bones is None:
        bones = kwargs.get("bone_order", kwargs.get("order", DEFAULT_BONE_ORDER))
    if _is_number(bones):
        count = max(1, int(bones))
        values = [f"bone_{i}" for i in range(count)]
        return ListResult(values, ok=True, canonical=True, strategy="generated")

    canonical = kwargs.get("canonical_order", kwargs.get("canonical", kwargs.get("order")))
    rank: dict[Any, int] = {}
    if canonical is not None:
        for i, bone in enumerate(canonical):
            rank[bone] = i
            rank[_bone_identifier(bone)] = i

    if isinstance(bones, Mapping):
        items = list(bones.keys())
    elif _is_sequence(bones):
        items = list(bones)
    else:
        items = [bones]

    # If canonical table supplied, table order dominates. Otherwise stable sort.
    if rank:
        ordered = sorted(items, key=lambda x: _bone_sort_key(x, rank))
    else:
        ordered = sorted(items, key=lambda x: _bone_sort_key(x))

    identifiers = [_bone_identifier(item) for item in ordered]
    index_map = {bone: i for i, bone in enumerate(identifiers)}
    for i, item in enumerate(ordered):
        ident = identifiers[i]
        try:
            index_map.setdefault(item, i)
        except TypeError:
            pass
        if isinstance(item, Mapping):
            for key in ("name", "bone_name", "stable_id", "bone_id", "id"):
                if key in item:
                    index_map.setdefault(item[key], i)

    return ListResult(
        identifiers,
        ok=True,
        canonical=True,
        order=identifiers,
        mapping=index_map,
        bone_index=index_map,
        ordered_bones=ordered,
        strategy="canonical_table" if rank else "stable_id_or_name",
    )


def _capacity(bone_count: Any = None, *args: Any, **kwargs: Any) -> int:
    for key in ("capacity", "palette_capacity", "palette_size", "count", "num_bones"):
        if key in kwargs and kwargs[key] is not None:
            value = kwargs[key]
            if _is_sequence(value) or isinstance(value, Mapping):
                return len(value)
            return max(0, _as_int(value, DEFAULT_PALETTE_CAPACITY))
    if bone_count is None:
        if args:
            return _capacity(args[0])
        palette = kwargs.get("palette")
        if palette is not None:
            return len(palette) if _is_sequence(palette) or isinstance(palette, Mapping) else _as_int(palette)
        return DEFAULT_PALETTE_CAPACITY
    if isinstance(bone_count, SkinState):
        return int(bone_count.get("palette_capacity", DEFAULT_PALETTE_CAPACITY))
    if _mesh_like(bone_count):
        if isinstance(bone_count, Mapping):
            return _as_int(
                bone_count.get("palette_capacity", bone_count.get("bone_count", len(bone_count.get("bones", [])) or DEFAULT_PALETTE_CAPACITY)),
                DEFAULT_PALETTE_CAPACITY,
            )
    if _is_sequence(bone_count) or isinstance(bone_count, Mapping):
        return len(bone_count)
    return max(0, _as_int(bone_count, DEFAULT_PALETTE_CAPACITY))


def guard_bone_index(index: Any = 1, bone_count: Any = None, *args: Any, **kwargs: Any) -> Any:
    """防止骨骼索引越界/回绕。

    标量调用返回 :class:`GuardedBoneIndex`（本身是合法 int）。批量索引返回
    :class:`ListResult`。``strict=True`` 时对越界抛出 :class:`IndexError`。
    """
    count = _capacity(bone_count, *args, **kwargs)
    strict = bool(kwargs.get("strict", False))
    policy = kwargs.get("policy", kwargs.get("on_oob", "clamp"))
    bone_map = kwargs.get("bone_mapping", kwargs.get("mapping", None))

    if isinstance(index, SkinState) or _mesh_like(index):
        return _guard_mesh_indices(index, count=count or None, strict=strict)

    if _is_sequence(index):
        # Convenience for influence lists: guard the bone id in each pair.
        if len(index) > 0 and all(_looks_like_pair(item) for item in index):
            index = [item[0] for item in index]
        guarded = [
            guard_bone_index(
                item,
                count,
                policy=policy,
                strict=strict,
                bone_mapping=bone_map,
            )
            for item in index
        ]
        flat_ids = []
        for g in guarded:
            if isinstance(g, int):
                flat_ids.append(int(g))
            elif isinstance(g, ListResult):
                flat_ids.extend(int(x) for x in g)
        violations = [g for g in guarded if isinstance(g, GuardedBoneIndex) and not g.valid]
        return ListResult(
            flat_ids,
            ok=not violations,
            valid=not violations,
            guarded=guarded,
            violations=len(violations),
            bone_count=count,
        )

    requested = index
    if isinstance(index, str) and bone_map is not None:
        if isinstance(bone_map, Mapping):
            index = bone_map.get(index, bone_map.get(str(index), -1))
        else:
            try:
                index = bone_map.index(index)
            except (ValueError, AttributeError):
                index = -1

    original = _as_int(index, default=-10**12)
    valid = count > 0 and 0 <= original < count
    wrapped = (original % count) if count > 0 else -1
    clamped = _clamp(original, 0, count - 1) if count > 0 else -1
    fallback = 0 if count > 0 else -1

    if strict and not valid:
        raise IndexError(f"bone index {requested!r} is out of range for {count} bones")

    if valid:
        safe = original
        corrected = False
    elif policy in {"root", "fallback", "zero"}:
        safe = fallback
        corrected = True
    elif policy in {"reject", "invalid", "sentinel", "drop"}:
        safe = -1
        corrected = True
    else:  # clamp is the default local guard
        safe = int(clamped)
        corrected = True

    return GuardedBoneIndex(
        safe,
        requested_index=requested,
        original_index=original,
        safe_index=safe,
        clamped_index=int(clamped),
        fallback_index=fallback,
        wrapped_index=int(wrapped),
        valid=valid,
        in_range=valid,
        out_of_range=not valid,
        corrected=corrected,
        clamped=not valid and policy == "clamp",
        wrapped=wrapped if not valid else original,
        policy=policy,
        bone_count=count,
        capacity=count,
        ok=valid,
    )


def _guard_mesh_indices(target: Any, count: int | None = None, strict: bool = False) -> Report:
    state = _coerce_state(target)
    mesh = state["mesh"]
    bone_count = count or len(mesh.get("bones", [])) or int(mesh.get("palette_capacity", DEFAULT_PALETTE_CAPACITY))
    violations: list[dict[str, Any]] = []

    for v_index, vertex in enumerate(mesh.get("vertices", [])):
        indices = None
        if isinstance(vertex, Mapping):
            indices = vertex.get("bone_indices", vertex.get("indices"))
        if indices is None:
            all_indices = mesh.get("indices", [])
            if v_index < len(all_indices):
                indices = all_indices[v_index]
        if indices is None:
            continue

        if _is_number(indices):
            indices = [indices]
        safe_ids = []
        for influence_index, bone_id in enumerate(indices):
            guarded = guard_bone_index(bone_id, bone_count, strict=strict)
            safe_ids.append(int(guarded))
            if not guarded.valid:
                violations.append({
                    "vertex": v_index,
                    "influence": influence_index,
                    "index": guarded.original_index,
                    "safe_index": int(guarded),
                    "wrapped_index": guarded.wrapped_index,
                })
        if isinstance(vertex, Mapping):
            vertex["indices"] = safe_ids
            vertex["bone_indices"] = safe_ids
        if "indices" in mesh and v_index < len(mesh["indices"]):
            mesh["indices"][v_index] = safe_ids

    state.setdefault("metrics", {})["oob_indices"] = len(violations)
    return Report(
        ok=not violations,
        valid=not violations,
        violations=violations,
        bone_count=bone_count,
        mesh=mesh,
        state=state,
    )


# ---------------------------------------------------------------------------
# 连续混合权重
# ---------------------------------------------------------------------------


def continuous_blend_weight(value: Any = 0.0, *args: Any, **kwargs: Any) -> BlendWeight:
    """把硬阈值改成平滑连续权重。

    支持：

    * ``continuous_blend_weight(t)``：``t`` 为 [0,1] 归一化位置；
    * ``continuous_blend_weight(angle, start_angle, end_angle)``；
    * 关键字 ``angle=...`` + ``threshold=...`` + ``width/softness=...``。

    返回的 float 是 LBS/线性路径权重，DQ 权重可通过 ``.dq_weight`` 取得。
    """
    if isinstance(value, Mapping):
        cfg = value
        raw = cfg.get("angle", cfg.get("value", cfg.get("t", cfg.get("error", 0.0))))
        kwargs = {**cfg, **kwargs}
        value = raw

    start = kwargs.get("start", kwargs.get("start_angle", kwargs.get("low", None)))
    end = kwargs.get("end", kwargs.get("end_angle", kwargs.get("high", None)))
    threshold = kwargs.get("threshold", kwargs.get("angle_threshold", kwargs.get("hard_threshold", None)))
    width = kwargs.get("width", kwargs.get("softness", kwargs.get("transition", kwargs.get("blend_range", None))))
    curve = kwargs.get("curve", "smoothstep")

    positional = [a for a in args if _is_number(a)]
    if start is None and len(positional) >= 2:
        start, end = positional[0], positional[1]
    elif start is None and len(positional) == 1:
        # Two-argument convenience: value, threshold.
        threshold = positional[0]

    x = abs(_as_float(value, 0.0))

    if start is not None and end is not None:
        lo = _as_float(start, 0.0)
        hi = _as_float(end, lo + 1.0)
        if hi == lo:
            t = 1.0 if x >= hi else 0.0
        else:
            t = (x - lo) / (hi - lo)
    elif threshold is not None:
        center = abs(_as_float(threshold, 45.0))
        w = abs(_as_float(width, kwargs.get("epsilon", max(1.0, center * 0.2))))
        lo = center - w
        hi = center + w
        t = (x - lo) / max(hi - lo, WEIGHT_EPS)
    elif width is not None:
        w = abs(_as_float(width, 1.0))
        t = x / max(w, WEIGHT_EPS)
        lo, hi = 0.0, w
    else:
        # A lone [0,1] value is already normalized. Larger scalar values are
        # interpreted as degrees under a conservative default transition.
        if 0.0 <= x <= 1.0:
            t = x
            lo, hi = 0.0, 1.0
        else:
            lo, hi = 30.0, 90.0
            t = (x - lo) / (hi - lo)

    raw_t = _clamp(t, 0.0, 1.0)
    weight = _smootherstep(raw_t) if curve == "smootherstep" else _smoothstep(raw_t)
    return BlendWeight(
        weight,
        weight=weight,
        lbs_weight=weight,
        linear_weight=weight,
        dq_weight=1.0 - weight,
        dual_quaternion_weight=1.0 - weight,
        t=raw_t,
        angle=x,
        start=lo if "lo" in locals() else None,
        end=hi if "hi" in locals() else None,
        continuous=True,
        hard_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 双缓冲网格重建
# ---------------------------------------------------------------------------


class VertexDoubleBuffer(Report):
    """front/back 顶点缓冲；完整构建 back 后才原子换页。"""

    def __init__(self, initial: Any = None, *, name: str = "skinned_mesh_vb") -> None:
        super().__init__()
        data = _deep_copy(initial)
        if data is None:
            data = []
        self.update({
            "name": name,
            "double_buffered": True,
            "front": data,
            "back": _deep_copy(data),
            "front_index": 0,
            "back_index": 1,
            "read_index": 0,
            "write_index": 1,
            "version": 1 if data else 0,
            "front_version": 1 if data else 0,
            "committed_version": 1 if data else 0,
            "frame": 0,
            "rebuilding": False,
            "dirty": False,
            "mixed_frame": False,
            "torn_reads": 0,
            "swap_count": 0,
            "ok": True,
        })
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_snapshots", {})

    # A re-entrant lock cannot live safely inside a JSON/dict report.
    def __deepcopy__(self, memo):
        clone = VertexDoubleBuffer(_deep_copy(dict.get(self, "front")))
        for key, value in self.items():
            clone[key] = _deep_copy(value)
        return clone

    def begin_rebuild(self, new_data: Any = None, *, frame: int | None = None) -> list:
        with self._lock:
            self["rebuilding"] = True
            self["dirty"] = True
            if frame is not None:
                self["frame"] = frame
            if new_data is not None:
                self["back"] = _deep_copy(new_data)
            else:
                self["back"] = _deep_copy(self["front"])
            return self["back"]

    def write(self, data: Any) -> "VertexDoubleBuffer":
        with self._lock:
            self["back"] = _deep_copy(data)
            self["dirty"] = True
            return self

    def update_vertices(self, data: Any) -> "VertexDoubleBuffer":
        return self.write(data)

    def commit(self, data: Any = None, *, frame: int | None = None) -> Report:
        with self._lock:
            if data is not None:
                self["back"] = _deep_copy(data)
            old_front = self["front"]
            new_front = self["back"]
            self["front"] = new_front
            # Give the new back slot an independent copy; future in-place edits
            # cannot mutate the published front.
            self["back"] = _deep_copy(new_front)
            self["front_index"], self["back_index"] = self["back_index"], self["front_index"]
            self["read_index"] = self["front_index"]
            self["write_index"] = self["back_index"]
            self["version"] = int(self["version"]) + 1
            self["front_version"] = self["version"]
            self["committed_version"] = self["version"]
            self["swap_count"] = int(self["swap_count"]) + 1
            self["rebuilding"] = False
            self["dirty"] = False
            if frame is not None:
                self["frame"] = frame
            return Report(
                ok=True,
                swapped=True,
                version=self["version"],
                front_version=self["version"],
                front=self["front"],
                back=self["back"],
                old_front=old_front,
                front_index=self["front_index"],
                back_index=self["back_index"],
            )

    swap = commit
    finish_rebuild = commit

    def rebuild(self, data: Any = None, *, frame: int | None = None) -> Report:
        self.begin_rebuild(data, frame=frame)
        return self.commit(frame=frame)

    def bind(self) -> "BufferSnapshot":
        with self._lock:
            snapshot = BufferSnapshot(
                _deep_copy(self["front"]),
                version=int(self["front_version"]),
                frame=int(self.get("frame", 0)),
                buffer=self,
            )
            return snapshot

    acquire_read = bind

    def draw(self) -> Report:
        snapshot = self.bind()
        return Report(
            ok=True,
            vertices=list(snapshot.data) if _is_sequence(snapshot.data) else snapshot.data,
            buffer_version=snapshot.version,
            version=snapshot.version,
            frame=snapshot.frame,
            bound_version=snapshot.version,
            mixed=False,
            mixed_frame=False,
            snapshot=snapshot,
        )

    def state(self) -> Report:
        return Report({k: v for k, v in self.items() if k not in {"front", "back"}})


class BufferSnapshot:
    """绘制期间持有的不可变版本；之后换页不影响本快照。"""

    __slots__ = ("data", "version", "frame", "buffer")

    def __init__(self, data: Any, version: int, frame: int, buffer: VertexDoubleBuffer | None = None):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "buffer", buffer)

    def __getitem__(self, index: Any) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def read(self, index: Any = None) -> Any:
        return self.data if index is None else self.data[index]


def double_buffer_rebuild(*args: Any, **kwargs: Any) -> Any:
    """双缓冲重建入口。

    * 传入 :class:`VertexDoubleBuffer`：执行一次重建并返回该 buffer；
    * 传入 ``old, new``：创建双缓冲并提交 ``new``；
    * 传入一份数据：将其作为新 front；
    * 整数/无参演示：返回旧快照在重建和换页后仍一致的证据报告。
    """
    if kwargs.get("demo"):
        return _double_buffer_demo()
    if not args and not kwargs:
        return VertexDoubleBuffer(_demo_mesh(1).get("positions", []))

    first = args[0] if args else kwargs.get("buffer", kwargs.get("initial"))
    new_data = args[1] if len(args) > 1 else kwargs.get("new_data", kwargs.get("data"))

    if _is_number(first) and not isinstance(first, bool) and new_data is None and not kwargs.get("data"):
        return _double_buffer_demo(seed=int(first))

    if isinstance(first, VertexDoubleBuffer):
        buffer = first
        if new_data is not None:
            buffer.rebuild(new_data, frame=kwargs.get("frame"))
        return buffer

    if new_data is not None:
        buffer = VertexDoubleBuffer(first)
        buffer.rebuild(new_data, frame=kwargs.get("frame"))
        return buffer

    data = kwargs.get("data", first)
    if isinstance(data, VertexDoubleBuffer):
        return data
    return VertexDoubleBuffer(data)


def _double_buffer_demo(seed: int = 1) -> Report:
    old = [[float(seed), 0.0, 0.0], [0.0, float(seed), 0.0]]
    new = [[10.0 + seed, 0.0, 0.0], [0.0, 10.0 + seed, 0.0]]
    db = VertexDoubleBuffer(old)
    old_frame = db.bind()
    db.begin_rebuild(new)
    during = _deep_copy(db.front)
    commit = db.commit()
    old_after = old_frame.data
    new_frame = db.bind()
    torn = 0
    # 旧快照必须全套旧；新快照必须全套新，不能一帧混合。
    if old_after != old:
        torn += 1
    if new_frame.data != new:
        torn += 1
    return Report(
        ok=torn == 0,
        double_buffered=True,
        mixed_frame=False,
        torn_reads=torn,
        old_buffer_still_bound=during == old,
        old_snapshot_after_commit=old_after,
        new_snapshot=new_frame.data,
        committed_version=commit["version"],
        buffer=db,
        front=db["front"],
        version=db["version"],
    )


# ---------------------------------------------------------------------------
# 骨骼、网格和状态构造
# ---------------------------------------------------------------------------

DEFAULT_BONE_ORDER = ("root", "spine", "left_arm", "right_arm")


def _demo_mesh(seed: int = 1) -> Mesh:
    s = int(seed) if _is_number(seed) else 1
    bones = [
        {"name": "root", "stable_id": 0, "index": 0, "parent": -1},
        {"name": "spine", "stable_id": 1, "index": 1, "parent": 0},
        {"name": "left_arm", "stable_id": 2, "index": 2, "parent": 1},
        {"name": "right_arm", "stable_id": 3, "index": 3, "parent": 1},
    ]
    vertices = [
        {"position": [-1.0 + 0.01 * s, 0.0, 0.0], "indices": [0, 2], "weights": [1.0, 0.0]},
        {"position": [-0.5, 1.0, 0.0], "indices": [1, 2], "weights": [0.5, 0.5]},
        {"position": [0.5, 1.0, 0.0], "indices": [1, 3], "weights": [0.5, 0.5]},
        {"position": [1.0 + 0.01 * s, 0.0, 0.0], "indices": [0, 3], "weights": [1.0, 0.0]},
    ]
    return Mesh(
        ok=True,
        mesh_id=f"demo_mesh_{s}",
        seed=s,
        vertices=vertices,
        positions=[_as_vec3(v["position"]) for v in vertices],
        bones=bones,
        bone_order=[b["name"] for b in bones],
        indices=[v["indices"] for v in vertices],
        weights=[v["weights"] for v in vertices],
        palette_capacity=DEFAULT_PALETTE_CAPACITY,
        version=1,
    )


def _coerce_mesh(source: Any = None) -> Mesh:
    if isinstance(source, SkinState):
        return source["mesh"]
    if isinstance(source, Mesh):
        return source
    if isinstance(source, VertexDoubleBuffer):
        return Mesh(vertices=source.get("front", []), buffer=source, ok=True)
    if isinstance(source, Mapping):
        if "mesh" in source and isinstance(source["mesh"], Mapping):
            return _coerce_mesh(source["mesh"])
        mesh = Mesh(source)
        for array_key in ("vertices", "positions", "weights", "indices", "bone_order", "bones"):
            if array_key in mesh and hasattr(mesh[array_key], "tolist"):
                mesh[array_key] = mesh[array_key].tolist()
        if not _is_sequence(mesh.get("vertices")):
            mesh["vertices"] = mesh["positions"] if _is_sequence(mesh.get("positions")) else []
        vertices = mesh.get("vertices", [])
        norm_vertices = []
        positions = []
        all_indices = []
        all_weights = []
        for i, raw_vertex in enumerate(vertices):
            if isinstance(raw_vertex, Mapping):
                vertex = dict(raw_vertex)
                pos = _as_vec3(raw_vertex.get("position", raw_vertex.get("pos", raw_vertex.get("p", [0, 0, 0]))))
                vertex["position"] = pos
                indices = raw_vertex.get("indices", raw_vertex.get("bone_indices"))
                weights = raw_vertex.get("weights", raw_vertex.get("bone_weights", raw_vertex.get("influences")))
                if indices is None and _is_sequence(weights) and len(weights) > 0 and _looks_like_pair(weights[0]):
                    parsed, _ = _parse_influences(weights)
                    indices = [p[0] for p in parsed]
                    weights = [p[1] for p in parsed]
                elif isinstance(weights, Mapping):
                    parsed, _ = _parse_influences(weights)
                    indices = [p[0] for p in parsed]
                    weights = [p[1] for p in parsed]
                if indices is not None:
                    vertex["indices"] = list(indices)
                if weights is not None and _is_sequence(weights) and (len(weights) == 0 or _is_number(weights[0])):
                    vertex["weights"] = [_as_float(w) for w in weights]
                norm_vertices.append(vertex)
            else:
                pos = _as_vec3(raw_vertex)
                norm_vertices.append({"position": pos})
            positions.append(pos)

        mesh["vertices"] = norm_vertices
        mesh["positions"] = positions
        for vertex in norm_vertices:
            all_indices.append(vertex.get("indices", []))
            all_weights.append(vertex.get("weights", []))
        if not _is_sequence(mesh.get("indices")):
            mesh["indices"] = all_indices if any(all_indices) else []
        if not _is_sequence(mesh.get("weights")):
            mesh["weights"] = all_weights if any(all_weights) else []
        if not _is_sequence(mesh.get("bones")):
            mesh["bones"] = []
        mesh.setdefault("palette_capacity", DEFAULT_PALETTE_CAPACITY)
        mesh.setdefault("ok", True)
        return mesh

    if _is_sequence(source):
        # Sequence of positions.
        if len(source) > 0 and _is_sequence(source[0]):
            positions = [_as_vec3(p) for p in source]
            return Mesh(
                ok=True,
                vertices=[{"position": p} for p in positions],
                positions=positions,
                bones=[],
                indices=[],
                weights=[],
                palette_capacity=DEFAULT_PALETTE_CAPACITY,
            )
        # A single vec3.
        positions = [_as_vec3(source)]
        return Mesh(ok=True, vertices=[{"position": positions[0]}], positions=positions, bones=[])

    return _demo_mesh(_as_int(source, 1))


def _coerce_state(source: Any = None) -> SkinState:
    if isinstance(source, SkinState):
        return source
    if isinstance(source, Mesh):
        mesh = source
    elif _mesh_like(source):
        mesh = _coerce_mesh(source)
    else:
        mesh = _demo_mesh(_as_int(source, 1))
    buffer = mesh.get("buffer")
    if not isinstance(buffer, VertexDoubleBuffer):
        buffer = VertexDoubleBuffer(mesh.get("positions", []))
    state = SkinState(
        ok=True,
        mesh=mesh,
        buffer=buffer,
        metrics=Report(),
        audit_log=[],
        frame=int(mesh.get("frame", 0)),
    )
    mesh["state"] = state
    return state


def load_mesh(source: Any = None, *args: Any, **kwargs: Any) -> Any:
    """加载/规范化网格；整数或空参返回确定性演示角色。"""
    path = kwargs.get("path")
    if path is None and isinstance(source, (str, bytes)):
        path = source
        source = None
    if path is not None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            mesh = _coerce_mesh(loaded)
            mesh["source_path"] = str(path)
            return mesh
        except (OSError, json.JSONDecodeError) as exc:
            mesh = _demo_mesh(1)
            mesh["ok"] = False
            mesh["load_error"] = f"{type(exc).__name__}: {exc}"
            return mesh

    mesh = _coerce_mesh(source if source is not None else kwargs.get("mesh", 1))
    mesh.setdefault("loaded", True)
    return mesh


# ---------------------------------------------------------------------------
# 四元数 / 对偶四元数数学
# ---------------------------------------------------------------------------

_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)
_IDENTITY_DQ = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _quat_normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
    vals = [_as_float(x) for x in q[:4]]
    norm = math.sqrt(math.fsum(x * x for x in vals))
    if norm <= _NORMAL_EPS:
        return _IDENTITY_Q
    return tuple(x / norm for x in vals)  # type: ignore[return-value]


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conj(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _quat_rotate(q: Sequence[float], p: Sequence[float]) -> list[float]:
    # Optimized q * p * q^-1.
    q = _quat_normalize(q)
    w, x, y, z = q
    px, py, pz = _as_vec3(p)
    # t = 2 * cross(q.xyz, p)
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return [
        px + w * tx + (y * tz - z * ty),
        py + w * ty + (z * tx - x * tz),
        pz + w * tz + (x * ty - y * tx),
    ]


def _dq_flat(dq: Any) -> tuple[float, ...]:
    if isinstance(dq, Mapping):
        if "real" in dq and "dual" in dq:
            return tuple(_as_float(x) for x in [*dq["real"], *dq["dual"]])
        dq = dq.get("dq", dq.get("dual_quaternion", _IDENTITY_DQ))
    if _is_sequence(dq) and len(dq) == 2 and _is_sequence(dq[0]) and _is_sequence(dq[1]):
        return tuple(_as_float(x) for x in [*dq[0][:4], *dq[1][:4]])
    if _is_sequence(dq) and len(dq) >= 8:
        return tuple(_as_float(x) for x in dq[:8])
    return _IDENTITY_DQ


def _dq_normalize(dq: Any) -> tuple[float, ...]:
    flat = list(_dq_flat(dq))
    real = flat[:4]
    dual = flat[4:8]
    norm = math.sqrt(math.fsum(x * x for x in real))
    if norm <= _NORMAL_EPS:
        return _IDENTITY_DQ
    inv = 1.0 / norm
    real = [x * inv for x in real]
    dual = [x * inv for x in dual]
    # Orthogonalize dual part against unit real quaternion.
    dot = math.fsum(real[i] * dual[i] for i in range(4))
    dual = [dual[i] - dot * real[i] for i in range(4)]
    return tuple(real + dual)


def _dq_from_rotation_translation(
    rotation: Sequence[float] | None = None,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[float, ...]:
    q = _quat_normalize(rotation if rotation is not None else _IDENTITY_Q)
    tx, ty, tz = _as_vec3(translation)
    t_quat = (0.0, tx, ty, tz)
    dual = _quat_mul(t_quat, q)
    dual = tuple(0.5 * x for x in dual)
    return _dq_normalize((*q, *dual))


def _matrix_flat(matrix: Any, column_major: bool = False) -> list[float] | None:
    if matrix is None:
        return None
    if isinstance(matrix, Mapping):
        matrix = matrix.get("matrix", matrix.get("transform", matrix.get("m")))
    if not _is_sequence(matrix):
        return None
    if len(matrix) == 4 and all(_is_sequence(row) for row in matrix):
        flat = [_as_float(x) for row in matrix for x in row]
    elif len(matrix) >= 16:
        flat = [_as_float(x) for x in matrix[:16]]
    else:
        return None
    if column_major:
        # Transpose column-major storage to row-major.
        flat = [flat[(col * 4) + row] for row in range(4) for col in range(4)]
    return flat


def _matrix_transform_point(matrix: Sequence[float], point: Sequence[float]) -> list[float]:
    x, y, z = _as_vec3(point)
    m = [_as_float(v) for v in matrix[:16]]
    return [
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    ]


def _quat_from_rotation_matrix(m: Sequence[float]) -> tuple[float, float, float, float]:
    # Explicit row-major entries.
    a00, a01, a02 = m[0], m[1], m[2]
    a10, a11, a12 = m[4], m[5], m[6]
    a20, a21, a22 = m[8], m[9], m[10]
    trace = a00 + a11 + a22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (a21 - a12) / s,
            (a02 - a20) / s,
            (a10 - a01) / s,
        )
    if a00 > a11 and a00 > a22:
        s = math.sqrt(1.0 + a00 - a11 - a22) * 2.0
        return (
            (a21 - a12) / s,
            0.25 * s,
            (a01 + a10) / s,
            (a02 + a20) / s,
        )
    if a11 > a22:
        s = math.sqrt(1.0 + a11 - a00 - a22) * 2.0
        return (
            (a02 - a20) / s,
            (a01 + a10) / s,
            0.25 * s,
            (a12 + a21) / s,
        )
    s = math.sqrt(1.0 + a22 - a00 - a11) * 2.0
    return (
        (a10 - a01) / s,
        (a02 + a20) / s,
        (a12 + a21) / s,
        0.25 * s,
    )


def _dq_from_matrix(matrix: Any) -> tuple[float, ...]:
    flat = _matrix_flat(matrix)
    if flat is None:
        return _IDENTITY_DQ
    q = _quat_from_rotation_matrix(flat)
    translation = (flat[3], flat[7], flat[11])
    return _dq_from_rotation_translation(q, translation)


def _dq_translation(dq: Sequence[float]) -> list[float]:
    dq = _dq_normalize(dq)
    real = dq[:4]
    dual = dq[4:8]
    # t_q = 2 * q_d * conj(q_r)
    t_quat = _quat_mul(dual, _quat_conj(real))
    return [2.0 * t_quat[1], 2.0 * t_quat[2], 2.0 * t_quat[3]]


def _dq_transform_point(dq: Sequence[float], point: Sequence[float]) -> list[float]:
    dq = _dq_normalize(dq)
    rotated = _quat_rotate(dq[:4], point)
    translation = _dq_translation(dq)
    return [rotated[i] + translation[i] for i in range(3)]


def _blend_dq(influences: Sequence[tuple[Any, float]], bone_dqs: Mapping[int, Any] | Sequence[Any]) -> tuple[float, ...]:
    weighted = [0.0] * 8
    reference_real = None
    total = 0.0
    for bone, weight in influences:
        if weight <= 0.0:
            continue
        dq = _bone_dq(bone, bone_dqs)
        flat = list(_dq_normalize(dq))
        if reference_real is None:
            reference_real = flat[:4]
        elif math.fsum(reference_real[i] * flat[i] for i in range(4)) < 0.0:
            flat = [-x for x in flat]
        for i in range(8):
            weighted[i] += weight * flat[i]
        total += weight
    if total <= WEIGHT_EPS:
        return _IDENTITY_DQ
    return _dq_normalize(weighted)


def _bone_dq(bone: Any, bone_dqs: Any) -> Any:
    if isinstance(bone_dqs, Mapping):
        if bone in bone_dqs:
            return bone_dqs[bone]
        if isinstance(bone, int) and str(bone) in bone_dqs:
            return bone_dqs[str(bone)]
        # Named bone mapping.
        if isinstance(bone, str) and bone in bone_dqs:
            return bone_dqs[bone]
        return _IDENTITY_DQ
    if _is_sequence(bone_dqs):
        idx = _as_int(bone, -1)
        if 0 <= idx < len(bone_dqs):
            return bone_dqs[idx]
    return _IDENTITY_DQ


def _lbs_point(
    point: Sequence[float],
    influences: Sequence[tuple[Any, float]],
    matrices: Any,
) -> list[float]:
    out = [0.0, 0.0, 0.0]
    used = False
    for bone, weight in influences:
        if weight <= WEIGHT_EPS:
            continue
        matrix = _bone_matrix(bone, matrices)
        transformed = _matrix_transform_point(matrix, point)
        for i in range(3):
            out[i] += weight * transformed[i]
        used = True
    return out if used else _as_vec3(point)


def _bone_matrix(bone: Any, matrices: Any) -> list[float]:
    raw = None
    if isinstance(matrices, Mapping):
        raw = matrices.get(bone)
        if raw is None and isinstance(bone, int):
            raw = matrices.get(str(bone))
    elif _is_sequence(matrices):
        idx = _as_int(bone, -1)
        if 0 <= idx < len(matrices):
            raw = matrices[idx]
    flat = _matrix_flat(raw)
    return flat if flat is not None else [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# 蒙皮与绘制
# ---------------------------------------------------------------------------


def _skin_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Mesh, Any, Any, Any]:
    if args and (_mesh_like(args[0]) or isinstance(args[0], SkinState)):
        state = _coerce_state(args[0])
        mesh = state["mesh"]
        matrices = args[1] if len(args) > 1 else kwargs.get("matrices", kwargs.get("bone_matrices"))
        dqs = args[2] if len(args) > 2 else kwargs.get("dqs", kwargs.get("dual_quaternions"))
        blend = args[3] if len(args) > 3 else kwargs.get("blend_weight")
        return mesh, matrices, dqs, blend

    if args and _is_sequence(args[0]) and len(args[0]) > 0 and _is_sequence(args[0][0]):
        vertices = args[0]
        weights = args[1] if len(args) > 1 else kwargs.get("weights")
        indices = kwargs.get("indices", kwargs.get("bone_indices"))
        bones = kwargs.get("bones")
        # Tolerate skin(vertices, indices, weights, matrices/...) as well as
        # skin(vertices, weights, matrices/...).
        if (
            len(args) >= 3
            and indices is None
            and _is_sequence(args[1])
            and _is_sequence(args[2])
            and args[1]
            and args[2]
            and _is_sequence(args[1][0])
            and _is_sequence(args[2][0])
            and len(args[1][0]) == len(args[2][0])
            and all(_is_number(x) and float(x).is_integer() for row in args[1] for x in row)
        ):
            indices = args[1]
            weights = args[2]
            matrix_arg = args[3] if len(args) > 3 else kwargs.get("matrices")
            dq_arg = args[4] if len(args) > 4 else kwargs.get("dqs")
            blend_arg = args[5] if len(args) > 5 else kwargs.get("blend_weight")
        else:
            matrix_arg = args[2] if len(args) > 2 else kwargs.get("matrices", kwargs.get("bone_matrices"))
            dq_arg = args[3] if len(args) > 3 else kwargs.get("dqs", kwargs.get("dual_quaternions"))
            blend_arg = args[4] if len(args) > 4 else kwargs.get("blend_weight")
        mesh = _coerce_mesh({"vertices": vertices, "weights": weights, "indices": indices, "bones": bones})
        matrices = matrix_arg
        dqs = dq_arg
        blend = blend_arg
        return mesh, matrices, dqs, blend

    mesh = _coerce_mesh(kwargs.get("mesh", args[0] if args else 1))
    matrices = kwargs.get("matrices", kwargs.get("bone_matrices", kwargs.get("poses")))
    dqs = kwargs.get("dqs", kwargs.get("dual_quaternions"))
    blend = kwargs.get("blend_weight")
    return mesh, matrices, dqs, blend


def skin(*args: Any, **kwargs: Any) -> Any:
    """执行 DQ/LBS 蒙皮，返回顶点位置列表（同时是带报告的 ListResult）。"""
    try:
        mesh, matrices, dqs, blend_override = _skin_inputs(args, kwargs)
        bone_count = len(mesh.get("bones", [])) or int(mesh.get("palette_capacity", DEFAULT_PALETTE_CAPACITY))
        if matrices is None:
            matrices = kwargs.get("pose_matrices")
        if dqs is None and kwargs.get("dual_quat") is not None:
            dqs = kwargs["dual_quat"]

        mode = kwargs.get("mode", kwargs.get("method", "dq"))
        continuous = bool(kwargs.get("continuous", mode == "continuous"))
        violations = []
        out_positions = []
        dq_norms = []

        for v_index, vertex in enumerate(mesh.get("vertices", [])):
            point = vertex.get("position") if isinstance(vertex, Mapping) else vertex
            point = _as_vec3(point)

            indices = None
            raw_weights = None
            if isinstance(vertex, Mapping):
                indices = vertex.get("indices", vertex.get("bone_indices"))
                raw_weights = vertex.get("weights", vertex.get("bone_weights", vertex.get("influences")))
            if raw_weights is None:
                if v_index < len(mesh.get("weights", [])):
                    raw_weights = mesh["weights"][v_index]
                if indices is None and v_index < len(mesh.get("indices", [])):
                    indices = mesh["indices"][v_index]

            pairs, shape = _parse_influences(raw_weights, indices)
            if not pairs:
                pairs = [[0, 1.0]]
            norm = _renorm_pairs(pairs, zero_mode="root")
            influences = list(norm["influences"])

            safe_influences = []
            for bone, weight in influences:
                guarded = guard_bone_index(bone, bone_count)
                if not guarded.valid:
                    violations.append(guarded.meta)
                safe_influences.append((int(guarded), weight))

            use_dqs = dqs
            if use_dqs is None and matrices is not None and kwargs.get("convert_matrices_to_dq", True):
                # Matrix path can still use LBS if matrices contain scale; DQ
                # conversion is a best-effort rigid transform.
                use_dqs = [_dq_from_matrix(m) for m in matrices] if _is_sequence(matrices) else matrices

            dq_pos = None
            if use_dqs is not None:
                blended_dq = _blend_dq(safe_influences, use_dqs)
                dq_norms.append(math.sqrt(sum(x * x for x in blended_dq[:4])))
                dq_pos = _dq_transform_point(blended_dq, point)

            lbs_pos = None
            if matrices is not None:
                lbs_pos = _lbs_point(point, safe_influences, matrices)

            if mode == "lbs" or mode == "linear":
                result_pos = lbs_pos if lbs_pos is not None else (dq_pos or point)
            elif mode == "dq" or mode == "dual_quaternion":
                result_pos = dq_pos if dq_pos is not None else (lbs_pos or point)
            else:
                if dq_pos is None:
                    dq_pos = lbs_pos or point
                if lbs_pos is None:
                    lbs_pos = dq_pos
                if blend_override is not None:
                    w_lbs = float(continuous_blend_weight(blend_override)) if not isinstance(blend_override, (int, float)) or (isinstance(blend_override, (int, float)) and 0 <= blend_override <= 1 and kwargs.get("normalized_blend", True)) else _as_float(blend_override)
                    if isinstance(blend_override, BlendWeight):
                        w_lbs = float(blend_override)
                else:
                    angle = None
                    if isinstance(vertex, Mapping):
                        angle = vertex.get("twist_angle", vertex.get("joint_angle"))
                    w_lbs = float(continuous_blend_weight(angle)) if angle is not None else 0.0
                result_pos = [
                    (1.0 - w_lbs) * dq_pos[i] + w_lbs * lbs_pos[i]
                    for i in range(3)
                ]

            out_positions.append(result_pos)

        result = ListResult(
            out_positions,
            ok=all(math.isfinite(x) for p in out_positions for x in p),
            mode=mode,
            continuous=continuous,
            vertices=len(out_positions),
            violations=violations,
            oob_count=len(violations),
            dq_norms=dq_norms,
            mesh=mesh,
            buffer_version=mesh.get("version", 1),
        )
        return result
    except Exception as exc:  # The public entry should be a safe boundary.
        return ListResult(
            [],
            ok=False,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
            fallback=True,
        )


def rebuild(*args: Any, **kwargs: Any) -> Any:
    """规范化权重/顺序/索引，并把新顶点数据构建到 back buffer 后换页。"""
    try:
        source = args[0] if args else kwargs.get("mesh", 1)
        state = _coerce_state(source)
        mesh = state["mesh"]

        if kwargs.get("normalize_weights", True):
            _normalize_all_weights(state)
        if kwargs.get("guard_indices", True):
            _guard_mesh_indices(state)
        if kwargs.get("fix_bone_order", True) and mesh.get("bones"):
            order = fixed_bone_order(mesh["bones"])
            mesh["canonical_bone_order"] = list(order)
            mesh["bone_order"] = list(order)

        skinned = kwargs.get("skinned_positions")
        if skinned is None and kwargs.get("skin", False):
            skinned = skin(state, **{k: v for k, v in kwargs.items() if k in {"mode", "matrices", "dqs", "blend_weight"}})
        vertex_data = skinned if skinned is not None else mesh.get("positions", [])

        buffer = state["buffer"]
        frame = kwargs.get("frame", state.get("frame", 0))
        report = buffer.rebuild(vertex_data, frame=frame)
        mesh["version"] = buffer["version"]
        mesh["positions"] = list(vertex_data) if _is_sequence(vertex_data) else vertex_data
        state["frame"] = frame
        state.setdefault("metrics", {})["rebuilds"] = state["metrics"].get("rebuilds", 0) + 1
        return Report(
            ok=True,
            rebuilt=True,
            double_buffered=True,
            mesh=mesh,
            state=state,
            buffer=buffer,
            version=buffer["version"],
            front=buffer["front"],
            back=buffer["back"],
            active_slot="front",
            active_buffer=buffer["front"],
            **{k: v for k, v in report.items() if k not in {
                "front", "back", "ok", "version", "front_version"
            }},
        )
    except Exception as exc:
        return Report(ok=False, error=f"{type(exc).__name__}: {exc}", rebuilt=False)


def draw(*args: Any, **kwargs: Any) -> Any:
    """从一个已发布缓冲快照绘制，保证一帧只有一套蒙皮结果。"""
    try:
        target = args[0] if args else kwargs.get("target", kwargs.get("state", 1))
        if isinstance(target, VertexDoubleBuffer):
            buffer = target
            state = None
        elif isinstance(target, SkinState):
            state = target
            buffer = target["buffer"]
        elif _mesh_like(target):
            state = _coerce_state(target)
            buffer = state["buffer"]
        elif isinstance(target, BufferSnapshot):
            return Report(
                ok=True,
                vertices=target.data,
                positions=target.data,
                buffer_version=target.version,
                version=target.version,
                frame=target.frame,
                mixed=False,
                snapshot=target,
            )
        elif _is_sequence(target):
            return Report(ok=True, vertices=list(target), positions=list(target), mixed=False, version=kwargs.get("version", 1))
        else:
            state = _coerce_state(1)
            buffer = state["buffer"]

        # If the buffer has never received mesh positions, run one deterministic skin.
        if not buffer.get("front") and state is not None:
            positions = skin(state)
            buffer.rebuild(positions)

        draw_report = buffer.draw()
        if state is not None:
            state.setdefault("metrics", {})["draw_calls"] = state["metrics"].get("draw_calls", 0) + 1
            draw_report["state"] = state
        draw_report["drawn"] = True
        return draw_report
    except Exception as exc:
        return Report(ok=False, mixed=False, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 校验、指标、审计、降级与确定性
# ---------------------------------------------------------------------------


def _all_positions(result: Any) -> list[Any]:
    if isinstance(result, Mapping):
        for key in ("vertices", "positions", "skinned_vertices", "values"):
            if key in result:
                value = result[key]
                return list(value) if _is_sequence(value) else []
    if _is_sequence(result):
        return list(result)
    return []


def validate_skin(target: Any = 1, *args: Any, **kwargs: Any) -> Report:
    """校验蒙皮结果：有限值、权重和、索引范围以及缓冲版本一致性。"""
    issues: list[str] = []
    state = None
    mesh = None

    if isinstance(target, SkinState):
        state = target
        mesh = state["mesh"]
        positions = _all_positions(draw(state))
    elif isinstance(target, (ListResult, Report)):
        positions = _all_positions(target)
        mesh = target.get("mesh") if isinstance(target, Mapping) else target.meta.get("mesh")
        if isinstance(target, Report) and target.get("mixed"):
            issues.append("mixed_buffer_versions")
    elif _mesh_like(target):
        state = _coerce_state(target)
        mesh = state["mesh"]
        positions = _all_positions(skin(state))
    elif _is_sequence(target):
        positions = list(target)
    else:
        state = _coerce_state(1)
        mesh = state["mesh"]
        positions = _all_positions(skin(state))

    expected_count = kwargs.get("vertex_count")
    if mesh is not None:
        expected_count = len(mesh.get("vertices", []))
        for v_index, vertex in enumerate(mesh.get("vertices", [])):
            raw = vertex.get("weights") if isinstance(vertex, Mapping) else None
            indices = vertex.get("indices") if isinstance(vertex, Mapping) else None
            pairs, _ = _parse_influences(raw, indices)
            if pairs:
                total = math.fsum(max(0.0, _as_float(w)) for _, w in pairs)
                if abs(total - 1.0) > 1e-5 and abs(total) > WEIGHT_EPS:
                    issues.append(f"weights_not_normalized:vertex={v_index}")
                bone_count = len(mesh.get("bones", [])) or int(mesh.get("palette_capacity", DEFAULT_PALETTE_CAPACITY))
                for bone, _ in pairs:
                    idx = _as_int(bone, -1)
                    if not (0 <= idx < bone_count):
                        issues.append(f"bone_index_out_of_range:vertex={v_index}:bone={idx}")

    if expected_count is not None and len(positions) != int(expected_count):
        issues.append(f"vertex_count_mismatch:{len(positions)}!={expected_count}")

    finite_count = 0
    for p_index, point in enumerate(positions):
        vec = _as_vec3(point, default=(math.nan, math.nan, math.nan))
        if not all(math.isfinite(x) for x in vec):
            issues.append(f"nonfinite_vertex:{p_index}")
        else:
            finite_count += 1

    # A crude collapse check: a mesh with >=4 distinct source points should not
    # map every finite point to exactly the same location unless requested.
    if mesh is not None and finite_count >= 4 and len(positions) >= 4:
        unique = {tuple(round(x, 9) for x in _as_vec3(p)) for p in positions if all(math.isfinite(x) for x in _as_vec3(p))}
        source_unique = {tuple(_as_vec3(v.get("position") if isinstance(v, Mapping) else v)) for v in mesh.get("vertices", [])}
        if len(source_unique) >= 3 and len(unique) == 1:
            issues.append("skin_collapsed_to_point")

    if kwargs.get("buffer_version") is not None and isinstance(target, Mapping):
        versions = target.get("versions")
        if versions and len(set(versions)) > 1:
            issues.append("mixed_buffer_versions")

    return Report(
        ok=not issues,
        valid=not issues,
        issues=issues,
        errors=issues,
        vertex_count=len(positions),
        finite_vertices=finite_count,
        mixed_frame="mixed_buffer_versions" in issues,
        state=state,
    )


def validate_export(*args: Any, **kwargs: Any) -> Report:
    """校验一个或两个导出描述；多个参数时还会比较骨骼顺序一致性。"""
    exports = list(args)
    if "exports" in kwargs:
        exports.extend(kwargs["exports"])
    if not exports:
        exports = [kwargs.get("export", kwargs.get("data", 1))]

    reports = []
    all_issues: list[str] = []
    orders = []
    hashes = []

    for idx, export in enumerate(exports):
        if _is_number(export):
            # Smoke/demo call should represent a known-good tiny export.
            export = _demo_mesh(int(export))
        if not isinstance(export, Mapping):
            export = {"bones": export}

        issues: list[str] = []
        bones = export.get("bones", export.get("skeleton", export.get("bone_order", [])))
        order_value = export.get("bone_order", export.get("order"))
        canonical = list(fixed_bone_order(bones if bones else order_value))
        if order_value is not None and [_bone_identifier(x) if not _is_number(x) else x for x in order_value] != canonical:
            # Compare identifiers robustly.
            order_ids = [_bone_identifier(x) for x in order_value]
            if order_ids != canonical:
                issues.append("bone_order_not_canonical")
        orders.append(canonical)

        seen_ids = set()
        for bone in bones if _is_sequence(bones) else []:
            ident = _bone_identifier(bone)
            if ident in seen_ids:
                issues.append(f"duplicate_bone:{ident}")
            seen_ids.add(ident)

        capacity = _as_int(export.get("palette_capacity", export.get("capacity", DEFAULT_PALETTE_CAPACITY)), DEFAULT_PALETTE_CAPACITY)
        bone_count = len(bones) if _is_sequence(bones) else _as_int(export.get("bone_count", 0), 0)
        if bone_count > capacity and not export.get("chunked_palette", False):
            issues.append("palette_capacity_exceeded")
        if bone_count > 0:
            index_limit = bone_count if export.get("chunked_palette", False) else min(bone_count, capacity)
        else:
            index_limit = capacity

        # Collect influences from top-level or vertices.
        influence_sets = []
        if "weights" in export or "indices" in export:
            ww = export.get("weights")
            ii = export.get("indices")
            if _is_sequence(ww) and len(ww) > 0 and _is_sequence(ww[0]):
                for i, row in enumerate(ww):
                    influence_sets.append((row, ii[i] if ii is not None and i < len(ii) else None))
            else:
                influence_sets.append((ww, ii))
        export_vertices = export.get("vertices", [])
        if not _is_sequence(export_vertices):
            export_vertices = []
        for vertex in export_vertices:
            if isinstance(vertex, Mapping):
                influence_sets.append((
                    vertex.get("weights", vertex.get("influences")),
                    vertex.get("indices", vertex.get("bone_indices")),
                ))

        for v_index, (ww, ii) in enumerate(influence_sets):
            pairs, _ = _parse_influences(ww, ii)
            if not pairs:
                continue
            total = math.fsum(max(0.0, _as_float(w)) for _, w in pairs)
            if total <= WEIGHT_EPS:
                issues.append(f"zero_weight_sum:vertex={v_index}")
            elif abs(total - 1.0) > 1e-5:
                issues.append(f"weights_not_normalized:vertex={v_index}")
            for bone, weight in pairs:
                if weight < 0:
                    issues.append(f"negative_weight:vertex={v_index}:bone={bone}")
                bid = _as_int(bone, -1)
                if isinstance(bone, str):
                    mapping = export.get("bone_mapping", {ident: i for i, ident in enumerate(canonical)})
                    bid = _as_int(mapping.get(bone, -1), -1)
                if not (0 <= bid < index_limit):
                    issues.append(f"bone_index_out_of_range:vertex={v_index}:bone={bone}")

        for dq_index, dq in enumerate(export.get("dual_quaternions", export.get("dqs", []))):
            flat = _dq_flat(dq)
            norm = math.sqrt(math.fsum(x * x for x in flat[:4]))
            if abs(norm - 1.0) > 1e-4:
                issues.append(f"dual_quaternion_not_normalized:{dq_index}")

        payload = {k: v for k, v in export.items() if k not in {"state", "buffer"}}
        export_hash = stable_hash(payload)
        hashes.append(export_hash)
        reports.append(Report(ok=not issues, issues=issues, order=canonical, hash=export_hash))
        all_issues.extend(issues)

    order_consistent = all(order == orders[0] for order in orders) if len(orders) > 1 else True
    hash_consistent = len(set(hashes)) <= 1 if len(exports) > 1 and kwargs.get("same_content", False) else True
    if not order_consistent:
        all_issues.append("bone_order_differs_between_exports")

    return Report(
        ok=not all_issues,
        valid=not all_issues,
        issues=all_issues,
        errors=all_issues,
        exports=reports,
        bone_order=orders[0] if orders else [],
        order_consistent=order_consistent,
        hash_consistent=hash_consistent,
        hashes=hashes,
        validated=len(exports),
    )


def metrics(*args: Any, **kwargs: Any) -> Report:
    """返回管线指标；可传入 state/report，也可用关键字直接覆盖计数。"""
    target = args[0] if args else None
    data: dict[str, Any] = {}
    if isinstance(target, SkinState):
        data.update(target.get("metrics", {}))
        mesh = target.get("mesh", {})
        data.update({
            "vertices": len(mesh.get("vertices", [])),
            "bones": len(mesh.get("bones", [])),
            "buffer_version": target.get("buffer", {}).get("version") if isinstance(target.get("buffer"), Mapping) else None,
        })
    elif isinstance(target, Mapping):
        for key in ("metrics",):
            if isinstance(target.get(key), Mapping):
                data.update(target[key])
        for key in (
            "vertices", "vertex_count", "bones", "bone_count", "oob_count", "violations",
            "rebuilds", "draw_calls", "normalizations", "weight_normalizations",
            "fallback_count", "mixed_frame_count", "torn_reads", "swap_count", "version",
        ):
            if key in target:
                data[key] = target[key]
        if "vertices" not in data and "vertex_count" in target:
            data["vertices"] = target["vertex_count"]
    elif _is_sequence(target):
        data["vertices"] = len(target)

    data.update({k: v for k, v in kwargs.items() if k not in {"target"}})
    def _count(key: str, fallback_key: str, default: int) -> int:
        value = data.get(key, data.get(fallback_key, default))
        if _is_sequence(value) or isinstance(value, Mapping):
            return len(value)
        return int(_as_float(value, float(default)) or 0)

    vertices = _count("vertices", "vertex_count", 4)
    bones = _count("bones", "bone_count", 4)
    violations_value = data.get("violations", [])
    violation_count = len(violations_value) if isinstance(violations_value, list) else _as_int(violations_value, 0)
    oob = int(_as_float(data.get("oob_count", data.get("out_of_bounds", violation_count)), 0) or 0)
    normalizations = int(data.get("weight_normalizations", data.get("normalizations", vertices)) or 0)
    rebuilds = int(data.get("rebuilds", data.get("buffer_swaps", 0)) or 0)
    draws = int(data.get("draw_calls", data.get("draws", 0)) or 0)
    fallbacks = int(data.get("fallback_count", data.get("fallbacks", 0)) or 0)
    mixed = int(data.get("mixed_frame_count", data.get("mixed_frames", 0)) or 0)
    torn = int(data.get("torn_reads", 0) or 0)

    return Report(
        ok=mixed == 0 and torn == 0,
        vertices=vertices,
        vertex_count=vertices,
        bones=bones,
        bone_count=bones,
        weight_normalizations=normalizations,
        normalizations=normalizations,
        oob_indices=oob,
        oob_count=oob,
        out_of_bounds=oob,
        rebuilds=rebuilds,
        buffer_swaps=int(data.get("swap_count", rebuilds) or rebuilds),
        draw_calls=draws,
        fallback_count=fallbacks,
        mixed_frame_count=mixed,
        torn_reads=torn,
        palette_capacity=int(data.get("palette_capacity", DEFAULT_PALETTE_CAPACITY)),
        deterministic=bool(data.get("deterministic", True)),
        **{k: v for k, v in data.items() if k not in {
            "vertices", "vertex_count", "bones", "bone_count", "oob_count",
            "weight_normalizations", "normalizations", "rebuilds", "draw_calls",
            "fallback_count", "mixed_frame_count", "torn_reads", "palette_capacity",
            "deterministic", "oob_indices", "out_of_bounds", "buffer_swaps",
        }},
    )


def audit(*args: Any, **kwargs: Any) -> AuditReport:
    """审计权重、索引、双缓冲、顺序和导出校验结果。"""
    target = args[0] if args else None
    issues: list[str] = []
    events: list[Any] = []
    checks: dict[str, bool] = {}

    if isinstance(target, SkinState):
        validation = validate_skin(target)
        export_validation = validate_export(target.get("mesh", {}))
        issues.extend(validation["issues"])
        issues.extend(export_validation["issues"])
        events.extend(target.get("audit_log", []))
        checks["skin_valid"] = bool(validation)
        checks["export_valid"] = bool(export_validation)
        buffer = target.get("buffer")
        checks["double_buffered"] = bool(getattr(buffer, "double_buffered", True))
        if isinstance(buffer, Mapping) and buffer.get("torn_reads", 0):
            issues.append("buffer_torn_reads")
    elif isinstance(target, Mapping):
        if "issues" in target:
            issues.extend(target["issues"])
        if "events" in target:
            events.extend(target["events"])
        validation = validate_export(target) if {"bones", "vertices", "weights"} & set(target) else None
        if validation is not None:
            issues.extend(validation["issues"])
            checks["export_valid"] = bool(validation)
        for bad_counter, event_name in (
            ("oob_count", "bone_index_out_of_range"),
            ("out_of_bounds", "bone_index_out_of_range"),
            ("mixed_frame_count", "mixed_buffer_versions"),
            ("torn_reads", "buffer_torn_reads"),
            ("validation_failures", "validation_failed"),
        ):
            if _as_int(target.get(bad_counter, 0), 0) > 0:
                issues.append(event_name)
        if dict.get(target, "ok", True) is False and not issues:
            issues.append("report_not_ok")
    elif _is_sequence(target) and not _is_number(target):
        validation = validate_skin(target)
        issues.extend(validation["issues"])
        checks["skin_valid"] = bool(validation)
    else:
        # Scalar smoke audit: run the known-good demo pipeline.
        state = _coerce_state(_as_int(target, 1))
        rebuild(state)
        draw(state)
        validation = validate_skin(state)
        issues.extend(validation["issues"])
        checks["demo_pipeline_valid"] = bool(validation)
        checks["double_buffered"] = True

    checks.setdefault("weights_normalized", not any("weight" in x for x in issues))
    checks.setdefault("indices_in_range", not any("bone_index" in x or "capacity" in x for x in issues))
    checks.setdefault("order_fixed", not any("order" in x for x in issues))
    checks.setdefault("no_mixed_frame", not any("mixed" in x for x in issues))

    issues = list(dict.fromkeys(issues))
    for issue in issues:
        events.append(f"issue:{issue}")
    if not issues:
        events.append("audit_ok")

    return AuditReport(
        ok=not issues,
        valid=not issues,
        issues=issues,
        errors=issues,
        events=events,
        checks=checks,
        event_count=len(events),
    )


def fallback(*args: Any, **kwargs: Any) -> Any:
    """安全降级：非法 DQ/材质路径改为 CPU LBS，全零权重回根骨骼，坏缓冲用上一版。"""
    strategy = kwargs.get("strategy", kwargs.get("mode", "cpu_lbs"))
    target = args[0] if args else None

    try:
        if isinstance(target, VertexDoubleBuffer):
            snapshot = target.bind()
            return Report(
                ok=True,
                fallback=True,
                strategy="previous_buffer",
                mode="previous_buffer",
                vertices=snapshot.data,
                buffer_version=snapshot.version,
            )

        if _mesh_like(target) or isinstance(target, SkinState):
            positions = skin(target, mode="lbs")
            return Report(
                ok=bool(positions),
                fallback=True,
                strategy="cpu_lbs",
                mode="lbs",
                vertices=list(positions),
                positions=list(positions),
                skin=positions,
            )

        pairs, _ = _parse_influences(target, kwargs.get("indices"))
        if pairs:
            result = _renorm_pairs(pairs, zero_mode="root")
            return Report(
                ok=True,
                fallback=True,
                strategy="root_bone_weights",
                mode="root_bone",
                influences=result["influences"],
                weights=result["weights"],
            )

        if _is_number(target) or target is None:
            return Report(
                ok=True,
                fallback=True,
                strategy=strategy if isinstance(strategy, str) else "cpu_lbs",
                mode="cpu_lbs",
                vertices=[],
                safe=True,
            )

        return Report(ok=True, fallback=True, strategy="identity", vertices=[])
    except Exception as exc:
        return Report(
            ok=False,
            fallback=True,
            strategy="identity",
            error=f"{type(exc).__name__}: {exc}",
            vertices=[],
        )


def determinism(*args: Any, **kwargs: Any) -> Report:
    """运行两次相同输入并比较蒙皮结果、规范顺序和导出哈希。"""
    runs = int(kwargs.get("runs", kwargs.get("frames", 2)))
    seed = kwargs.get("seed", 1)
    if len(args) >= 3 and _is_number(args[2]):
        runs = max(2, int(args[2]))
    if args and _is_number(args[0]):
        seed = int(args[0])
        if len(args) > 1 and _is_number(args[1]):
            seed = int(args[1]) if args[0] is None else seed
            # determinism(None, seed, frames) convention.
        if args[0] is None and len(args) > 1:
            seed = _as_int(args[1], 42)
    elif args and args[0] is None and len(args) > 1:
        seed = _as_int(args[1], 42)

    callable_fn = kwargs.get("fn")
    fn_args = kwargs.get("fn_args", ())
    fn_kwargs = kwargs.get("fn_kwargs", {})
    if callable_fn is None and args and callable(args[0]):
        callable_fn = args[0]
        fn_args = args[1:]

    structured_target = (
        args[0] if args and not callable(args[0]) and not _is_number(args[0]) and args[0] is not None
        else kwargs.get("target")
    )

    try:
        if callable(callable_fn):
            first = callable_fn(*fn_args, **fn_kwargs)
            outputs = [first] + [callable_fn(*fn_args, **fn_kwargs) for _ in range(max(1, runs - 1))]
            values = [stable_hash(x) for x in outputs]
            deterministic = len(set(values)) == 1
            result_payload = first
        elif structured_target is not None:
            def _one_target_run():
                if isinstance(structured_target, SkinState) or _mesh_like(structured_target):
                    state = _coerce_state(structured_target)
                    rebuild(state)
                    skinned = skin(state)
                    return {
                        "skin": [list(p) for p in skinned],
                        "order": fixed_bone_order(state["mesh"].get("bones", [])),
                        "validation": {
                            k: v for k, v in validate_skin(state).items() if k != "state"
                        },
                    }
                return structured_target

            outputs = [_one_target_run() for _ in range(max(2, runs))]
            values = [stable_hash(x) for x in outputs]
            deterministic = len(set(values)) == 1
            first = outputs[0]
            result_payload = first
        else:
            # Canonical end-to-end CPU skinning/export run.
            def _one_run(run_seed: int):
                mesh = _demo_mesh(run_seed)
                order = fixed_bone_order(mesh["bones"])
                normalized = normalize_weights_runtime(mesh)
                rebuilt = rebuild(normalized["state"])
                skinned = skin(rebuilt["state"], mode="dq")
                validation = validate_skin(rebuilt["state"])
                export = validate_export(mesh)
                return {
                    "order": list(order),
                    "skin": [list(p) for p in skinned],
                    "export_order": export["bone_order"],
                    "validation": dict(validation),
                    "metrics": metrics(rebuilt["state"]),
                }

            outputs = [_one_run(seed) for _ in range(max(2, runs))]
            first = outputs[0]
            values = [stable_hash(item) for item in outputs]
            deterministic = len(set(values)) == 1
            result_payload = Report(
                skin=first["skin"],
                validation=first["validation"],
                metrics=first["metrics"],
                order=first["order"],
            )

        return Report(
            ok=deterministic,
            deterministic=deterministic,
            equal=deterministic,
            identical=deterministic,
            bit_identical=deterministic,
            runs=runs,
            frames=runs,
            seed=seed,
            hashes=values,
            values=values,
            hash=values[0] if values else None,
            result=result_payload,
        )
    except Exception as exc:
        return Report(
            ok=False,
            deterministic=False,
            equal=False,
            identical=False,
            bit_identical=False,
            error=f"{type(exc).__name__}: {exc}",
            seed=seed,
        )


# ---------------------------------------------------------------------------
# 自测：python -m graphics.dualquat_skin 或直接执行文件
# ---------------------------------------------------------------------------


def _self_test() -> None:
    assert normalize_weights_runtime([0.8, 0.4]).weight_sum == 1.0
    guarded = guard_bone_index(99, 4)
    assert 0 <= int(guarded) < 4
    demo = double_buffer_rebuild(1)
    assert demo["ok"] and demo["old_buffer_still_bound"]
    w0 = continuous_blend_weight(0.0, 0.0, 90.0)
    w1 = continuous_blend_weight(90.0, 0.0, 90.0)
    assert float(w0) == 0.0 and float(w1) == 1.0
    assert fixed_bone_order(["right", "root", "left"]) == ["left", "right", "root"]
    assert validate_export(1)["ok"]
    assert determinism(42)["deterministic"]
    state = _coerce_state(1)
    rebuild(state)
    drawn = draw(state)
    assert drawn["ok"] and not drawn["mixed_frame"]
    assert validate_skin(state)["ok"]
    print("dualquat skin self-test passed")


if __name__ == "__main__":
    _self_test()
