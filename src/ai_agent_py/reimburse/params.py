"""参数库存储：内存缓存 + YAML 落盘。

- 读：首次从文件加载，之后走内存；
- 写：先校验 → 原子落盘 → 落盘成功后才替换内存（失败则内存保持旧值）；
- `invalidate()` 强制下次重新读文件，用于手工改了 YAML 后的热重载。
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).with_name("data")
DEFAULT_PATH = DATA_DIR / "policy_params.yaml"


class ParamError(ValueError):
    """参数不合法（结构 / 类型 / 范围）。"""


def _require_number(value: Any, name: str, minimum: float = 0.0) -> float:
    """参数校验：必须是数字（bool 不算）且不小于下限，否则抛 ParamError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamError(f"{name} 必须是数字")
    if value < minimum:
        raise ParamError(f"{name} 不能小于 {minimum}")
    return float(value)


def validate_params(params: dict[str, Any]) -> None:
    """按已知参数键做结构校验；未知键放行（便于扩展），但必须是可序列化的 JSON 值。"""
    if not isinstance(params, dict) or not params:
        raise ParamError("params 必须是非空对象")

    lodging = params.get("lodging_cap_per_night")
    if lodging is not None:
        if not isinstance(lodging, dict) or not lodging:
            raise ParamError("lodging_cap_per_night 必须是 {城市档: 金额} 对象")
        for tier, cap in lodging.items():
            _require_number(cap, f"lodging_cap_per_night.{tier}", 1)

    for key in ("entertainment_per_head", "invoice_amount_requires_contract"):
        if key in params:
            _require_number(params[key], key, 1)

    kinds = params.get("allowed_kinds")
    if kinds is not None:
        if not isinstance(kinds, dict) or not kinds:
            raise ParamError("allowed_kinds 必须是 {费用类型: [票种]} 对象")
        for category, items in kinds.items():
            if not isinstance(items, list) or not items:
                raise ParamError(f"allowed_kinds.{category} 必须是非空数组")
            if not all(isinstance(item, str) and item.strip() for item in items):
                raise ParamError(f"allowed_kinds.{category} 里的票种必须是非空字符串")


class PolicyParamsStore:
    """参数库的单一读写入口：进程内缓存 + 原子落盘。"""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        """绑定参数文件路径；文件内容留到第一次 ``load()`` 再读。"""
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] | None = None

    @property
    def path(self) -> Path:
        """当前参数文件路径（接口回给前端展示用）。"""
        return self._path

    def load(self) -> dict[str, Any]:
        """读参数库：首次读文件并缓存，之后直接返回缓存对象。"""
        with self._lock:
            if self._data is None:
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ParamError("参数文件格式不正确：顶层必须是对象")
                self._data = raw
            return self._data

    def invalidate(self) -> None:
        """让缓存失效：下次 ``load()`` 重新读文件（手工改 YAML 后的热重载）。"""
        with self._lock:
            self._data = None

    def save(self, params: dict[str, Any]) -> dict[str, Any]:
        """校验 → 写临时文件 → 原子替换 → 更新缓存。"""
        validate_params(params)
        with self._lock:
            current = self.load()
            data = dict(current)
            data["params"] = params
            data["updated_at"] = datetime.now().isoformat(timespec="seconds")

            serialized = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
            tmp = self._path.with_suffix(".yaml.tmp")
            tmp.write_text(serialized, encoding="utf-8")
            os.replace(tmp, self._path)  # 原子替换：写一半失败也不会破坏原文件
            self._data = data
            return data


_store: PolicyParamsStore | None = None


def get_store() -> PolicyParamsStore:
    """进程级单例；测试可替换模块变量 `_store` 指向临时文件。"""
    global _store
    if _store is None:
        _store = PolicyParamsStore(DEFAULT_PATH)
    return _store
