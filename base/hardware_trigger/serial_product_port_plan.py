"""Shared validation and port boundaries for product serial triggering."""

from dataclasses import dataclass

from base.hardware_trigger.serial_full_frame_matcher import (
    normalize_frame_candidates,
    normalize_hex_frame,
)


@dataclass(frozen=True)
class SerialProductPortPlan:
    frames: tuple[str, ...]
    ports: tuple[tuple[int, ...], ...]
    idle_frame: str
    candidates: tuple[str, ...]


def build_serial_product_port_plan(conditions, idle_code=""):
    if not isinstance(idle_code, str):
        raise ValueError("端口切换空闲码必须是字符串")
    idle_frame = normalize_hex_frame(idle_code) if idle_code.strip() else ""
    frames = []
    port_indexes = {}
    for index, condition in enumerate(conditions):
        port_name = str(condition.get("group_name") or "默认端口")
        indexes = port_indexes.setdefault(port_name, [])
        frame = normalize_hex_frame(condition.get("trigger_state", ""))
        if any(frames[previous] == frame for previous in indexes):
            raise ValueError(f"{port_name}内状态码重复：{frame}")
        frames.append(frame)
        indexes.append(index)

    ports = tuple(tuple(indexes) for indexes in port_indexes.values())
    names = tuple(port_indexes)
    if idle_frame:
        if idle_frame in frames:
            raise ValueError(f"端口切换空闲码与档位状态码重复：{idle_frame}")
    else:
        for index in range(1, len(ports)):
            if frames[ports[index - 1][-1]] == frames[ports[index][0]]:
                raise ValueError(
                    f"{names[index - 1]}末档与{names[index]}首档状态码重复："
                    f"{frames[ports[index][0]]}；请配置端口切换空闲码或修改状态码"
                )
        if ports and frames[ports[-1][-1]] == frames[ports[0][0]]:
            raise ValueError(
                f"本轮末档与下一轮首档状态码相同：{frames[ports[0][0]]}；"
                "请配置空闲码或修改状态码"
            )
    candidates = list(dict.fromkeys(frames))
    if idle_frame:
        candidates.append(idle_frame)
    return SerialProductPortPlan(
        tuple(frames), ports, idle_frame, normalize_frame_candidates(candidates)
    )
