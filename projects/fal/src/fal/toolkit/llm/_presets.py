import re
from typing import Dict, List, Optional, Sequence

# Preset flag per engine. Every flag here takes a value except the all-reduce
# switch (see _BOOLEAN_FLAGS).
_TP_FLAG = {"vllm": "--tensor-parallel-size", "sglang": "--tp-size"}
_MAX_SEQS_FLAG = {"vllm": "--max-num-seqs", "sglang": "--max-running-requests"}
_ALL_REDUCE_FLAG = "--disable-custom-all-reduce"
_BOOLEAN_FLAGS = frozenset({_ALL_REDUCE_FLAG})

# Spellings that mean the same thing to an engine, keyed by a canonical group.
_FLAG_GROUPS: Dict[str, str] = {
    "--tensor-parallel-size": "tp",
    "--tp-size": "tp",
    "--tp": "tp",
    "-tp": "tp",
    "--max-num-seqs": "max_seqs",
    "--max-running-requests": "max_seqs",
    _ALL_REDUCE_FLAG: "all_reduce",
    "--pipeline-parallel-size": "pp",
    "--pp-size": "pp",
    "-pp": "pp",
    "--data-parallel-size": "dp",
    "--dp-size": "dp",
    "--dp": "dp",
    "-dp": "dp",
}
# A user split across pipeline or data parallel ranks owns the GPU layout, so
# the tensor parallel preset (all GPUs) is dropped.
_LAYOUT_GROUPS = frozenset({"tp", "pp", "dp"})

# GPUs without NVLink between cards (PCIe only), matched against the name
# `fal.toolkit.compilation.get_gpu_type()` returns. That function keeps only
# the first word after "NVIDIA ", so RTX PRO 6000, RTX 4090 and RTX 5090 all
# arrive as "RTX" and GeForce boards as "GeForce". A100 must not match A10,
# hence exact names for the short ones.
_NO_NVLINK_PREFIXES = ("RTX", "GEFORCE", "L40")
_NO_NVLINK_EXACT = frozenset({"A10", "A10G", "L4"})


def _normalize_gpu(gpu_type: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", gpu_type.upper())


def _is_pcie_only(gpu_type: str) -> bool:
    name = _normalize_gpu(gpu_type)
    return name in _NO_NVLINK_EXACT or name.startswith(_NO_NVLINK_PREFIXES)


def engine_preset_args(
    engine: str,
    gpu_type: str,
    num_gpus: int,
    max_num_seqs: Optional[int] = None,
) -> List[str]:
    """Engine arguments that match the machine, for ``vllm`` or ``sglang``.

    - ``num_gpus > 1``: tensor parallel across all GPUs (``--tensor-parallel-size``
      for vLLM, ``--tp-size`` for SGLang).
    - ``num_gpus > 1`` on PCIe-only GPUs (RTX PRO 6000, RTX 4090/5090, L40/L40S,
      L4, A10): ``--disable-custom-all-reduce``, since the custom all-reduce
      kernel needs NVLink.
    - ``max_num_seqs``: ``--max-num-seqs`` (vLLM) or ``--max-running-requests``
      (SGLang). Pass the app's ``max_multiplexing`` so the router sends exactly
      what the engine can run at once.

    ``gpu_type`` is what ``fal.toolkit.compilation.get_gpu_type()`` returns.
    Use ``merge_engine_args`` to put user arguments on top.
    """
    if engine not in _TP_FLAG:
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")
    args: List[str] = []
    if num_gpus > 1:
        args += [_TP_FLAG[engine], str(num_gpus)]
        if _is_pcie_only(gpu_type):
            args.append(_ALL_REDUCE_FLAG)
    if max_num_seqs is not None:
        args += [_MAX_SEQS_FLAG[engine], str(max_num_seqs)]
    return args


def _flag_group(token: str) -> Optional[str]:
    if not token.startswith("-"):
        return None
    name = token.split("=", 1)[0].replace("_", "-")
    # vLLM accepts --no-<flag> for boolean switches.
    if name.startswith("--no-"):
        name = "--" + name[5:]
    return _FLAG_GROUPS.get(name)


def _user_flag_value(args: Sequence[str], group: str) -> Optional[str]:
    """Value of the last ``--flag value`` or ``--flag=value`` in ``group``."""
    found: Optional[str] = None
    for i, token in enumerate(args):
        if _flag_group(token) != group:
            continue
        if "=" in token:
            found = token.split("=", 1)[1]
        elif i + 1 < len(args):
            found = args[i + 1]
    return found


def merge_engine_args(preset: Sequence[str], user: Sequence[str]) -> List[str]:
    """Preset arguments followed by user arguments, user winning per flag.

    A preset flag is dropped (with its value, unless it is a boolean switch)
    when the user arguments mention it as ``--flag value``, ``--flag=value``
    or bare ``--flag``. User arguments are never altered and come last.
    ``preset`` must come from ``engine_preset_args``.
    """
    user_groups = {g for g in map(_flag_group, user) if g}
    if user_groups & _LAYOUT_GROUPS:
        user_groups.add("tp")
    merged: List[str] = []
    i = 0
    while i < len(preset):
        flag = preset[i]
        width = 1 if flag in _BOOLEAN_FLAGS else 2
        if _flag_group(flag) not in user_groups:
            merged += preset[i : i + width]
        i += width
    return [*merged, *user]


def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def engine_arg_warnings(
    user: Sequence[str], num_gpus: int, max_multiplexing: Optional[int]
) -> List[str]:
    """Messages for user arguments that disagree with the app's resources."""
    messages: List[str] = []
    sizes = {g: _int_or_none(_user_flag_value(user, g)) for g in ("tp", "pp", "dp")}
    if any(v is not None for v in sizes.values()):
        ranks = 1
        for v in sizes.values():
            ranks *= v or 1
        if ranks != num_gpus:
            messages.append(
                f"engine_args use {ranks} GPUs (tensor x pipeline x data "
                f"parallel) but num_gpus is {num_gpus}"
            )
    seqs = _int_or_none(_user_flag_value(user, "max_seqs"))
    if seqs is not None and max_multiplexing is not None and seqs != max_multiplexing:
        messages.append(
            f"engine_args set max concurrent sequences {seqs} but "
            f"max_multiplexing is {max_multiplexing}; the router would "
            "starve the engine or queue requests inside it"
        )
    return messages
