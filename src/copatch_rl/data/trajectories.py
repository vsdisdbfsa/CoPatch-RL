"""Validation only; synthetic fixtures are never evidence of real policy sampling."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Pending agreement with B; all envelope assumptions are confined to adapt_event.
B_ALIGNMENT_FIELDS = (
    "event_id",
    "kind",
    "sequence",
    "timestamp",
    "episode_id",
    "task_id",
    "workspace_id",
    "group_id",
    "policy_version",
    "initial_hash",
    "role",
    "code_hash",
    "action_id",
    "visible_observation_ids",
    "synthetic",
    "input_ids",
    "generated_ids",
    "old_logprobs",
    "loss_mask",
    "authoritative workspace hash history keyed by (episode_id, sequence)",
)


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    sequence: int
    timestamp: float
    episode_id: str
    task_id: str
    workspace_id: str
    group_id: str
    policy_version: str
    initial_hash: str
    role: str
    code_hash: str
    action_id: str | None
    visible: tuple[str, ...]
    synthetic: bool
    sampling: tuple[tuple | None, ...]


def adapt_event(raw: Mapping) -> Event:
    """The sole B-envelope adapter; no Event Store or data production."""
    text_keys = (
        "event_id",
        "kind",
        "episode_id",
        "task_id",
        "workspace_id",
        "policy_version",
        "initial_hash",
        "role",
        "code_hash",
    )
    if any(not isinstance(raw.get(k), str) or not raw[k] for k in text_keys):
        raise ValueError("missing event identity")
    if raw["kind"] not in {"action", "observation"}:
        raise ValueError("unsupported event")
    if type(raw.get("sequence")) is not int or raw["sequence"] < 0:
        raise ValueError("invalid sequence")
    timestamp = raw.get("timestamp")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ValueError("invalid timestamp")
    if type(raw.get("synthetic")) is not bool:
        raise ValueError("explicit provenance required")
    group = raw.get("group_id", "")
    if not isinstance(group, str):
        raise ValueError("invalid group")
    action = raw.get("action_id")
    if raw["kind"] == "observation" and (not isinstance(action, str) or not action):
        raise ValueError("observation needs action ID")
    visible = raw.get("visible_observation_ids", ())
    if not isinstance(visible, (list, tuple)) or any(not isinstance(x, str) for x in visible):
        raise ValueError("invalid visible observations")
    sampling = []
    for key in ("input_ids", "generated_ids", "old_logprobs", "loss_mask"):
        value = raw.get(key)
        if value is not None and not isinstance(value, (list, tuple)):
            raise ValueError("invalid sampling array")
        sampling.append(None if value is None else tuple(value))
    return Event(
        raw["event_id"],
        raw["kind"],
        raw["sequence"],
        timestamp,
        raw["episode_id"],
        raw["task_id"],
        raw["workspace_id"],
        group,
        raw["policy_version"],
        raw["initial_hash"],
        raw["role"],
        raw["code_hash"],
        action,
        tuple(visible),
        raw["synthetic"],
        tuple(sampling),
    )


@dataclass(frozen=True)
class Validation:
    valid: bool
    errors: tuple[str, ...]
    sampling: dict[str, str]
    synthetic: bool


def validate(
    raw_events: Sequence[Mapping],
    workspace_hashes: Mapping[tuple[str, int], str],
    *,
    mode: str,
    label_ids: Sequence[str] = (),
) -> Validation:
    """Hash history must come from trusted workspace records, not event self-claims.

    Sampling availability means fields exist, not that probabilities are genuine.
    No labels, masks, trajectories or probabilities are produced or repaired.
    """
    if mode not in {"sft", "grpo"}:
        raise ValueError("mode must be sft or grpo")
    try:
        events = [adapt_event(raw) for raw in raw_events]
    except (ValueError, TypeError, KeyError):
        return Validation(False, ("INVALID_ENVELOPE",), {}, False)
    errors = set()
    if not events:
        errors.add("EMPTY_TRAJECTORY")
    ids = {e.event_id: e for e in events}
    if len(ids) != len(events):
        errors.add("DUPLICATE_EVENT")
    seen = {}
    pending = {}
    paired = set()
    episodes = {}
    availability = {}
    last = {}
    for e in events:
        identity = (
            e.task_id,
            e.workspace_id,
            e.group_id,
            e.policy_version,
            e.initial_hash,
            e.synthetic,
        )
        if e.episode_id in episodes and episodes[e.episode_id] != identity:
            errors.add("EPISODE_IDENTITY")
        episodes[e.episode_id] = identity
        previous = last.get(e.episode_id)
        if previous and (e.sequence <= previous[0] or e.timestamp < previous[1]):
            errors.add("EVENT_ORDER")
        last[e.episode_id] = (e.sequence, e.timestamp)
        expected = workspace_hashes.get((e.episode_id, e.sequence))
        if expected is None:
            errors.add("HASH_UNAVAILABLE")
        elif e.code_hash != expected:
            errors.add("STALE_HASH")
        if e.kind == "action":
            if e.episode_id in pending:
                errors.add("UNPAIRED_ACTION")
            pending[e.episode_id] = e.event_id
            for ref in e.visible:
                obs = seen.get(ref)
                if obs is None or obs.kind != "observation":
                    errors.add("FUTURE_FEEDBACK")
                elif obs.episode_id != e.episode_id or obs.workspace_id != e.workspace_id:
                    errors.add("PAIR_IDENTITY")
                elif obs.code_hash != e.code_hash:
                    errors.add("STALE_FEEDBACK")
            available = all(value is not None and len(value) > 0 for value in e.sampling)
            availability[e.event_id] = "available" if available else "unavailable"
            if available:
                inputs, tokens, probs, mask = e.sampling
                if (
                    len(tokens) != len(probs)
                    or len(tokens) != len(mask)
                    or any(type(x) is not int or x < 0 for x in (*inputs, *tokens))
                    or any(
                        type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in probs
                    )
                    or any(type(x) is not bool for x in mask)
                ):
                    errors.add("INVALID_SAMPLING_FIELDS")
        else:
            action = ids.get(e.action_id)
            if action is None or action.kind != "action":
                errors.add("UNPAIRED_OBSERVATION")
            else:
                if (action.episode_id, action.task_id, action.workspace_id) != (
                    e.episode_id,
                    e.task_id,
                    e.workspace_id,
                ):
                    errors.add("PAIR_IDENTITY")
                if (
                    action.event_id not in seen
                    or e.sequence <= action.sequence
                    or e.timestamp < action.timestamp
                ):
                    errors.add("OBSERVATION_BEFORE_ACTION")
                if action.event_id in paired:
                    errors.add("DUPLICATE_OBSERVATION")
                paired.add(action.event_id)
                if pending.get(e.episode_id) == action.event_id:
                    pending.pop(e.episode_id)
        seen[e.event_id] = e
    if any(e.kind == "action" and e.event_id not in paired for e in events):
        errors.add("UNPAIRED_ACTION")
    if len(set(label_ids)) != len(label_ids):
        errors.add("INVALID_SFT_LABEL")
    for label in label_ids:
        e = ids.get(label)
        if mode != "sft" or e is None or e.kind != "action" or e.role != "main":
            errors.add("INVALID_SFT_LABEL")
    if mode == "grpo":
        groups = {}
        for episode, identity in episodes.items():
            groups.setdefault(identity[2], []).append((episode, identity))
        for group, members in groups.items():
            if not group or len(members) != 4:
                errors.add("GROUP_SIZE")
            if len({i[1] for _, i in members}) != len(members):
                errors.add("WORKSPACE_REUSED")
            if len({(i[0], i[3], i[4], i[5]) for _, i in members}) != 1:
                errors.add("GROUP_IDENTITY")
    if len({e.synthetic for e in events}) > 1:
        errors.add("MIXED_PROVENANCE")
    return Validation(
        not errors, tuple(sorted(errors)), availability, any(e.synthetic for e in events)
    )
