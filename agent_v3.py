"""Tier P0 v3 deterministic triage; demonstration policy, never production policy.

Supervised early-warning probability is the only WATCH signal. MAD is stored
as secondary novelty evidence and never independently creates an order. The
RUL thresholds, confirmation, hysteresis, deduplication and manual closure are
retained from the separately archived v1 implementation. No labels are read.

process_observation(event, state_dir, config=None) is the integration API.
Local JSON storage deliberately supports one writer at a time; an exclusive
lock rejects concurrent writers instead of silently overwriting actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TriageConfig:
    critical_rul: float = 30.0
    warning_rul: float = 60.0
    emergency_rul: float = 5.0
    confirmation_cycles: int = 2
    hysteresis_cycles: float = 10.0
    clearance_cycles: int = 3
    namespace: str = "teliport-s4-case1-v3-p0-demo"


SEVERITY_RANK = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3}

# A v3 work order carries the actual supervised signal. The old v1 anomaly
# fields are not relabelled probabilities; MAD retains its explicit own name.
WORK_ORDER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "work_order_id", "namespace", "asset", "episode",
                 "status", "severity", "action", "created_cycle", "last_observed_cycle",
                 "model_version", "evidence", "policy", "policy_scope", "history"],
    "properties": {
        "schema_version": {"const": 3},
        "work_order_id": {"type": "string", "pattern": "^[a-f0-9]{24}$"},
        "namespace": {"type": "string", "minLength": 1},
        "asset": {"type": "object", "required": ["subset", "engine_id"],
                  "properties": {"subset": {"enum": ["FD001", "FD002", "FD003", "FD004"]},
                                 "engine_id": {"type": "integer", "minimum": 1}}},
        "episode": {"type": "integer", "minimum": 1},
        "status": {"enum": ["OPEN", "CLOSED"]},
        "severity": {"enum": ["WATCH", "WARNING", "CRITICAL"]},
        "action": {"enum": ["INSPECTION", "MAINTENANCE"]},
        "created_cycle": {"type": "integer", "minimum": 1},
        "last_observed_cycle": {"type": "integer", "minimum": 1},
        "model_version": {"type": "string", "minLength": 1},
        "policy_scope": {"const": "DEMONSTRATION_ONLY_NOT_PRODUCTION"},
        "history": {"type": "array", "minItems": 1},
        "policy": {"type": "object"},
        "evidence": {"$ref": "#/$defs/evidence"},
        "latest_evidence": {"$ref": "#/$defs/evidence"},
    },
    "$defs": {"evidence": {
        "type": "object",
        "required": ["event_id", "predicted_rul", "warning_probability", "warning_threshold",
                     "warning_horizon", "warning_label_definition", "threshold_source",
                     "mad_score", "mad_threshold", "mad_role", "model_version", "input_hash"],
        "properties": {
            "event_id": {"type": "string", "minLength": 1},
            "predicted_rul": {"type": "number", "minimum": 0},
            "warning_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "warning_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "warning_horizon": {"enum": [20, 30, 50]},
            "warning_label_definition": {"type": "string"},
            "threshold_source": {"const": "TRAIN_ONLY_OOF_MAX_F2"},
            "mad_score": {"type": ["number", "null"], "minimum": 0},
            "mad_threshold": {"type": ["number", "null"], "minimum": 0},
            "mad_role": {"const": "SECONDARY_NOVELTY_ONLY"},
            "model_version": {"type": "string", "minLength": 1},
            "input_hash": {"type": "string", "minLength": 1},
        },
    }},
}


def _config(value: dict | TriageConfig | None) -> TriageConfig:
    if value is None:
        result = TriageConfig()
    elif isinstance(value, TriageConfig):
        result = value
    else:
        if "triage" in value:
            value = value["triage"]
        names = {f.name for f in fields(TriageConfig)}
        aliases = {"confirm_cycles": "confirmation_cycles", "hysteresis": "hysteresis_cycles"}
        values = {aliases.get(k, k): v for k, v in value.items()}
        result = TriageConfig(**{k: v for k, v in values.items() if k in names})
    if not (0 <= result.emergency_rul <= result.critical_rul < result.warning_rul):
        raise ValueError("Require 0 <= emergency_rul <= critical_rul < warning_rul")
    if result.confirmation_cycles < 1 or result.clearance_cycles < 1 or result.hysteresis_cycles < 0:
        raise ValueError("Confirmation/clearance must be positive; hysteresis nonnegative")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == payload:
            return  # Exact replay does not rewrite already-correct snapshots.
    except OSError:
        pass  # The bounded replacement below reports a persistent lock clearly.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            transient_lock = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}
            if not transient_lock or attempt == 9:
                raise OSError(f"Atomic write failed after {attempt + 1} attempt(s): {path}; recovery temp: {temporary}") from exc
            time.sleep(min(0.01 * (2 ** attempt), 0.2))


@contextmanager
def _exclusive(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = state_dir / ".agent.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Agent state is busy; only one writer is supported") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _load(state_dir: Path) -> dict:
    path = state_dir / "agent_state.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 3:
            raise ValueError("Unsupported agent state schema")
        return value
    return {"schema_version": 3, "engines": {}, "work_orders": {}, "exceptions": [],
            "counts": {"observations": 0, "data_quality": 0, "rejected": 0,
                       "orders_created": 0, "orders_updated": 0, "orders_closed": 0}}


def _persist(state_dir: Path, state: dict) -> None:
    # agent_state.json is authoritative. Other files are convenient snapshots.
    _write_json(state_dir / "agent_state.json", state)
    _write_json(state_dir / "work_orders.json", list(state["work_orders"].values()))
    _write_json(state_dir / "exceptions.json", state["exceptions"])


def _result(status: str, severity: str, action: str, reasons: list[str], state: dict,
            order: dict | None = None, human_review: bool = False,
            confirmed: bool = True, warnings: list[str] | None = None) -> dict:
    return {"status": status, "decision": {"severity": severity, "action": action,
            "reason_codes": reasons, "human_review": human_review, "confirmed": confirmed},
            "work_order": order, "warnings": warnings or [], "counts": dict(state["counts"])}


def _reject(state_dir: Path, state: dict, event: dict, reason: str) -> dict:
    state["counts"]["rejected"] += 1
    state["exceptions"].append({"reason": reason, "event": _json_safe(event), "human_review": True})
    _persist(state_dir, state)
    return _result("rejected", "DATA_QUALITY", "HUMAN_REVIEW", [reason], state, human_review=True)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def process_observation(event: dict, state_dir: str | Path,
                        config: dict | TriageConfig | None = None) -> dict:
    """Process one observation and persist its deterministic decision.

    Required fields: subset, engine_id, cycle, predicted_rul (or legacy rul),
    warning_probability, warning_threshold, warning_horizon, valid,
    model_version, input_hash. Optional: event_id (otherwise derived from
    subset/engine/cycle), mad_score, mad_threshold, run_id,
    top_features. MAD may be absent because it is secondary novelty evidence.
    Probability >= threshold activates WATCH (including equality, matching the
    classifier's OOF threshold convention). Negative/NaN/out-of-range warning
    probabilities are visible data-quality exceptions, never normal decisions.
    Exact historical replay is a no-op; changed event IDs at the same cycle are
    conflicting payloads. Previously unseen out-of-order cycles are rejected.
    """
    cfg = _config(config)
    directory = Path(state_dir)
    with _exclusive(directory):
        state = _load(directory)
        if not isinstance(event, dict):
            return _reject(directory, state, {"received": str(event)}, "EVENT_NOT_OBJECT")
        event = dict(event)  # Never mutate the caller's event.
        if "predicted_rul" not in event and "rul" in event:
            event["predicted_rul"] = event["rul"]
        if "event_id" not in event:
            event["event_id"] = f"{event.get('subset')}:{event.get('engine_id')}:{event.get('cycle')}"
        if "mad_score" not in event and "anomaly_score" in event:
            event["mad_score"] = event["anomaly_score"]
        if "mad_threshold" not in event and "anomaly_threshold" in event:
            event["mad_threshold"] = event["anomaly_threshold"]
        subset = event.get("subset")
        engine_value, cycle_value = _number(event.get("engine_id")), _number(event.get("cycle"))
        if subset not in {"FD001", "FD002", "FD003", "FD004"}:
            return _reject(directory, state, event, "MISSING_OR_INVALID_SUBSET")
        if engine_value is None or engine_value < 1 or not engine_value.is_integer():
            return _reject(directory, state, event, "INVALID_ENGINE_ID")
        if cycle_value is None or cycle_value < 1 or not cycle_value.is_integer():
            return _reject(directory, state, event, "INVALID_CYCLE")
        engine_id, cycle = int(engine_value), int(cycle_value)
        key = f"{subset}:{engine_id}"
        current = state["engines"].setdefault(key, {"subset": subset, "engine_id": engine_id,
            "last_cycle": 0, "last_valid": False, "severity": "NORMAL", "candidate": "NORMAL",
            "candidate_count": 0, "clear_count": 0, "episode": 0, "open_order_id": None,
            "history": {}, "policy": asdict(cfg)})
        if current["policy"] != asdict(cfg):
            return _reject(directory, state, event, "POLICY_CHANGE_REQUIRES_NEW_STATE_DIRECTORY")
        safe_event = _json_safe(event)
        fingerprint = hashlib.sha256(_canonical(safe_event).encode("utf-8")).hexdigest()
        historic = current["history"].get(str(cycle))
        if historic:
            if historic["fingerprint"] != fingerprint:
                return _reject(directory, state, event, "CONFLICTING_EVENT_AT_SAME_CYCLE")
            existing = state["work_orders"].get(current["open_order_id"])
            # A previous crash/lock may have committed authoritative state before
            # a projection. Restore projections without changing the event/counts.
            _persist(directory, state)
            return _result("replayed", historic["severity"], "REPLAY_NO_OP", ["IDENTICAL_EVENT"],
                           state, existing, historic["human_review"], historic["confirmed"])
        if cycle <= current["last_cycle"]:
            return _reject(directory, state, event, "OUT_OF_ORDER_CYCLE")
        consecutive = current["last_valid"] and cycle == current["last_cycle"] + 1
        rul = _number(event.get("predicted_rul"))
        probability = _number(event.get("warning_probability"))
        threshold = _number(event.get("warning_threshold"))
        horizon = _number(event.get("warning_horizon"))
        mad_score = _number(event.get("mad_score"))
        mad_threshold = _number(event.get("mad_threshold"))
        errors = []
        if event.get("valid") is not True:
            errors.append("MODEL_OUTPUT_INVALID")
        if rul is None or rul < 0:
            errors.append("MISSING_OR_INVALID_PREDICTED_RUL")
        if "rul" in event and _number(event["rul"]) != rul:
            errors.append("CONFLICTING_RUL_ALIASES")
        for name, value in (("WARNING_PROBABILITY", probability), ("WARNING_THRESHOLD", threshold)):
            if value is None or not 0 <= value <= 1:
                errors.append(f"MISSING_OR_INVALID_{name}")
        if horizon is None or horizon not in {20, 30, 50}:
            errors.append("MISSING_OR_INVALID_WARNING_HORIZON")
        if not isinstance(event.get("event_id"), str) or not event["event_id"].strip():
            errors.append("MISSING_EVENT_ID")
        for name, value in (("MAD_SCORE", mad_score), ("MAD_THRESHOLD", mad_threshold)):
            # Omitted novelty metadata is allowed, but supplied invalid data is
            # routed visibly rather than silently changing the evidence.
            if event.get(name.lower()) is not None and (value is None or value < 0):
                errors.append(f"INVALID_{name}")
        if not isinstance(event.get("model_version"), (str, int)) or str(event.get("model_version", "")).strip() == "":
            errors.append("MISSING_MODEL_VERSION")
        if not isinstance(event.get("input_hash"), str) or not event["input_hash"].strip():
            errors.append("MISSING_INPUT_HASH")
        state["counts"]["observations"] += 1
        current["last_cycle"] = cycle
        if errors:
            current.update(last_valid=False, candidate="NORMAL", candidate_count=0, clear_count=0)
            current["history"][str(cycle)] = {"fingerprint": fingerprint, "severity": "DATA_QUALITY",
                                             "human_review": True, "confirmed": False}
            state["counts"]["data_quality"] += 1
            state["exceptions"].append({"subset": subset, "engine_id": engine_id, "cycle": cycle,
                                        "reason_codes": errors, "event": safe_event, "human_review": True})
            _persist(directory, state)
            return _result("data_quality", "DATA_QUALITY", "HUMAN_REVIEW", errors, state,
                           state["work_orders"].get(current["open_order_id"]), True, False)

        early_warning = probability >= threshold
        novelty = mad_score is not None and mad_threshold is not None and mad_score > mad_threshold
        target = "CRITICAL" if rul <= cfg.critical_rul else "WARNING" if rul <= cfg.warning_rul else "WATCH" if early_warning else "NORMAL"
        emergency = rul <= cfg.emergency_rul
        reasons = []
        if target == "CRITICAL":
            reasons.append("RUL_AT_OR_BELOW_CRITICAL")
        elif target == "WARNING":
            reasons.append("RUL_AT_OR_BELOW_WARNING")
        if early_warning:
            reasons.append("SUPERVISED_EARLY_WARNING_AT_OR_ABOVE_OOF_THRESHOLD")
        if novelty:
            reasons.append("MAD_NOVELTY_SECONDARY_ONLY")
        if emergency:
            reasons.append("EMERGENCY_RUL_IMMEDIATE")
        if not consecutive:
            current["candidate_count"] = current["clear_count"] = 0
        if target == current["candidate"] and consecutive:
            current["candidate_count"] += 1
        else:
            current["candidate"], current["candidate_count"] = target, 1
        active = current["severity"]
        if SEVERITY_RANK[target] > SEVERITY_RANK[active]:
            current["clear_count"] = 0
            if emergency or current["candidate_count"] >= cfg.confirmation_cycles:
                current["severity"] = target
                reasons.append("CONFIRMED_ACTIONABLE" if not emergency else "IMMEDIATE_ACTIONABLE")
            else:
                reasons.append("AWAITING_CONSECUTIVE_CONFIRMATION")
        elif SEVERITY_RANK[target] < SEVERITY_RANK[active]:
            boundary = cfg.critical_rul if active == "CRITICAL" else cfg.warning_rul
            clear = rul > boundary + cfg.hysteresis_cycles and not early_warning
            current["clear_count"] = current["clear_count"] + 1 if clear else 0
            if current["clear_count"] >= cfg.clearance_cycles:
                current["severity"] = target
                current["clear_count"] = 0
                reasons.append("DISPLAY_SEVERITY_CLEARED_AFTER_HYSTERESIS")
            else:
                reasons.append("HELD_BY_HYSTERESIS")
        else:
            current["clear_count"] = 0
        current["last_valid"] = True
        severity = current["severity"]
        confirmed = severity != "NORMAL" or target == "NORMAL"
        action = "NO_ACTION"
        order = state["work_orders"].get(current["open_order_id"])
        evidence = {"event_id": event["event_id"], "predicted_rul": rul,
                    "warning_probability": probability, "warning_threshold": threshold,
                    "warning_horizon": int(horizon),
                    "warning_label_definition": f"true_RUL_at_or_below_{int(horizon)}_cycles",
                    "threshold_source": "TRAIN_ONLY_OOF_MAX_F2",
                    "mad_score": mad_score, "mad_threshold": mad_threshold,
                    "mad_role": "SECONDARY_NOVELTY_ONLY",
                    "input_hash": event["input_hash"], "top_features": safe_event.get("top_features", []),
                    "run_id": safe_event.get("run_id"), "model_version": str(event["model_version"])}
        if severity != "NORMAL" and order is None:
            current["episode"] += 1
            identity = f"{cfg.namespace}|{subset}|{engine_id}|{current['episode']}"
            order_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            order = {"schema_version": 3, "work_order_id": order_id, "namespace": cfg.namespace,
                     "asset": {"subset": subset, "engine_id": engine_id}, "episode": current["episode"],
                     "status": "OPEN", "severity": severity,
                     "action": "INSPECTION" if severity == "WATCH" else "MAINTENANCE",
                     "created_cycle": cycle, "last_observed_cycle": cycle,
                     "model_version": str(event["model_version"]), "evidence": evidence,
                     "policy": asdict(cfg), "policy_scope": "DEMONSTRATION_ONLY_NOT_PRODUCTION",
                     "history": [{"cycle": cycle, "severity": severity, "reason_codes": reasons,
                                  "model_version": str(event["model_version"])}]}
            current["open_order_id"] = order_id
            state["work_orders"][order_id] = order
            state["counts"]["orders_created"] += 1
            action = "CREATE_WORK_ORDER"
        elif order is not None:
            order["last_observed_cycle"] = cycle
            order["latest_evidence"] = evidence
            if SEVERITY_RANK[severity] > SEVERITY_RANK[order["severity"]]:
                order.update(severity=severity, action="MAINTENANCE")
                order["history"].append({"cycle": cycle, "severity": severity,
                                         "reason_codes": reasons, "model_version": str(event["model_version"])})
                state["counts"]["orders_updated"] += 1
                action = "ESCALATE_WORK_ORDER"
            # A recovered model score cannot establish that maintenance occurred.
            if severity == "NORMAL":
                reasons.append("EXISTING_ORDER_REQUIRES_MANUAL_CLOSURE")
        current["history"][str(cycle)] = {"fingerprint": fingerprint, "severity": severity,
                                         "human_review": False, "confirmed": confirmed}
        _persist(directory, state)
        return _result("processed", severity, action, reasons or ["NO_THRESHOLD_TRIGGER"], state,
                       order, False, confirmed,
                       ["Demonstration policy. Early warning predicts a RUL-horizon label, not measured fault onset; MAD is secondary novelty only."])


def close_work_order(state_dir: str | Path, work_order_id: str, reason: str) -> dict:
    """Explicit manual closure; predictions never close a work order."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Manual closure requires a reason")
    directory = Path(state_dir)
    with _exclusive(directory):
        state = _load(directory)
        order = state["work_orders"].get(work_order_id)
        if order is None:
            raise KeyError(work_order_id)
        if order["status"] == "CLOSED":
            return order
        order.update(status="CLOSED", closure_reason=reason, closure_mode="MANUAL")
        key = f"{order['asset']['subset']}:{order['asset']['engine_id']}"
        current = state["engines"][key]
        current.update(open_order_id=None, severity="NORMAL", candidate="NORMAL",
                       candidate_count=0, clear_count=0, last_valid=False)
        state["counts"]["orders_closed"] += 1
        _persist(directory, state)
        return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="JSON list or JSONL observations")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    text = args.events.read_text(encoding="utf-8")
    events = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    if config and "triage" in config:
        config = config["triage"]
    summary = {"processed": 0, "replayed": 0, "rejected": 0, "data_quality": 0}
    sample = None
    for event in events:
        result = process_observation(event, args.state_dir, config)
        summary[result["status"]] += 1
        if result["work_order"]:
            sample = result["work_order"]
    print(json.dumps({"event_counts": summary, "sample_work_order": sample}, indent=2))


if __name__ == "__main__":
    main()
