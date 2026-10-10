"""Deterministic safety validator for model-generated flight plans.

The language model only *proposes* a plan. Nothing reaches the simulator unless
this module accepts every step. There is no AI in this file on purpose: the same
plan and drone state always produce the same verdict, whatever the model said
and whatever the operator typed.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------------------
# World definition (metres). Origin is the top-left corner, x grows east,
# y grows south, so "north" means decreasing y.
# ----------------------------------------------------------------------------
WORLD = {
    "size": 100,
    "locations": {
        "home": {"pos": [10, 90], "aliases": ["base", "start", "launch pad", "launchpad"]},
        "farm": {"pos": [20, 30], "aliases": ["field", "the farm"]},
        "tower": {"pos": [82, 18], "aliases": ["radio tower", "the tower"]},
        "lake": {"pos": [68, 78], "aliases": ["the lake", "pond"]},
        "checkpoint": {"pos": [45, 42], "aliases": ["check point", "the checkpoint"]},
        "airport": {
            "pos": [75, 48],
            "aliases": ["restricted zone", "restricted area", "no fly zone", "no fly area"],
        },
    },
    "obstacles": [
        {"id": "warehouse", "type": "rect", "x": 30, "y": 58, "w": 18, "h": 14},
        {"id": "tree line", "type": "circle", "x": 55, "y": 12, "r": 7},
    ],
    "no_fly": [
        {"id": "airport airspace", "type": "rect", "x": 62, "y": 36, "w": 28, "h": 24},
    ],
    # Hidden points of interest that a scan can reveal (client-side only).
    "pois": [[22, 26], [81, 22], [66, 82], [47, 40]],
}

RULES = {
    "max_steps": 12,
    "max_move_m": 50,
    "battery_per_m": 0.4,
    "takeoff_cost": 3,
    "land_cost": 2,
    "scan_cost": 4,
    "battery_reserve": 10,
    "obstacle_margin_m": 1.5,
    "scan_radius_m": 15,
}

ACTIONS = ("takeoff", "land", "move", "goto", "scan", "return_home")

_S = math.sqrt(0.5)
DIRECTIONS = {
    "N": (0, -1), "NE": (_S, -_S), "E": (1, 0), "SE": (_S, _S),
    "S": (0, 1), "SW": (-_S, _S), "W": (-1, 0), "NW": (-_S, -_S),
}
_DIR_WORDS = {
    "north": "N", "northeast": "NE", "east": "E", "southeast": "SE",
    "south": "S", "southwest": "SW", "west": "W", "northwest": "NW",
}
_DIR_NAMES = {v: k for k, v in _DIR_WORDS.items()}


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _seg_hits_rect(p, q, x, y, w, h, margin):
    """Liang-Barsky clip of segment p->q against a rectangle grown by margin."""
    xmin, ymin, xmax, ymax = x - margin, y - margin, x + w + margin, y + h + margin
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - xmin), (dx, xmax - p[0]), (-dy, p[1] - ymin), (dy, ymax - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


def _seg_hits_circle(p, q, cx, cy, r, margin):
    dx, dy = q[0] - p[0], q[1] - p[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((cx - p[0]) * dx + (cy - p[1]) * dy) / length_sq))
    nx, ny = p[0] + t * dx, p[1] + t * dy
    return math.hypot(cx - nx, cy - ny) <= r + margin


def _hits(shape, p, q, margin):
    if shape["type"] == "rect":
        return _seg_hits_rect(p, q, shape["x"], shape["y"], shape["w"], shape["h"], margin)
    return _seg_hits_circle(p, q, shape["x"], shape["y"], shape["r"], margin)


def _hazard(p, q):
    """Return a human-readable reason if the straight path p->q is unsafe."""
    for zone in WORLD["no_fly"]:
        if _hits(zone, p, q, 0):
            return f"Path enters the no-fly zone ({zone['id']})."
    for obstacle in WORLD["obstacles"]:
        if _hits(obstacle, p, q, RULES["obstacle_margin_m"]):
            return f"Path collides with the {obstacle['id']}."
    return None


# ----------------------------------------------------------------------------
# Normalisation helpers
# ----------------------------------------------------------------------------
def _norm_direction(value):
    if not isinstance(value, str):
        return None
    t = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if t.upper() in DIRECTIONS:
        return t.upper()
    return _DIR_WORDS.get(t)


def _clean_name(value):
    t = " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())
    return t[4:] if t.startswith("the ") else t


def resolve_location(value):
    """Map a free-text place name to a key in WORLD['locations'], or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    wanted = _clean_name(value)
    for key, loc in WORLD["locations"].items():
        names = [key] + loc["aliases"]
        if wanted in {_clean_name(n) for n in names}:
            return key
    return None


def _r(v):
    return round(float(v), 2)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _plan_step(step, cur):
    """Work out what one step would do. Never mutates cur.

    Returns dict(label, to, cost, flying, reason). reason is None when the step
    is allowed.
    """
    pos = (cur["x"], cur["y"])
    out = {"label": "Invalid step", "to": pos, "cost": 0.0, "flying": cur["flying"], "reason": None}

    action = step.get("action") if isinstance(step, dict) else None
    action = action.strip().lower() if isinstance(action, str) else None
    out["action"] = action

    if action not in ACTIONS:
        out["label"] = f"Unknown action {action!r}"
        out["reason"] = f"Action {action!r} is not on the allow-list ({', '.join(ACTIONS)})."
        return out

    target = None

    if action == "takeoff":
        out["label"] = "Take off"
        if cur["flying"]:
            out["reason"] = "The drone is already in the air."
        else:
            out["cost"], out["flying"] = RULES["takeoff_cost"], True

    elif action == "land":
        out["label"] = "Land"
        if not cur["flying"]:
            out["reason"] = "The drone is already on the ground."
        else:
            out["cost"], out["flying"] = RULES["land_cost"], False
            spot = _hazard(pos, pos)
            if spot:
                out["reason"] = spot.replace("Path enters", "Landing spot is inside").replace(
                    "Path collides with", "Landing spot is too close to")

    elif action == "scan":
        out["label"] = "Scan the area"
        if not cur["flying"]:
            out["reason"] = "The drone must be airborne to scan."
        else:
            out["cost"] = RULES["scan_cost"]

    elif action == "move":
        direction = _norm_direction(step.get("direction"))
        meters = step.get("meters")
        if direction is None:
            out["label"] = "Move (unknown direction)"
            out["reason"] = f"Unknown direction {step.get('direction')!r}. Use N, NE, E, SE, S, SW, W or NW."
        elif isinstance(meters, bool) or not isinstance(meters, (int, float)) or not math.isfinite(meters):
            out["label"] = f"Move {_DIR_NAMES[direction]}"
            out["reason"] = "Distance must be a number of metres."
        else:
            out["label"] = f"Move {_DIR_NAMES[direction]} {meters:g} m"
            if meters <= 0 or meters > RULES["max_move_m"]:
                out["reason"] = (
                    f"Distance must be between 0 and {RULES['max_move_m']} m per move (got {meters:g} m)."
                )
            else:
                dx, dy = DIRECTIONS[direction]
                target = (_r(pos[0] + dx * meters), _r(pos[1] + dy * meters))

    else:  # goto / return_home
        if action == "return_home":
            key = "home"
        else:
            key = resolve_location(step.get("location"))
        if key is None:
            out["label"] = "Go to unknown place"
            out["reason"] = f"Unknown location {step.get('location')!r}. Known places: {', '.join(WORLD['locations'])}."
        else:
            out["label"] = "Return home" if action == "return_home" else f"Go to {key}"
            target = tuple(WORLD["locations"][key]["pos"])

    if target is not None:
        out["to"] = target
        if not cur["flying"]:
            out["reason"] = "The drone is on the ground. Take off first."
        elif not (0 <= target[0] <= WORLD["size"] and 0 <= target[1] <= WORLD["size"]):
            out["reason"] = (
                f"Target ({target[0]:g}, {target[1]:g}) is outside the {WORLD['size']} m map boundary."
            )
        else:
            out["reason"] = _hazard(pos, target)
        out["cost"] = math.hypot(target[0] - pos[0], target[1] - pos[1]) * RULES["battery_per_m"]

    if out["reason"] is None and cur["battery"] - out["cost"] < RULES["battery_reserve"]:
        out["reason"] = (
            f"Battery would drop to {cur['battery'] - out['cost']:.0f}% "
            f"(the safety reserve is {RULES['battery_reserve']}%)."
        )
    return out


def validate(steps, state):
    """Check a whole plan against the safety rules, starting from state.

    The plan is accepted only if every step is allowed. On the first rejected
    step, later steps are marked "skipped" and nothing should be executed.
    """
    cur = {
        "x": _r(state["x"]), "y": _r(state["y"]),
        "flying": bool(state["flying"]), "battery": float(state["battery"]),
    }
    result = {"ok": False, "plan_error": None, "steps": [], "final": dict(cur), "warnings": []}

    if not isinstance(steps, list) or not steps:
        result["plan_error"] = "The plan has no executable steps."
        return result
    if len(steps) > RULES["max_steps"]:
        result["plan_error"] = f"The plan has {len(steps)} steps; the limit is {RULES['max_steps']}."

    failed = result["plan_error"] is not None
    for i, step in enumerate(steps):
        if failed:
            result["steps"].append({
                "index": i, "action": None, "label": _skipped_label(step), "status": "skipped",
                "reason": None, "from": [cur["x"], cur["y"]], "to": [cur["x"], cur["y"]],
                "battery_after": round(cur["battery"], 1),
            })
            continue

        res = _plan_step(step, cur)
        entry = {
            "index": i, "action": res["action"], "label": res["label"],
            "from": [cur["x"], cur["y"]], "to": [_r(res["to"][0]), _r(res["to"][1])],
        }
        if res["reason"]:
            failed = True
            entry.update(status="rejected", reason=res["reason"], battery_after=round(cur["battery"], 1))
        else:
            cur["x"], cur["y"] = _r(res["to"][0]), _r(res["to"][1])
            cur["flying"] = res["flying"]
            cur["battery"] -= res["cost"]
            entry.update(status="ok", reason=None, battery_after=round(cur["battery"], 1))
        result["steps"].append(entry)

    result["final"] = {**cur, "battery": round(cur["battery"], 1)}
    result["ok"] = not failed
    if result["ok"]:
        if cur["flying"]:
            result["warnings"].append("The plan ends with the drone still in the air.")
        if cur["battery"] < 25:
            result["warnings"].append(f"Low battery after this plan ({cur['battery']:.0f}%).")
    return result


def _skipped_label(step):
    if isinstance(step, dict) and isinstance(step.get("action"), str):
        return step["action"]
    return "Step"
