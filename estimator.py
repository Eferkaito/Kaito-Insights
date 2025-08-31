# %%
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- paths/config ---
ROOT = Path(__file__).resolve().parent
LEADERBOARDS_DIR = ROOT / "leaderboards"
LBYAPS_DIR = ROOT / "lbyaps"
YAPS_DB = ROOT / "yaps.json"
STATS_PATH = ROOT / "statistics.json"
HIST_PATH = ROOT / "historical_lbyaps.json"

TIMEFRAMES = ["7D", "30D", "3M", "6M", "12M"]


# ---------- utils ----------
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_items(payload: Any) -> Optional[List[dict]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return None


def coerce_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def at_username(u: Optional[str]) -> Optional[str]:
    if not isinstance(u, str) or not u:
        return None
    return u if u.startswith("@") else f"@{u}"


def strip_at(u: Optional[str]) -> Optional[str]:
    if not isinstance(u, str) or not u:
        return None
    return u[1:] if u.startswith("@") else u


def compute_stats_with_users(pairs: List[Tuple[Optional[str], float]]) -> Dict[str, Optional[float]]:
    """
    pairs: (username, value) where username may be None.
    Returns min/max/avg and min-user/max-user (with @ prefix).
    """
    if not pairs:
        return {"min": None, "max": None, "avg": None, "min-user": None, "max-user": None}
    vals = [v for _, v in pairs]
    avg = sum(vals) / len(vals)
    min_u, min_v = min(pairs, key=lambda t: t[1])
    max_u, max_v = max(pairs, key=lambda t: t[1])
    return {
        "min": min_v,
        "max": max_v,
        "avg": avg,
        "min-user": at_username(min_u),
        "max-user": at_username(max_u),
    }


# ---------- load yaps DB (username -> yaps_all) ----------
def load_yaps_map(path: Path) -> Dict[str, Optional[float]]:
    """
    Expects: {"data":[{"user":"...", "yaps": <float|null>, ...}, ...]}
    """
    if not path.exists():
        raise SystemExit(f"yaps DB not found: {path}")
    raw = read_json(path)
    rows = raw.get("data", []) if isinstance(raw, dict) else []
    out: Dict[str, Optional[float]] = {}
    for row in rows:
        if isinstance(row, dict):
            u = row.get("user")
            y = row.get("yaps")
            if isinstance(u, str) and u:
                out[u] = coerce_float(y)
    return out


# ---------- scan leaderboards ----------
def iter_files_ordered() -> List[Tuple[str, str, Path]]:
    """
    Returns (project, timeframe, path) ordered by timeframe then project name.
    Expects filenames: {project}-{timeframe}.json in leaderboards/.
    """
    if not LEADERBOARDS_DIR.exists():
        raise SystemExit(
            f"leaderboards directory not found: {LEADERBOARDS_DIR}")

    by_project: Dict[str, Dict[str, Path]] = {}
    for p in LEADERBOARDS_DIR.glob("*.json"):
        name = p.name
        for tf in TIMEFRAMES:
            suf = f"-{tf}.json"
            if name.endswith(suf):
                proj = name[: -len(suf)]
                by_project.setdefault(proj, {})[tf] = p
                break

    ordered: List[Tuple[str, str, Path]] = []
    for tf in TIMEFRAMES:
        for proj in sorted(by_project.keys()):
            if tf in by_project[proj]:
                ordered.append((proj, tf, by_project[proj][tf]))
    return ordered


# ---------- injector + stats for one file ----------
def inject_and_summarize(
    project: str, timeframe: str, src_path: Path, yaps_map: Dict[str, Optional[float]]
) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[str], Optional[float], Optional[str]]:
    """
    Injects yaps_all into items and writes:
      lbyaps/{project}/{project}-{timeframe}.json
    Returns (stats_record, smart_min_val, smart_min_user, yaps_min_val, yaps_min_user)
    """
    try:
        payload = read_json(src_path)
    except Exception:
        return None, None, None, None, None

    items = get_items(payload)
    if items is None:
        return None, None, None, None, None

    # Inject yaps_all by username
    for it in items:
        if not isinstance(it, dict):
            continue
        u = it.get("username")
        if isinstance(u, str) and u in yaps_map:
            it["yaps_all"] = yaps_map[u]

    # Persist annotated payload
    out_dir = LBYAPS_DIR / project
    out_path = out_dir / f"{project}-{timeframe}.json"
    write_json(out_path, payload)

    # Build (username, value) pairs
    followers_pairs: List[Tuple[Optional[str], float]] = []
    smart_pairs: List[Tuple[Optional[str], float]] = []
    yaps_pairs: List[Tuple[Optional[str], float]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        u = it.get("username")

        f = coerce_float(it.get("follower_count"))
        if f is not None:
            followers_pairs.append((u, f))

        s = coerce_float(it.get("smart_follower_count"))
        if s is not None:
            smart_pairs.append((u, s))

        y = coerce_float(it.get("yaps_all"))
        if y is not None:
            yaps_pairs.append((u, y))

    followers_stats = compute_stats_with_users(followers_pairs)
    smart_stats = compute_stats_with_users(smart_pairs)
    yaps_stats = compute_stats_with_users(yaps_pairs)

    stats_record = {
        "project": project,
        "timeframe": timeframe,
        "followers": followers_stats,
        "smart_followers": smart_stats,
        "yaps": yaps_stats,
        "count": len(items),
        "path": str(out_path.relative_to(ROOT)),
    }

    # For historical: store username without '@'
    smart_min_user_plain = strip_at(smart_stats["min-user"])
    yaps_min_user_plain = strip_at(yaps_stats["min-user"])

    return (
        stats_record,
        smart_stats["min"],
        smart_min_user_plain,
        yaps_stats["min"],
        yaps_min_user_plain,
    )


# ---------- historical (per project + timeframe) ----------
def load_historical(path: Path) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Shape:
    {
      "<project>": {
        "7D":   {"smart_followers_min":[...], "smart_followers_min_users":[...],
                 "yaps_min":[...], "yaps_min_users":[...]},
      }, ...
    }
    Backward compatible with older shapes (without *_users).
    """
    if not path.exists():
        return {}
    raw = read_json(path)
    hist: Dict[str, Dict[str, Dict[str, List]]] = {}

    if isinstance(raw, dict):
        for proj, v in raw.items():
            if not isinstance(v, dict):
                continue
            hist[proj] = {}
            # Per-timeframe blocks
            if any(k in TIMEFRAMES for k in v.keys()):
                for tf, d in v.items():
                    if tf not in TIMEFRAMES or not isinstance(d, dict):
                        continue
                    sf = d.get("smart_followers_min", [])
                    sf_u = d.get("smart_followers_min_users", [])
                    yp = d.get("yaps_min", [])
                    yp_u = d.get("yaps_min_users", [])
                    hist[proj][tf] = {
                        "smart_followers_min": [x for x in sf if isinstance(x, (int, float))],
                        "smart_followers_min_users": [str(x) for x in sf_u if isinstance(x, (str, int, float))],
                        "yaps_min": [x for x in yp if isinstance(x, (int, float))],
                        "yaps_min_users": [str(x) for x in yp_u if isinstance(x, (str, int, float))],
                    }
            else:
                # Legacy: flat project-level arrays → migrate under 7D
                sf = v.get("smart_followers_min", [])
                yp = v.get("yaps_min", [])
                hist[proj]["7D"] = {
                    "smart_followers_min": [x for x in sf if isinstance(x, (int, float))],
                    "smart_followers_min_users": [],
                    "yaps_min": [x for x in yp if isinstance(x, (int, float))],
                    "yaps_min_users": [],
                }
    return hist


def append_hist_entry(
    hist: Dict[str, Dict[str, Dict[str, List]]],
    project: str,
    timeframe: str,
    smart_min: Optional[float],
    smart_min_user: Optional[str],
    yaps_min: Optional[float],
    yaps_min_user: Optional[str],
    keep_last: int = 10,
) -> None:
    if smart_min is None and yaps_min is None:
        return
    proj_slot = hist.setdefault(project, {})
    tf_slot = proj_slot.setdefault(
        timeframe,
        {"smart_followers_min": [], "smart_followers_min_users": [],
            "yaps_min": [], "yaps_min_users": []},
    )

    # Append values and usernames (plain usernames, allow None)
    if smart_min is not None:
        tf_slot["smart_followers_min"].append(float(smart_min))
        tf_slot["smart_followers_min_users"].append(
            smart_min_user if smart_min_user is not None else None)
    if yaps_min is not None:
        tf_slot["yaps_min"].append(float(yaps_min))
        tf_slot["yaps_min_users"].append(
            yaps_min_user if yaps_min_user is not None else None)

    # Trim to last N while keeping arrays aligned
    for key_val, key_user in [
        ("smart_followers_min", "smart_followers_min_users"),
        ("yaps_min", "yaps_min_users"),
    ]:
        vals = tf_slot[key_val]
        users = tf_slot[key_user]
        if len(vals) > keep_last or len(users) > keep_last:
            vals[:] = vals[-keep_last:]
            users[:] = users[-keep_last:]
        # Hard alignment guard (in case of legacy inconsistencies)
        if len(vals) != len(users):
            m = min(len(vals), len(users))
            vals[:] = vals[-m:]
            users[:] = users[-m:]


# ---------- main ----------
def main():
    yaps_map = load_yaps_map(YAPS_DB)

    files = iter_files_ordered()
    stats_rows: List[Dict[str, Any]] = []
    hist = load_historical(HIST_PATH)

    for project, timeframe, src in files:
        rec, smart_min, smart_min_user, yaps_min, yaps_min_user = inject_and_summarize(
            project, timeframe, src, yaps_map
        )
        if rec:
            stats_rows.append(rec)
        append_hist_entry(
            hist, project, timeframe, smart_min, smart_min_user, yaps_min, yaps_min_user, keep_last=10
        )

    # statistics.json (sorted by project then timeframe)
    tf_rank = {tf: i for i, tf in enumerate(TIMEFRAMES)}
    stats_rows.sort(key=lambda r: (
        r["project"], tf_rank.get(r["timeframe"], 999)))
    write_json(STATS_PATH, {"data": stats_rows})

    # historical_lbyaps.json
    write_json(HIST_PATH, hist)

    print(
        f"Injected {len(stats_rows)} files into lbyaps/, updated statistics (min/max with users) and historical (values+users)."
    )


if __name__ == "__main__":
    main()
