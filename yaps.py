# %%
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote
import requests
from dotenv import load_dotenv

# --- paths/config ---
ROOT = Path(__file__).resolve().parent
LEADERBOARDS_DIR = ROOT / "leaderboards"
DB_PATH = ROOT / "yaps.json"

TIMEFRAMES = ["7D", "30D", "3M", "6M", "12M"]


# ---------- utils ----------
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


# ---------- scan leaderboards ----------
def iter_files_ordered() -> List[Tuple[str, str, Path]]:
    """
    Returns (project, timeframe, path) ordered by timeframe priority then project name.
    Expects filenames like: {project}-{timeframe}.json
    """
    if not LEADERBOARDS_DIR.exists():
        raise SystemExit(
            f"leaderboards directory not found: {LEADERBOARDS_DIR}"
        )

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


def extract_user_records(payload: Any) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Returns list of (username, follower_count, smart_follower_count).
    Accepts:
      - {"data": [ {...} ]}
      - [ {...} ]
    """
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items = payload["data"]
    elif isinstance(payload, list):
        items = payload
    else:
        return []
    out: List[Tuple[str, Optional[float], Optional[float]]] = []
    for it in items:
        if isinstance(it, dict):
            u = it.get("username")
            if isinstance(u, str) and u:
                f = it.get("follower_count")
                s = it.get("smart_follower_count")
                try:
                    f = float(f) if f is not None else None
                except Exception:
                    f = None
                try:
                    s = float(s) if s is not None else None
                except Exception:
                    s = None
                out.append((u, f, s))
    return out


# ---------- robust bulk fetch with concise logging ----------
def fetch_yaps_bulk(
    base_url: str,
    usernames: List[str],
    max_count: int,
    max_url_len: int,
    verbose_split: bool,
) -> Tuple[Dict[str, Optional[float]], int]:
    """
    Returns (username -> yaps_all, chunks_used)
    Splits by count/URL length, and on HTTP 5xx splits recursively.
    """
    out: Dict[str, Optional[float]] = {}
    chunks_used = 0
    if not usernames:
        return out, chunks_used

    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for u in usernames:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    session = requests.Session()

    def parse_rows(rows: Any) -> Dict[str, Optional[float]]:
        res: Dict[str, Optional[float]] = {}
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    u = row.get("username")
                    if isinstance(u, str) and u:
                        val = row.get("yaps_all")
                        try:
                            res[u] = float(val) if val is not None else None
                        except (TypeError, ValueError):
                            res[u] = None
        return res

    def process(group: List[str]) -> None:
        nonlocal chunks_used
        # enforce count + encoded URL length
        qs_raw = ",".join(group)
        qs_encoded = quote(qs_raw, safe=",")
        url_len = len(base_url) + len("?users=") + len(qs_encoded)
        if len(group) > max_count or url_len > max_url_len:
            mid = max(1, len(group) // 2)
            left, right = group[:mid], group[mid:]
            if verbose_split:
                reasons = []
                if len(group) > max_count:
                    reasons.append(f"count>{max_count}")
                if url_len > max_url_len:
                    reasons.append(f"url>{max_url_len}")
                print(
                    f"↔️ split(pre) [{', '.join(reasons)}]: {len(group)} -> {len(left)} + {len(right)}"
                )
            process(left)
            process(right)
            return

        # request with light retry; split on 5xx
        for attempt in range(3):
            try:
                resp = session.get(base_url, params={
                                   "users": qs_raw}, timeout=25)
                chunks_used += 1
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("data", []) if isinstance(data, dict) else []
                parsed = parse_rows(rows)
                for u in group:
                    out[u] = parsed.get(u, None)
                if verbose_split:
                    print(f"✅ chunk ok: {len(group)} users")
                return
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status and status >= 500 and len(group) > 1:
                    mid = max(1, len(group) // 2)
                    left, right = group[:mid], group[mid:]
                    if verbose_split:
                        print(
                            f"🧩 split(5xx {status}): {len(group)} -> {len(left)} + {len(right)}"
                        )
                    process(left)
                    process(right)
                    return
                time.sleep(0.6 * (attempt + 1))
            except requests.RequestException:
                time.sleep(0.6 * (attempt + 1))

        # total failure -> split or give up on single user
        if len(group) == 1:
            u = group[0]
            out[u] = None
            print(f"⚠️ bulk failed for user '{u}' → recorded yaps=None")
        else:
            mid = max(1, len(group) // 2)
            left, right = group[:mid], group[mid:]
            if verbose_split:
                print(
                    f"🧩 split(fallback): {len(group)} -> {len(left)} + {len(right)}"
                )
            process(left)
            process(right)

    process(uniq)
    return out, chunks_used


# ---------- main ----------
def main():
    load_dotenv()

    bulk_url = os.getenv("ENDPOINT_YAP_BULK_ALLTIME")
    if not bulk_url:
        raise SystemExit("ENDPOINT_YAP_BULK_ALLTIME not set in environment.")

    BULK_BATCH_SIZE = int(os.getenv("YAP_BULK_BATCH_SIZE", "100"))
    MAX_URL_LEN = int(os.getenv("YAP_MAX_URL_LEN", "1800"))
    VERBOSE_SPLIT = os.getenv("YAP_VERBOSE_SPLIT", "0") == "1"

    # fresh DB every run
    if DB_PATH.exists():
        DB_PATH.unlink()

    # in-memory DB: user -> {"yaps": float|None, "by_tf": {tf: set(projects)}, "followers": float|None, "smart_followers": float|None}
    db: Dict[str, Dict[str, Any]] = {}

    # staging for new users (not yet fetched)
    staged_order: List[str] = []
    staged_set: Set[str] = set()
    # user -> tf -> {projects}
    staged_projects: Dict[str, Dict[str, Set[str]]] = {}
    # user -> {"followers": float|None, "smart": float|None} (first-seen values only)
    staged_counts: Dict[str, Dict[str, Optional[float]]] = {}

    def ensure_entry(user: str) -> Dict[str, Any]:
        entry = db.setdefault(
            user, {"yaps": None, "by_tf": {}, "followers": None, "smart_followers": None})
        if "by_tf" not in entry or not isinstance(entry["by_tf"], dict):
            entry["by_tf"] = {}
        if "followers" not in entry:
            entry["followers"] = None
        if "smart_followers" not in entry:
            entry["smart_followers"] = None
        return entry

    def persist_db():
        rows: List[Dict[str, Any]] = []
        for user, entry in db.items():
            row: Dict[str, Any] = {"user": user, "yaps": entry.get("yaps")}
            by_tf: Dict[str, Set[str]] = entry.get("by_tf", {})
            for tf in TIMEFRAMES:
                projs = by_tf.get(tf, set())
                if projs:
                    row[tf] = sorted(projs)
            # write single followers/smart_followers if available (from first-seen timeframe)
            if isinstance(entry.get("followers"), (int, float)):
                row["followers"] = float(entry["followers"])
            if isinstance(entry.get("smart_followers"), (int, float)):
                row["smart_followers"] = float(entry["smart_followers"])
            rows.append(row)

        def sort_key(r: Dict[str, Any]):
            y = r.get("yaps")
            return (0, -y, r["user"]) if isinstance(y, (int, float)) else (1, 0, r["user"])

        rows.sort(key=sort_key)
        write_json(DB_PATH, {"data": rows})

    bulk_runs = 0

    def flush_stage():
        nonlocal staged_order, staged_set, staged_projects, staged_counts, db, bulk_runs

        if not staged_set:
            return

        pre_count = len(db)
        batches = list(chunked(staged_order, BULK_BATCH_SIZE))
        total_chunks = 0
        total_users = len(staged_order)

        for batch in batches:
            fetched, chunks_used = fetch_yaps_bulk(
                bulk_url, batch, BULK_BATCH_SIZE, MAX_URL_LEN, VERBOSE_SPLIT
            )
            total_chunks += chunks_used
            for u in batch:
                yaps_val = fetched.get(u, None)
                entry = ensure_entry(u)
                entry["yaps"] = yaps_val
                # attach projects
                for tf, proj_set in staged_projects.get(u, {}).items():
                    entry["by_tf"].setdefault(tf, set()).update(proj_set)
                # attach first-seen counts (only if not already set)
                counts = staged_counts.get(u, {})
                f = counts.get("followers")
                s = counts.get("smart")
                if entry.get("followers") is None and isinstance(f, (int, float)):
                    entry["followers"] = float(f)
                if entry.get("smart_followers") is None and isinstance(s, (int, float)):
                    entry["smart_followers"] = float(s)

        bulk_runs += 1
        added = len(db) - pre_count
        print(
            f"📦 BULK #{bulk_runs}: fetched {total_users} users in {total_chunks} chunk(s) → +{added} new users (DB={len(db)})"
        )

        # persist incrementally after each bulk
        persist_db()

        # clear stage
        staged_order = []
        staged_set.clear()
        staged_projects.clear()
        staged_counts.clear()

    files = iter_files_ordered()
    processed_files = 0

    for project, tf, path in files:
        try:
            payload = read_json(path)
        except Exception:
            continue

        # [(username, followers, smart_followers)]
        records = extract_user_records(payload)
        if not records:
            processed_files += 1
            continue

        for u, f, s in records:
            if u in db:
                entry = ensure_entry(u)
                entry["by_tf"].setdefault(tf, set()).add(project)
                # set first-seen counts if not set yet
                if entry.get("followers") is None and isinstance(f, (int, float)):
                    entry["followers"] = float(f)
                if entry.get("smart_followers") is None and isinstance(s, (int, float)):
                    entry["smart_followers"] = float(s)
            else:
                if u not in staged_set:
                    staged_set.add(u)
                    staged_order.append(u)
                staged_projects.setdefault(
                    u, {}).setdefault(tf, set()).add(project)
                # keep first-seen counts for this new user until flush
                sc = staged_counts.setdefault(u, {})
                if sc.get("followers") is None and isinstance(f, (int, float)):
                    sc["followers"] = float(f)
                if sc.get("smart") is None and isinstance(s, (int, float)):
                    sc["smart"] = float(s)

                if len(staged_set) >= BULK_BATCH_SIZE:
                    flush_stage()

        processed_files += 1

    # final flush
    if staged_set:
        flush_stage()

    # final persist
    persist_db()
    print(
        f"Done. Files processed: {processed_files} | DB users: {len(db)} | DB path: {DB_PATH}"
    )


if __name__ == "__main__":
    main()

# %%
