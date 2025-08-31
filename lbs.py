# %%
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set
import requests
from dotenv import load_dotenv

# --- paths/config ---
DEFAULT_TIMEFRAMES: List[str] = ["7D", "30D", "3M", "6M", "12M"]
PROJECTS_FILE = "projects.json"
EXCEPTIONS_FILE = "exceptions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "leaderboards"


# ---------- utils ----------
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_exclusions(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        return {"exclude_projects": set(), "exclude_timeframes": set()}
    raw = load_json(path)
    return {
        "exclude_projects": set(raw.get("exclude_projects", [])),
        "exclude_timeframes": set(raw.get("exclude_timeframes", [])),
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def filtered_links(
    projects_payload: Any, excluded: Set[str]
) -> List[tuple[str, int]]:
    if isinstance(
        projects_payload, dict
    ) and isinstance(
        projects_payload.get("data"), list
    ):
        items = projects_payload["data"]
    elif isinstance(projects_payload, list):
        items = projects_payload
    else:
        raise ValueError("Unexpected projects.json shape")

    links = []
    for it in items:
        if not isinstance(it, dict):
            continue
        link = it.get("link")
        pid = it.get("id")
        if not link:
            continue
        if link in excluded:
            continue
        if not link or pid is None:
            continue
        links.append((link, int(pid)))
    return links


def http_get_json(url: str, timeout: int = 20) -> Any:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(0.8 * (attempt + 1))
    return None


def save_payload(project_link: str, timeframe: str, payload: Any) -> Path:
    ensure_output_dir()
    out_path = OUTPUT_DIR / f"{project_link}-{timeframe}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


# ---------- main ----------
def main():
    load_dotenv()

    base = os.getenv("ENDPOINT_KAITO_LEADERBOARD")
    if not base:
        raise SystemExit("ENDPOINT_KAITO_LEADERBOARD not set in environment.")

    projects_path = Path(__file__).resolve().parent / PROJECTS_FILE
    if not projects_path.exists():
        raise SystemExit(
            f"{PROJECTS_FILE} not found. Run the projects fetch first.")

    exceptions_path = Path(__file__).resolve().parent / EXCEPTIONS_FILE
    exc = load_exclusions(exceptions_path)

    payload = load_json(projects_path)
    links = filtered_links(payload, exc["exclude_projects"])

    timeframes = [
        tf for tf in DEFAULT_TIMEFRAMES if tf not in exc["exclude_timeframes"]]

    print(f"Projects to query: {len(links)} | Timeframes: {timeframes}")

    # Precompute label width for aligned paths
    labels = [f"✅ 0000. Saved {link} (#{pid}): " for link, pid in links]
    max_label_len = max((len(s) for s in labels), default=0)

    counter = 1
    for link, pid in links:
        for tf in timeframes:
            url = f"{base.rstrip('/')}/{link}?duration={tf}"
            try:
                data = http_get_json(url)
                out_path = save_payload(link, tf, data)

                # Formatted counter and aligned path column
                label = f"✅ {counter:04d}. Saved {link} (#{pid}): "
                pad = " " * max(0, max_label_len - len(label))
                print(f"{label}{pad}{out_path}")

                counter += 1
            except Exception as e:
                print(f"Error for {link} {tf}: {e}")


if __name__ == "__main__":
    main()
