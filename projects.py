# %%
import os
import json
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv


# ---------- utils ----------
def fetch_projects(endpoint: str) -> Any:
    resp = requests.get(endpoint, timeout=20)
    resp.raise_for_status()
    return resp.json()


def sort_by_id(payload: Any) -> Any:
    if isinstance(
        payload, dict
    ) and "data" in payload and isinstance(
        payload["data"], list
    ):
        try:
            payload["data"] = sorted(
                payload["data"], key=lambda x: int(x.get("id", 0))
            )
        except Exception:
            pass
    return payload


def save_to_file(data: Any, filename: str = "projects.json") -> None:
    path = Path(__file__).resolve().parent / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {path}")


# ---------- main ----------
if __name__ == "__main__":
    load_dotenv()
    endpoint = os.getenv("ENDPOINT_KAITO_LEADERBOARD")
    if not endpoint:
        raise SystemExit(
            "ENDPOINT_KAITO_LEADERBOARD not found in environment variables.")

    data = fetch_projects(endpoint)
    data = sort_by_id(data)
    save_to_file(data)
