"""Download GH Archive hours in UTC; the end time is exclusive."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="UTC YYYY-MM-DDTHH")
    p.add_argument("--end", required=True, help="UTC YYYY-MM-DDTHH, exclusive")
    p.add_argument("--out", default="raw")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--manifest", default="results/raw_manifest.json")
    a = p.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%dT%H")
    end = datetime.strptime(a.end, "%Y-%m-%dT%H")
    if end <= start:
        raise ValueError("end must be later than start")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hours = [
        start + timedelta(hours=i)
        for i in range(int((end - start).total_seconds() // 3600))
    ]

    def fetch(h):
        name = h.strftime("%Y-%m-%d") + f"-{h.hour}.json.gz"
        path = out / name
        error = None
        if not path.exists():
            for attempt in range(3):
                try:
                    request = urllib.request.Request(
                        "https://data.gharchive.org/" + name,
                        headers={"User-Agent": "GHArchiveCourseDownloader/1.0"},
                    )
                    with (
                        urllib.request.urlopen(request, timeout=120) as response,
                        open(str(path) + ".part", "wb") as f,
                    ):
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                    Path(str(path) + ".part").replace(path)
                    error = None
                    break
                except Exception as e:
                    error = str(e)
                    time.sleep(attempt + 1)
        if error:
            return {"hour": h.isoformat(), "file": name, "error": error}
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
        return {
            "hour": h.isoformat(),
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records = []
        for i, record in enumerate(pool.map(fetch, hours), 1):
            records.append(record)
            if i % 24 == 0 or i == len(hours):
                print(f"{i}/{len(hours)}", flush=True)
    target = Path(a.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2))
    failed = [r for r in records if "error" in r]
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed; see {target}")
    print(
        "All requested files downloaded; availability does not prove source "
        "completeness."
    )


if __name__ == "__main__":
    main()
