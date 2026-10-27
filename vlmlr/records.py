"""Reading and writing per-item prediction files (JSON lines, one record per item and condition)."""
import collections
import json
import os
import sys


def read_jsonl(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path} (per-item predictions come from running the experiment, data files from DATA.md; see README)")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_item(rows, item="idx", cond="variant"):
    """{item id: {condition: record}}; the last record wins on duplicates."""
    out = collections.defaultdict(dict)
    for r in rows:
        out[r[item]][r[cond]] = r
    return out


def load_by_item(path, item="idx", cond="variant"):
    return by_item(read_jsonl(path), item, cond)


class ResumableWriter:
    """Append records to a JSONL file; keys already present (from an interrupted run) are skipped."""

    def __init__(self, path, key=("idx", "variant")):
        self.path, self.key = path, key
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            self.done = {tuple(r[k] for k in key) for r in read_jsonl(path)}
        self.f = open(path, "a")
        self.n = 0

    def __contains__(self, key):
        return tuple(key) in self.done

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
