#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path


def records_from_payload(payload):
    if isinstance(payload, dict):
        out = {}
        key_map = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
        }
        for key, canonical in key_map.items():
            if key in payload and isinstance(payload[key], list):
                out.setdefault(canonical, []).extend(payload[key])
        if out:
            return out

        for key in ("annotations", "records", "data", "images"):
            if key in payload and isinstance(payload[key], list):
                return records_from_payload(payload[key])

    if isinstance(payload, list):
        out = {"train": [], "val": [], "test": [], "unknown": []}
        for item in payload:
            split = str(item.get("split", item.get("subset", "unknown"))).lower()
            if split in ("valid", "validation"):
                split = "val"
            if split not in out:
                split = "unknown"
            out[split].append(item)
        return {k: v for k, v in out.items() if v}

    raise ValueError("Unsupported annotation format.")


def text_field(item, names):
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", required=True)
    args = parser.parse_args()

    path = Path(args.ann_path)
    payload = json.loads(path.read_text())
    splits = records_from_payload(payload)

    print("Annotation file:", path)
    print("Split counts:")

    total = 0
    for split in ("train", "val", "test", "unknown"):
        if split in splits:
            n = len(splits[split])
            total += n
            print("  {}: {}".format(split, n))
    print("  total:", total)

    section_names = {
        "findings": ("findings", "finding"),
        "impression": ("impression", "impressions"),
        "report": ("report", "caption", "text"),
    }

    print()
    print("Section availability:")
    for split, records in splits.items():
        print("  [{}]".format(split))
        for section, names in section_names.items():
            present = sum(1 for item in records if isinstance(item, dict) and text_field(item, names))
            missing = len(records) - present
            print("    {}: present={}, missing={}".format(section, present, missing))

    print()
    print("Common keys by split:")
    for split, records in splits.items():
        counter = Counter()
        for item in records[:100]:
            if isinstance(item, dict):
                counter.update(item.keys())
        keys = ", ".join(k for k, _ in counter.most_common(20))
        print("  {}: {}".format(split, keys))


if __name__ == "__main__":
    main()
