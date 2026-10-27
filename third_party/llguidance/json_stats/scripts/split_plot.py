#!/usr/bin/env python3

import os
import glob
import json

splits = [
    "Github_trivial",
    "Github_easy",
    "Github_hard",
    "Github_medium",
    "Github_ultra",
    "Glaiveai2K",
    "Kubernetes",
    "Snowplow",
    "WashingtonPost",
    "MCPspec",
    "JsonSchemaStore",
    # "Handwritten",
    # "Synthesized",
    "TOTAL",
]


def markdown_table(data):
    data = [[str(e) for e in l] for l in data]
    col_widths = [max(len(row[col]) for row in data) for col in range(len(data[0]))]

    # Generate the Markdown table
    def format_row(row):
        return (
            "| "
            + " | ".join(
                f"{cell:<{col_widths[i]}}" if i == 0 else f"{cell:>{col_widths[i]}}"
                for i, cell in enumerate(row)
            )
            + " |\n"
        )

    r = format_row(data[0])
    r += (
        "|"
        + "|".join(
            ":" + "-" * (width + 1) if i == 0 else "-" * (width + 1) + ":"
            for i, width in enumerate(col_widths)
        )
        + "|\n"
    )

    # Print the rows
    for row in data[1:]:
        r += format_row(row)

    return r


def perc(a, b):
    return f"{a/b:.0%}" if b > 0 else "N/A"


def json_markdown_table(data: list[dict]):
    hd = list(data[0].keys())
    rows = [hd]
    for d in data:
        row = [d[k] for k in hd]
        rows.append(row)
    return markdown_table(rows)


def main():
    stats_by_id: dict[str, dict] = {}
    for f in glob.glob("tmp/splits/*.json"):
        id = os.path.basename(f).split(".")[0]
        with open(f, "r") as file:
            data = json.load(file)
            stats_by_id[id] = data
    # splits = [s.split("-")[0] for s in stats_by_id.keys()]
    # splits = list(set(splits))
    # splits.sort()
    rows = []
    for s in splits:
        valid = stats_by_id[f"{s}-valid"]
        invalid = stats_by_id[f"{s}-invalid"]
        compact = stats_by_id[f"{s}-valid-compact"]
        row = {
            "split": s,
            "schemas": valid["num_files"],
            "has tests": perc(
                valid["num_files"] - valid["num_testless_files"], valid["num_files"]
            ),
            "valid inst.": valid["num_valid_tests"],
            "invalid inst.": invalid["num_invalid_tests"],
            "tok/inst.": (
                valid["llg_json"]["num_all_tokens"] // valid["num_valid_tests"]
            ),
            "FF": perc(valid["llg"]["ff_fraction"], 1.0),
            "FF compact": perc(compact["llg"]["ff_fraction"], 1.0),
        }
        llg_row = {
            "TTFM us": valid["llg"]["ttfm_us"],
            "TBT us": valid["llg"]["mask_us"],
            "pass": perc(
                valid["llg"]["num_correct_schemas"],
                valid["num_files"],
            ),
        }
        rows.append(
            {
                **row,
                # **llg_row,
            }
        )
    print(json_markdown_table(rows))


if __name__ == "__main__":
    main()
