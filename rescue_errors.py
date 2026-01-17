import json
import argparse
from pathlib import Path


def rescue_errors(error_file: str, output_file: str, min_turns: int = 1):
    """
    Reads entries from error_file, checks if they have enough turns to be useful,
    and appends them to output_file.
    """
    error_path = Path(error_file)
    output_path = Path(output_file)

    if not error_path.exists():
        print(f"Error file {error_file} does not exist.")
        return

    print(f"Reading from {error_path}...")

    rescued_count = 0

    with open(error_path, "r", encoding="utf-8") as fin, open(
        output_path, "a", encoding="utf-8"
    ) as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)

                # Check metadata
                metadata = entry.get("metadata", {})
                turns = metadata.get("turns", 0)

                # Basic validation: must have messages
                if not entry.get("messages"):
                    continue

                # If it has enough turns, we keep it
                if turns >= min_turns:
                    # Optional: Remove the error flag or mark it as rescued
                    # metadata['rescued_from_error'] = True
                    # entry['metadata'] = metadata

                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    rescued_count += 1

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num}")

    print(f"Rescued {rescued_count} entries from {error_file} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rescue valid partial sessions from error dataset."
    )
    parser.add_argument("error_file", help="Path to the error dataset (jsonl)")
    parser.add_argument("output_file", help="Path to the main dataset (jsonl)")
    parser.add_argument(
        "--min_turns", type=int, default=5, help="Minimum turns to consider useful"
    )

    args = parser.parse_args()

    rescue_errors(args.error_file, args.output_file, args.min_turns)
