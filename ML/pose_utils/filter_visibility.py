#!/usr/bin/env python3
"""Filters a landmark CSV file based on a minimum visibility score.

This script reads a CSV file containing pose landmark data, filters out rows
where the landmark's 'visibility' score is below a specified threshold, and
then saves the result.

The script produces two potential outputs:
1.  A new filtered CSV is always created in the same directory as the source,
    with a name like: `<original_name>_filtered_visibility_<threshold>.csv`.
2.  If a `class_id` is provided, it will also overwrite the CSV file
    path specified for that class in the project's configuration.

Can be run as a standalone script from the command line or imported and used
in other modules.

Command-Line Usage:
    python filter_visibility.py /path/to/data.csv 0.90

    python filter_visibility.py /path/to/data.csv 0.90 --class_id 0

Module Usage:
    from filter_visibility import filter_by_visibility

    filter_by_visibility("path/to/data.csv", 0.95, class_id=1)
"""

from pathlib import Path

import pandas as pd

# Project-specific imports for configuration constants.
from config import DATA_BY_CLASS, CONF_THRES, PRE_PROCESS_INPUT,CLASS_ID


def filter_by_visibility(csv_path: str | Path,
                         visibility_threshold: float,
                         class_id: int | None = None) -> None:
    """
    Reads, filters, and saves a landmark CSV based on a visibility threshold.

    Args:
        csv_path: The path to the input landmark CSV file.
        visibility_threshold: The minimum visibility value. Rows with visibility
                              below this value will be removed.
        class_id: If provided, the filtered data will also be saved to the
                  path configured for this class ID in `DATA_BY_CLASS`.

    Raises:
        FileNotFoundError: If the input csv_path does not exist.
        ValueError: If a 'visibility' column cannot be found in the CSV.
        KeyError: If the provided class_id is not a valid key in `DATA_BY_CLASS`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Find the visibility column, case-insensitively.
    vis_cols = [c for c in df.columns if c.lower() == "visibility"]
    if not vis_cols:
        raise ValueError("No 'visibility' column found in the CSV file.")
    vis_col = vis_cols[0]

    # Filter the DataFrame.
    df_filtered = df[df[vis_col] > visibility_threshold]

    # --- Save Output 1: Local Filtered File ---
    # Create a new, descriptive filename for the filtered output.
    new_name = f"{csv_path.stem}_filtered_visibility_{visibility_threshold}{csv_path.suffix}"
    local_out = csv_path.parent / new_name
    df_filtered.to_csv(local_out, index=False)
    print(f"[✓] Saved filtered CSV locally → {local_out}")

    # --- Save Output 2: Class-Specific File (Optional) ---
    if class_id is not None:
        if class_id not in DATA_BY_CLASS:
            raise KeyError(f"class_id {class_id} does not exist in DATA_BY_CLASS.")

        target_path: Path = DATA_BY_CLASS[class_id]["csv"]
        # Ensure the target directory exists before saving.
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(target_path, index=False)
        print(f"[✓] Overwrote filtered CSV for class {class_id} → {target_path}")


def main():
    '''
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description="Filter a landmark CSV file by visibility score."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "visibility_threshold",
        type=float,
        help="Minimum visibility threshold (e.g., 0.90)."
    )
    parser.add_argument(
        "--class_id",
        type=int,
        choices=DATA_BY_CLASS.keys(),
        help="Optional: The class ID to save the output for, as defined in config."
    )
    args = parser.parse_args()

    filter_by_visibility(args.csv_path, args.visibility_threshold, args.class_id)
    '''
    filter_by_visibility(PRE_PROCESS_INPUT, CONF_THRES, CLASS_ID)
if __name__ == "__main__":
    main()