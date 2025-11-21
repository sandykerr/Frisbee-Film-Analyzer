import pandas as pd
import re
from datetime import datetime
from pathlib import Path


def parse_duration(cell):
    """Extracts start/end times and returns duration in seconds."""
    if not isinstance(cell, str):
        return None

    # Match patterns like "3:30-3:46" or "start-3:08"
    m = re.match(r"(start|[\d:]+)-([\d:]+)", cell)
    if not m:
        return None

    start, end = m.group(1), m.group(2)

    # Handle "start" as 0:00
    if start.lower() == "start":
        t0 = datetime.strptime("0:00", "%M:%S")
    else:
        fmt = "%M:%S" if start.count(":") == 1 else "%H:%M:%S"
        t0 = datetime.strptime(start, fmt)

    fmt_end = "%M:%S" if end.count(":") == 1 else "%H:%M:%S"
    t1 = datetime.strptime(end, fmt_end)

    return (t1 - t0).seconds


def find_long_points(df, o_threshold_secs=120, d_threshold_secs=180):
    all_o_points = []
    long_o_holds = []
    o_breaks = []
    long_o_breaks = []
    all_d_points = []
    long_d_holds = []
    d_breaks = []
    long_d_breaks = []

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            cell = df.iat[r, c]

            if isinstance(cell, str):
                duration = parse_duration(cell)
                is_o_point = re.search(r"\bO Point\b", cell, re.IGNORECASE)
                if duration is not None:
                    # Offense
                    if is_o_point:
                        all_o_points.append((r, c, duration))
                        # O Hold
                        if duration > o_threshold_secs and re.search(
                            r"\(H\)", cell
                        ):
                            long_o_holds.append((r, c, duration))
                        # O Breaks (short or long)
                        elif re.search(r"\(B\)", cell):
                            if duration > o_threshold_secs:
                                long_o_breaks.append((r, c, duration))
                                o_breaks.append((r, c, duration))
                            else:
                                o_breaks.append((r, c, duration))
                    # Defense
                    else:
                        all_d_points.append((r, c, duration))
                        # D Holds
                        if duration > d_threshold_secs and re.search(
                            r"\(H\)", cell
                        ):
                            long_d_holds.append((r, c, duration))
                        # D Breaks (short or long)
                        elif re.search(r"\(B\)", cell):
                            if duration > d_threshold_secs:
                                long_d_breaks.append((r, c, duration))
                                d_breaks.append((r, c, duration))
                            else:
                                d_breaks.append((r, c, duration))

    return {
        "all_o_points": all_o_points,
        "long_o_holds": long_o_holds,
        "o_breaks": o_breaks,
        "long_o_breaks": long_o_breaks,
        "all_d_points": all_d_points,
        "long_d_holds": long_d_holds,
        "d_breaks": d_breaks,
        "long_d_breaks": long_d_breaks,
    }


def format_durations(point_dict, verbosity=0):
    output = ""
    for k, v in point_dict.items():
        if "all_" in k and verbosity == 0:
            continue

        output += f"{k}\n"
        sorted_v = sorted(v, key=lambda x: category_key(df.iat[x[0], x[1]]))
        for row, col, duration in sorted_v:
            mins = duration // 60
            secs = duration % 60
            output += f"({row}, {col}) → {df.iat[row, col]}, {mins}m{secs}s\n"
    return output


def category_key(value):
    if re.search(r"D1", value):
        return 0
    elif re.search(r"D2", value):
        return 1
    elif re.search(r"Aggro", value):
        return 2
    elif re.search(r"Suns", value):
        return 3
    else:
        return 4


def extract_points(cell):
    entries = str(cell).split(",")
    result = []
    for e in entries:
        # Match anything, then capture the last part before "Point"
        pattern = r"(?:.*?\s)?([A-Za-z0-9+]+(?:\s+\S+)*)\s+Point\s+\((H|B)\)"
        match = re.search(pattern, e, re.IGNORECASE)
        if match:
            line, point_type = match.groups()
            point_type = point_type.upper().strip()
            result.append((line.strip(), point_type.upper()))
        # print(f"extract_points result: {result}")
    return result


def get_line_stats(point_dict, df):
    all_points = []
    df2 = df.copy()
    school_order = list(df2.columns)  # preserve CSV school order

    for col in df2.columns:
        df2_parsed = df2[col].apply(extract_points)
        df2_exploded = df2_parsed.explode().dropna()
        df2_exploded = pd.DataFrame(
            df2_exploded.tolist(), columns=["Line", "point_type"]
        )
        df2_exploded["School"] = col
        all_points.append(df2_exploded)

    df2_points = pd.concat(all_points, ignore_index=True)
    print(df2_points.head())
    print(df2_points.dtypes)
    print(df2_points["point_type"].unique())

    # Count points per school and line
    counts = (
        df2_points.groupby(["School", "Line", "point_type"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"counts: {counts}")

    # Calculate percentages
    percentages = counts.div(counts.sum(axis=1), axis=0).round(2) * 100

    # Combine counts and percentages into a single DataFrame
    counts.columns = ["B", "H"]  # raw counts
    percentages.columns = ["B%", "H%"]  # percentages
    combined = pd.concat([counts, percentages], axis=1)

    # Reset index to sort lines safely by category_key
    combined_reset = combined.reset_index()
    combined_reset["line_order"] = combined_reset["Line"].apply(category_key)

    # Preserve original CSV school order
    combined_reset["School"] = pd.Categorical(
        combined_reset["School"], categories=school_order, ordered=True
    )

    # Sort by school order and line category
    combined_reset = combined_reset.sort_values(["School", "line_order"]).drop(
        columns="line_order"
    )

    # Set MultiIndex
    combined_sorted = combined_reset.set_index(["School", "Line"])

    # --- Compute overall ---
    overall_counts = counts.groupby(level="Line").sum()
    overall_schools_present = counts.groupby(level="Line").apply(
        lambda df: (df.sum(axis=1) > 0).sum()
    )
    overall_percentages = overall_counts.div(overall_schools_present, axis=0)
    overall_percentages = (
        overall_percentages.div(overall_percentages.sum(axis=1), axis=0).round(
            2
        )
        * 100
    )

    overall_counts.index = pd.MultiIndex.from_product(
        [["Overall"], overall_counts.index], names=["School", "Line"]
    )
    overall_percentages.index = pd.MultiIndex.from_product(
        [["Overall"], overall_percentages.index], names=["School", "Line"]
    )

    # Combine counts and percentages for Overall
    overall_combined = pd.concat([overall_counts, overall_percentages], axis=1)
    overall_combined.columns = ["B", "H", "B%", "H%"]

    # Concatenate Overall first, then the rest
    final_combined = pd.concat([overall_combined, combined_sorted])

    # Ensure school order: Overall first, then CSV order
    school_levels = ["Overall"] + school_order
    final_combined.index = final_combined.index.set_levels(
        pd.CategoricalIndex(
            final_combined.index.levels[0],
            categories=school_levels,
            ordered=True,
        ),
        level=0,
    )

    # Sort by MultiIndex (school first, line second)
    final_combined = final_combined.sort_index(
        level=[0, 1], sort_remaining=False
    )

    return final_combined


if __name__ == "__main__":
    # read in timestamp csv
    CSV_FILENAME = "master_pat_games_timestamps.csv"
    df = pd.read_csv(CSV_FILENAME)

    # create dict of points by type
    point_dict = find_long_points(df)

    # format points
    formatted_durations = format_durations(point_dict, verbosity=1)

    # Get stats
    line_stats_df = get_line_stats(point_dict, df)
    # print(line_stats_df)
    line_stats_df.to_csv("line_stats.csv")

    # Save to output dir
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    durations_txt_path = f"{OUTPUT_DIR}/points_by_category_duration.txt"
    with open(durations_txt_path, "w") as f:
        f.write(formatted_durations)
    line_stats_path = f"{OUTPUT_DIR}/line_stats.csv"
    line_stats_df.to_csv(line_stats_path)
