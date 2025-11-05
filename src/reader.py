import sys


class GridReader:
    """Reads and parses crossword grid from file"""

    def __init__(self, filename):
        self.grid = self.read_grid(filename)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0

    def read_grid(self, filename):
        """Read grid from file where _ = empty, X/# = blocked"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read().strip()

                # Check if grid uses pipe separators
                if "|" in content:
                    print("  Grid normalization: Detected pipe-separated format")
                    rows = [line.strip() for line in content.split("|") if line.strip()]
                else:
                    print("  Grid normalization: Using line-per-row format")
                    rows = [
                        line.strip() for line in content.split("\n") if line.strip()
                    ]

                # Normalize characters: * -> X, # -> X, space -> _
                normalized_rows = []
                for row in rows:
                    normalized = (
                        row.replace("*", "X").replace("#", "X").replace(" ", "_")
                    )
                    normalized_rows.append(list(normalized))

                # Validate only X and _ remain
                for i, row in enumerate(normalized_rows):
                    for j, cell in enumerate(row):
                        if cell not in ["X", "_"]:
                            raise ValueError(
                                f"Invalid character '{cell}' at row {i}, col {j}. Only 'X', '#' and '_' allowed after normalization."
                            )

                # Ensure all rows have same length
                if normalized_rows:
                    max_len = max(len(row) for row in normalized_rows)
                    for i, row in enumerate(normalized_rows):
                        if len(row) < max_len:
                            deficit = max_len - len(row)
                            if deficit > 2:
                                raise ValueError(
                                    f"Row {i} has length {len(row)}, expected {max_len} (difference too large)"
                                )
                            print(
                                f"  Warning: Padding row {i} with {deficit} 'X' cells"
                            )
                            row.extend(["X"] * deficit)

                print(
                    f"  Grid normalized: {len(normalized_rows)} rows x {len(normalized_rows[0]) if normalized_rows else 0} cols"
                )
                return normalized_rows

        except FileNotFoundError:
            print(f"ERROR: Grid file '{filename}' not found.")
            sys.exit(1)
        except ValueError as ve:
            print(f"ERROR: {ve}")
            sys.exit(1)

    def find_slots(self):
        """Find all word slots (across and down) with length >= 2"""
        slots = []
        slot_id = 0

        # Find ACROSS slots (horizontal)
        for row_idx in range(self.rows):
            col = 0
            while col < self.cols:
                # Start slot if current cell is empty AND (at left edge OR left cell is blocked)
                if self.grid[row_idx][col] == "_" and (
                    col == 0 or self.grid[row_idx][col - 1] == "X"
                ):
                    start_col = col
                    length = 0
                    # Extend right while empty
                    while col < self.cols and self.grid[row_idx][col] == "_":
                        length += 1
                        col += 1
                    # Record if length >= 2
                    if length >= 2:
                        slots.append(
                            {
                                "id": slot_id,
                                "direction": "across",
                                "row": row_idx,
                                "col": start_col,
                                "length": length,
                            }
                        )
                        slot_id += 1
                else:
                    col += 1

        # Find DOWN slots (vertical)
        for col_idx in range(self.cols):
            row = 0
            while row < self.rows:
                # Start slot if current cell is empty AND (at top edge OR top cell is blocked)
                if self.grid[row][col_idx] == "_" and (
                    row == 0 or self.grid[row - 1][col_idx] == "X"
                ):
                    start_row = row
                    length = 0
                    # Extend down while empty
                    while row < self.rows and self.grid[row][col_idx] == "_":
                        length += 1
                        row += 1
                    # Record if length >= 2
                    if length >= 2:
                        slots.append(
                            {
                                "id": slot_id,
                                "direction": "down",
                                "row": start_row,
                                "col": col_idx,
                                "length": length,
                            }
                        )
                        slot_id += 1
                else:
                    row += 1

        print(f"  Slot detection: Found {len(slots)} slots")
        return slots
