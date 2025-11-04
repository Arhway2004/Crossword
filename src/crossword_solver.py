"""
============================================================================
CROSSWORD PUZZLE SOLVER - PYTHON MAIN PROGRAM (FIXED)
Beautiful UI with Full Backtracking & Confidence-Based Heuristic
============================================================================
"""

import time
import sys
import os
from solver import PythonCrosswordSolver
from drawer import CrosswordDrawer

# Check if pyswip and SWI-Prolog are available
try:
    from pyswip import Prolog

    PROLOG_AVAILABLE = True
except ImportError:
    Prolog = None
    PROLOG_AVAILABLE = False
    print("WARNING: pyswip not installed. Install with: pip install pyswip")
    print("Falling back to Python-only solver...\n")
except Exception as e:
    Prolog = None
    PROLOG_AVAILABLE = False
    print(f"WARNING: SWI-Prolog not found ({e})")
    print("Please install SWI-Prolog from: https://www.swi-prolog.org/download/stable")
    print("Falling back to Python-only solver...\n")


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
            print(f"ERROR: {filename} not found!")
            print("Creating sample grid.txt...")
            self.create_sample_grid()
            return self.read_grid(filename)

    def create_sample_grid(self):
        """Create a sample grid file"""
        sample = """___XXXX
_XXXXXX
_XXXXXX
_XXXXXX
_XXXXXX
___XXXX"""
        with open("grid.txt", "w") as f:
            f.write(sample)
        print("✓ Created sample grid.txt")

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


class PrologCrosswordSolver:
    """Prolog-based solver using the crossword_solver.pl file"""

    def __init__(self):
        if not PROLOG_AVAILABLE or Prolog is None:
            raise ImportError(
                "pyswip is required for PrologCrosswordSolver. Install with: pip install pyswip"
            )

        self.prolog = Prolog()
        self.slots = []
        self.step_callback = None
        self.current_placements = {}

        # Load the Prolog file
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prolog_file = os.path.join(script_dir, "crossword_solver.pl")
            self.prolog.consult(prolog_file)
            print("✓ Solver choice: Using Prolog solver with confidence heuristic")
        except Exception as e:
            raise RuntimeError(f"Failed to load crossword_solver.pl: {e}")

    def set_step_callback(self, callback):
        """Set callback function to visualize each step"""
        self.step_callback = callback

    def load_slots(self, slots):
        """Load slots into Prolog knowledge base"""
        self.slots = slots

        # Clear existing slots
        for result in self.prolog.query("retractall(slot(_, _, _, _, _))"):
            pass

        # Add slots to Prolog
        for slot in slots:
            slot_id = slot["id"]
            direction = slot["direction"]
            row = slot["row"]
            col = slot["col"]
            length = slot["length"]

            query = f"assertz(slot({slot_id}, {direction}, {row}, {col}, {length}))"
            list(self.prolog.query(query))

    def load_words(self, filename):
        """Load words into Prolog knowledge base"""
        # Try word.txt first, then words.txt
        filenames = [filename, "word.txt", "words.txt"]
        loaded = False
        words = []

        for fname in filenames:
            try:
                with open(fname, "r") as f:
                    words = []
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            word = word.replace("'", "_")
                            words.append(word)
                    print(f"✓ Loaded {len(words)} words from {fname}")
                    loaded = True
                    break
            except FileNotFoundError:
                continue

        if not loaded:
            raise FileNotFoundError(
                f"ERROR: No wordlist file found among: {', '.join(filenames)}"
            )

        # Clear existing words
        for result in self.prolog.query("retractall(word(_))"):
            pass

        # Add words to Prolog
        for word in words:
            query = f"assertz(word({word}))"
            list(self.prolog.query(query))

    def load_constraints(self, filename, slots):
        """Load constraints into Prolog knowledge base"""
        # Clear existing constraints
        for result in self.prolog.query("retractall(constraint(_, _))"):
            pass

        try:
            with open(filename, "r") as f:
                count = 0
                mismatches = 0
                for line_idx, line in enumerate(f):
                    line = line.strip().lower()
                    if not line:
                        continue

                    parts = line.split(":")
                    if len(parts) != 2:
                        print(
                            f"    Warning: Invalid constraint format at line {line_idx + 1}: '{line}'"
                        )
                        continue

                    try:
                        slot_id = int(parts[0].strip())
                        pattern = parts[1].strip()

                        # Validate slot exists
                        if slot_id >= len(slots):
                            print(
                                f"    Warning: Slot {slot_id} doesn't exist (line {line_idx + 1})"
                            )
                            mismatches += 1
                            continue

                        # Validate pattern length
                        expected_len = slots[slot_id]["length"]
                        if len(pattern) != expected_len:
                            print(
                                f"    Warning: Pattern '{pattern}' length {len(pattern)} doesn't match slot {slot_id} length {expected_len}"
                            )
                            mismatches += 1
                            continue

                        # Add constraint to Prolog
                        query = f"assertz(constraint({slot_id}, '{pattern}'))"
                        list(self.prolog.query(query))
                        count += 1

                    except ValueError:
                        print(
                            f"    Warning: Invalid slot ID at line {line_idx + 1}: '{parts[0]}'"
                        )
                        continue

                if count > 0:
                    print(f"✓ Loaded {count} constraints from {filename}")
                    if mismatches > 0:
                        print(f"  (Skipped {mismatches} invalid constraints)")
                else:
                    print(f"  No valid constraints found in {filename}")

        except FileNotFoundError:
            print(
                f"  No constraints file '{filename}' found - continuing without constraints"
            )

    def solve(self, slot_ids):
        """Solve using Prolog with confidence-based heuristic"""
        # Clear any existing placements
        for result in self.prolog.query("clear_placements"):
            pass
        print(f"  Cleared existing placements")

        # Convert slot_ids to Prolog list format
        slot_list = str(slot_ids).replace(" ", "")
        print(f"  Solving for slots: {slot_list}")

        # Try to find a solution
        query = f"solve_crossword({slot_list}, Solution)"
        print(f"  Prolog query: {query}")

        try:
            solutions = list(self.prolog.query(query))
            print(f"  Found {len(solutions)} solution(s)")
        except Exception as e:
            print(f"  Prolog query failed: {e}")
            return None

        if solutions:
            solution = solutions[0]["Solution"]
            print(f"  Raw solution from Prolog: {solution}")
            print(f"  Solution type: {type(solution)}")

            # Convert Prolog solution format to Python format
            result = []
            placements = {}

            for item in solution:
                # Handle the string format returned by pyswip: ",(slot_id, word)"
                if (
                    isinstance(item, str)
                    and item.startswith(",(")
                    and item.endswith(")")
                ):
                    # Parse ",(slot_id, word)" format
                    content = item[2:-1]  # Remove ",(" and ")"
                    parts = content.split(", ", 1)  # Split on first comma-space
                    if len(parts) == 2:
                        try:
                            slot_id = int(parts[0])
                            word = parts[1].strip()
                            result.append((slot_id, word))
                            placements[slot_id] = word

                            # Call step callback if available
                            if self.step_callback and slot_id < len(self.slots):
                                slot = self.slots[slot_id]
                                self.step_callback(
                                    slot_id, word, slot, True, placements.copy()
                                )
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse solution item: {item}")
                elif hasattr(item, "functor") and item.functor.name == ",":
                    # Handle compound term format (backup)
                    slot_id = int(str(item.args[0]))
                    word = str(item.args[1])
                    result.append((slot_id, word))
                    placements[slot_id] = word

                    # Call step callback if available
                    if self.step_callback and slot_id < len(self.slots):
                        slot = self.slots[slot_id]
                        self.step_callback(slot_id, word, slot, True, placements.copy())
                else:
                    print(
                        f"Warning: Unknown solution format: {item} (type: {type(item)})"
                    )

            print(f"  Parsed {len(result)} placements: {result}")
            return result

        print("  No solution found by Prolog solver")
        return None


def print_ascii_grid(grid, solution, slots):
    """Print ASCII representation of solved crossword"""
    cols = len(grid[0]) if grid else 0
    filled_grid = [row[:] for row in grid]

    for slot_id, word in solution:
        slot = slots[slot_id]
        row, col = slot["row"], slot["col"]
        direction = slot["direction"]

        for i, letter in enumerate(word):
            if direction == "across":
                filled_grid[row][col + i] = letter.upper()
            else:
                filled_grid[row + i][col] = letter.upper()

    print("\n" + "=" * (cols * 4 + 1))
    print("SOLVED CROSSWORD GRID (ASCII)")
    print("=" * (cols * 4 + 1))

    for row in filled_grid:
        line = "|"
        for cell in row:
            if cell == "X":
                line += "###|"
            elif cell == "_":
                line += "   |"
            else:
                line += f" {cell} |"
        print(line)
        print("-" * (cols * 4 + 1))

    print()


def main():
    """Main program entry point"""
    # Check for command-line arguments
    nogui = "--nogui" in sys.argv or "-n" in sys.argv
    grid = sys.argv[1] if len(sys.argv) > 1 else "grid.txt"
    wordlist = sys.argv[2] if len(sys.argv) > 2 else "word.txt"

    # Check for command-line arguments
    nogui = "--nogui" in sys.argv or "-n" in sys.argv

    print("=" * 70)
    print("   CROSSWORD PUZZLE SOLVER (FIXED VERSION)")
    print("   Confidence-Based Heuristic + Full Backtracking")
    if nogui:
        print("   Running in NO-GUI mode")
    print("=" * 70)

    print("\n[1] Reading grid...")
    grid_reader = GridReader(str(grid))
    print(f"    Grid: {grid_reader.rows} x {grid_reader.cols}")

    print("\n[2] Finding slots...")
    slots = grid_reader.find_slots()
    print(f"    Found {len(slots)} slots:")
    for slot in slots:
        print(
            f"      Slot {slot['id']}: {slot['direction']:6} at ({slot['row']},{slot['col']}) len={slot['length']}"
        )

    # Initialize graphics only if not in nogui mode
    drawer = None
    if not nogui:
        try:
            print("\n[3] Initializing graphics...")
            drawer = CrosswordDrawer(grid_reader.grid, cell_size=60, animate=True)
            drawer.draw_grid()
        except Exception as e:
            print(f"\n[3] Graphics initialization failed: {e}")
            print("    Continuing in no-GUI mode...")
            nogui = True

    print(f"\n[{'4' if nogui else '4'}] Initializing solver...")
    # Try Prolog solver first, fall back to Python solver if not available
    try:
        if PROLOG_AVAILABLE:
            solver = PrologCrosswordSolver()
        else:
            raise ImportError("Prolog not available")
    except Exception as e:
        print(f"  Prolog solver unavailable: {e}")
        print("  Using Python solver instead...")
        solver = PythonCrosswordSolver()

    # Set callback only if graphics are available
    if drawer and not nogui:
        solver.set_step_callback(drawer.animate_word_placement)

    print(f"\n[{'5' if nogui else '5'}] Loading data...")
    solver.load_slots(slots)
    solver.load_words(str(wordlist))
    solver.load_constraints("constraints.txt", slots)

    print(f"\n[{'6' if nogui else '6'}] Solving with confidence-based heuristic...")

    slot_ids = [s["id"] for s in slots]
    start_time = time.time()
    solution = solver.solve(slot_ids)
    end_time = time.time()

    if solution:
        print(f"\n✓ Solution found in {end_time - start_time:.2f} seconds!")
        print("\nFinal solution:")
        for slot_id, word in solution:
            slot = slots[slot_id]
            print(
                f"  Slot {slot_id} ({slot['direction']:6} at {slot['row']},{slot['col']}): {word.upper()}"
            )

        print_ascii_grid(grid_reader.grid, solution, slots)

        if drawer and not nogui:
            drawer.draw_solution(slots, solution, animated=True)
        else:
            print("\n✓ Crossword solved successfully!")
            if nogui:
                print(
                    "  Use 'python3 crossword_solver.py' (without --nogui) for visual display"
                )

        if drawer and not nogui:
            drawer.draw_solution(slots, solution, animated=True)
        else:
            print("\n✓ Crossword solved successfully!")
            if nogui:
                print(
                    "  Use 'python3 crossword_solver.py' (without --nogui) for visual display"
                )
    else:
        print(
            f"\n✗ No solution found (searched for {end_time - start_time:.2f} seconds)"
        )
        if drawer and not nogui:
            drawer.draw_solution(slots, None)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage: python3 crossword_solver.py [options]")
        print("Options:")
        print("  --nogui, -n    Run without graphical interface")
        print("  --help, -h     Show this help message")
    else:
        main()
