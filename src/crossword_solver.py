import time
import sys
import os
from src.solver import PythonCrosswordSolver
from src.drawer import CrosswordDrawer
from src.reader import GridReader

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

            query = (
                f"assertz(slot({slot_id}, {direction.lower()}, {row}, {col}, {length}))"
            )
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

    # -----------------------------
    # Trace utilities
    # -----------------------------
    def clear_steps(self):
        """Clear Prolog step log."""
        try:
            for _ in self.prolog.query("clear_steps"):
                pass
        except Exception:
            pass

    def fetch_steps(self):
        """Fetch recorded steps from Prolog as a list of dicts."""
        steps = []
        try:
            # Query as list-of-lists to avoid functor parsing differences across pyswip versions
            q = "findall([I,T,S,W,N], step(I,T,S,W,N), Steps)"
            out = list(self.prolog.query(q))
            if not out:
                return steps
            for row in out[0]["Steps"]:
                try:
                    # Expect row like [I,T,S,W,N]
                    i = int(str(row[0]))
                    t = str(row[1])
                    s = int(str(row[2]))
                    w = str(row[3])
                    n = str(row[4])
                    steps.append({
                        "i": i,
                        "type": t,
                        "slot": s,
                        "word": w,
                        "note": n,
                    })
                except Exception:
                    # Skip malformed entries
                    continue
        except Exception as e:
            print(f"WARNING: fetch_steps failed: {e}")
        return steps

    def solve_with_trace(self, slot_ids):
        """Solve and return (solution, steps) for replay visualization."""
        # Ensure a clean trace and placements
        self.clear_steps()
        for _ in self.prolog.query("clear_placements"):
            pass

        slot_list = str(slot_ids).replace(" ", "")
        query = f"trace_solve({slot_list}, Solution, Steps)"
        try:
            sols = list(self.prolog.query(query))
        except Exception as e:
            print(f"Prolog trace_solve failed: {e}")
            return None, []

        if not sols:
            return None, []

        solution_term = sols[0].get("Solution", [])

        # Parse solution similar to solve()
        result = []
        placements = {}
        try:
            for item in solution_term:
                if (
                    isinstance(item, str)
                    and item.startswith(",(")
                    and item.endswith(")")
                ):
                    content = item[2:-1]
                    parts = content.split(", ", 1)
                    if len(parts) == 2:
                        slot_id = int(parts[0])
                        word = parts[1].strip()
                        result.append((slot_id, word))
                        placements[slot_id] = word
                elif hasattr(item, "functor") and item.functor.name == ",":
                    slot_id = int(str(item.args[0]))
                    word = str(item.args[1])
                    result.append((slot_id, word))
                    placements[slot_id] = word
        except Exception as e:
            print(f"WARNING: Failed parsing solution term: {e}")

        steps = self.fetch_steps()
        return result if result else None, steps


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
