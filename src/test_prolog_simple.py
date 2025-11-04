#!/usr/bin/env python3
"""
Simple test for Prolog crossword solver without GUI
"""

import time

# Check if pyswip and SWI-Prolog are available
try:
    from pyswip import Prolog

    PROLOG_AVAILABLE = True
except ImportError:
    Prolog = None
    PROLOG_AVAILABLE = False
    print("WARNING: pyswip not installed. Install with: pip install pyswip")
    exit(1)
except Exception as e:
    Prolog = None
    PROLOG_AVAILABLE = False
    print(f"WARNING: SWI-Prolog not found ({e})")
    exit(1)


class GridReader:
    """Simplified GridReader for testing"""

    def __init__(self, filename):
        self.grid = self.read_grid(filename)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0

    def read_grid(self, filename):
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            return [
                list(line.replace("*", "X").replace("#", "X").replace(" ", "_"))
                for line in lines
            ]

    def find_slots(self):
        slots = []
        slot_id = 0

        # Find ACROSS slots
        for row_idx in range(self.rows):
            col = 0
            while col < self.cols:
                if self.grid[row_idx][col] == "_" and (
                    col == 0 or self.grid[row_idx][col - 1] == "X"
                ):
                    start_col = col
                    length = 0
                    while col < self.cols and self.grid[row_idx][col] == "_":
                        length += 1
                        col += 1
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

        # Find DOWN slots
        for col_idx in range(self.cols):
            row = 0
            while row < self.rows:
                if self.grid[row][col_idx] == "_" and (
                    row == 0 or self.grid[row - 1][col_idx] == "X"
                ):
                    start_row = row
                    length = 0
                    while row < self.rows and self.grid[row][col_idx] == "_":
                        length += 1
                        row += 1
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

        return slots


class SimplePrologSolver:
    """Simple Prolog solver for testing"""

    def __init__(self):
        self.prolog = Prolog()
        self.slots = []

        # Load the Prolog file
        try:
            self.prolog.consult("crossword_solver.pl")
            print("✓ Loaded crossword_solver.pl")
        except Exception as e:
            raise RuntimeError(f"Failed to load crossword_solver.pl: {e}")

    def load_slots(self, slots):
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
        words = []
        try:
            with open(filename, "r") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        word = word.replace("'", "_")
                        words.append(word)
            print(f"✓ Loaded {len(words)} words from {filename}")
        except FileNotFoundError:
            words = ["dog", "cat", "rat", "art", "now", "aware", "test"]
            print(f"✓ Using default words: {words}")

        # Clear existing words
        for result in self.prolog.query("retractall(word(_))"):
            pass

        # Add words to Prolog
        for word in words:
            query = f"assertz(word({word}))"
            list(self.prolog.query(query))

    def solve(self, slot_ids):
        print(f"\n🔍 Solving with Prolog for slots: {slot_ids}")

        # Clear any existing placements
        for result in self.prolog.query("clear_placements"):
            pass

        # Convert slot_ids to Prolog list format
        slot_list = str(slot_ids).replace(" ", "")

        # Try to find a solution
        query = f"solve_crossword({slot_list}, Solution)"
        print(f"Query: {query}")

        try:
            solutions = list(self.prolog.query(query))
            print(f"Found {len(solutions)} solution(s)")
        except Exception as e:
            print(f"Prolog query failed: {e}")
            return None

        if solutions:
            solution = solutions[0]["Solution"]
            print(f"Raw solution: {solution}")

            # Parse solution
            result = []
            for item in solution:
                if (
                    isinstance(item, str)
                    and item.startswith(",(")
                    and item.endswith(")")
                ):
                    # Parse ",(slot_id, word)" format
                    content = item[2:-1]  # Remove ",(" and ")"
                    parts = content.split(", ", 1)
                    if len(parts) == 2:
                        try:
                            slot_id = int(parts[0])
                            word = parts[1].strip()
                            result.append((slot_id, word))
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse: {item}")

            print(f"Parsed solution: {result}")
            return result

        print("No solution found")
        return None


def test_prolog_solver():
    print("=== SIMPLE PROLOG SOLVER TEST ===\n")

    # Read grid
    print("1. Reading grid...")
    grid_reader = GridReader("grid.txt")
    print(f"   Grid: {grid_reader.rows} x {grid_reader.cols}")

    # Find slots
    print("\n2. Finding slots...")
    slots = grid_reader.find_slots()
    print(f"   Found {len(slots)} slots:")
    for slot in slots:
        print(
            f"   Slot {slot['id']}: {slot['direction']} at ({slot['row']},{slot['col']}) len={slot['length']}"
        )

    # Initialize solver
    print("\n3. Initializing Prolog solver...")
    try:
        solver = SimplePrologSolver()
    except Exception as e:
        print(f"   Failed to initialize: {e}")
        return

    # Load data
    print("\n4. Loading data...")
    solver.load_slots(slots)
    solver.load_words("word.txt")

    # Test with different slot combinations
    test_cases = [
        [slots[0]["id"]] if slots else [],
        [s["id"] for s in slots[:2]] if len(slots) >= 2 else [],
        [s["id"] for s in slots[:3]] if len(slots) >= 3 else [],
        [s["id"] for s in slots] if len(slots) <= 5 else [s["id"] for s in slots[:5]],
    ]

    for i, slot_ids in enumerate(test_cases):
        if not slot_ids:
            continue
        print(f"\n5.{i + 1} Testing with slots {slot_ids}...")

        start_time = time.time()
        solution = solver.solve(slot_ids)
        end_time = time.time()

        if solution:
            print(f"   ✓ Solution found in {end_time - start_time:.3f}s:")
            for slot_id, word in solution:
                slot = slots[slot_id]
                print(
                    f"     Slot {slot_id} ({slot['direction']} at {slot['row']},{slot['col']}): {word.upper()}"
                )
        else:
            print(f"   ✗ No solution found (took {end_time - start_time:.3f}s)")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_prolog_solver()
