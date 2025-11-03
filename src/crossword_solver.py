"""
============================================================================
CROSSWORD PUZZLE SOLVER - PYTHON MAIN PROGRAM (FIXED)
Beautiful UI with Full Backtracking & Confidence-Based Heuristic
============================================================================
"""

import turtle
import os
import time
import sys

# Check if pyswip is available
try:
    from pyswip import Prolog

    PROLOG_AVAILABLE = True
except ImportError:
    PROLOG_AVAILABLE = False
    print("WARNING: pyswip not installed. Install with: pip install pyswip")
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


class PythonCrosswordSolver:
    """Python-based solver with confidence heuristic"""

    def __init__(self):
        self.words = []
        self.constraints = {}
        self.slots = []
        self.step_callback = None
        self.current_placements = {}  # Track current placements for visualization
        print("✓ Solver choice: Using Python solver with confidence heuristic")

    def set_step_callback(self, callback):
        """Set callback function to visualize each step"""
        self.step_callback = callback

    def load_slots(self, slots):
        self.slots = slots

    def load_words(self, filename):
        loaded = False

        if filename == "":
            raise ValueError("No wordlist filename provided")

        try:
            with open(filename, "r") as f:
                self.words = []
                for line in f:
                    word = line.strip().lower()
                    if word:
                        word = word.replace("'", "_")
                        self.words.append(word)
                print(f"✓ Loaded {len(self.words)} words from {filename}")
                loaded = True
        except FileNotFoundError:
            print(f"ERROR: {filename} not found!")

    def load_constraints(self, filename, slots):
        try:
            with open(filename, "r") as f:
                count = 0
                mismatches = 0
                for line_idx, line in enumerate(f):
                    line = line.strip().lower()
                    if not line:
                        continue

                    if ":" in line:
                        parts = line.split(":", 1)
                        try:
                            slot_id = int(parts[0].strip())
                            pattern = parts[1].strip()
                        except ValueError:
                            print(
                                f"  Warning: Invalid slot_id in line {line_idx+1}, skipping"
                            )
                            continue
                    else:
                        slot_id = line_idx
                        pattern = line

                    slot_match = None
                    for slot in slots:
                        if slot["id"] == slot_id:
                            slot_match = slot
                            break

                    if slot_match:
                        if len(pattern) != slot_match["length"]:
                            print(
                                f"  Warning: Constraint for slot {slot_id} has length {len(pattern)}, but slot length is {slot_match['length']} - ignoring"
                            )
                            mismatches += 1
                            continue

                        self.constraints[slot_id] = pattern
                        print(f"  Constraint slot {slot_id}: {pattern}")
                        count += 1
                    else:
                        print(
                            f"  Warning: Constraint references non-existent slot {slot_id}, ignoring"
                        )

                if count > 0:
                    print(f"✓ Loaded {count} constraints from {filename}")
                if mismatches > 0:
                    print(
                        f"  ({mismatches} constraint(s) ignored due to length mismatch)"
                    )
        except FileNotFoundError:
            print(f"  No {filename} found (constraints are optional)")

    def matches_pattern(self, word, pattern):
        """Check if word matches pattern (e.g., dog*** matches doctor)"""
        if len(word) != len(pattern):
            return False
        for w, p in zip(word, pattern):
            if p != "*" and p != w:
                return False
        return True

    def get_intersection(self, slot1, slot2):
        """Get intersection point between two slots if it exists"""
        if slot1["direction"] == slot2["direction"]:
            return None

        if slot1["direction"] == "across" and slot2["direction"] == "down":
            if (
                slot2["row"] <= slot1["row"] < slot2["row"] + slot2["length"]
                and slot1["col"] <= slot2["col"] < slot1["col"] + slot1["length"]
            ):
                pos1 = slot2["col"] - slot1["col"]
                pos2 = slot1["row"] - slot2["row"]
                return (pos1, pos2)

        elif slot1["direction"] == "down" and slot2["direction"] == "across":
            if (
                slot1["row"] <= slot2["row"] < slot1["row"] + slot1["length"]
                and slot2["col"] <= slot1["col"] < slot2["col"] + slot2["length"]
            ):
                pos1 = slot2["row"] - slot1["row"]
                pos2 = slot1["col"] - slot2["col"]
                return (pos1, pos2)

        return None

    def check_intersection(self, placements):
        """Check if current placements have valid intersections"""
        slot_ids = list(placements.keys())

        for i, slot1_id in enumerate(slot_ids):
            for slot2_id in slot_ids[i + 1 :]:
                slot1 = self.slots[slot1_id]
                slot2 = self.slots[slot2_id]
                word1 = placements[slot1_id]
                word2 = placements[slot2_id]

                intersection = self.get_intersection(slot1, slot2)

                if intersection:
                    pos1, pos2 = intersection
                    if word1[pos1] != word2[pos2]:
                        return False

        return True

    def calculate_confidence(self, slot_id, word, placements, used_words):
        """
        Calculate confidence score for placing a word in a slot.
        Score = number of intersecting slots that still have valid candidates.
        """
        slot = self.slots[slot_id]
        score = 0

        for other_slot in self.slots:
            other_id = other_slot["id"]

            if other_id == slot_id or other_id in placements:
                continue

            if slot["direction"] == other_slot["direction"]:
                continue

            intersection = self.get_intersection(slot, other_slot)
            if not intersection:
                continue

            pos_in_slot, pos_in_other = intersection
            letter = word[pos_in_slot]

            candidate_exists = False
            for candidate_word in self.words:
                if len(candidate_word) != other_slot["length"]:
                    continue

                if candidate_word in used_words:
                    continue

                if candidate_word[pos_in_other] != letter:
                    continue

                if other_id in self.constraints:
                    if not self.matches_pattern(
                        candidate_word, self.constraints[other_id]
                    ):
                        continue

                # Check if this candidate would be compatible with existing placements
                test_placements = placements.copy()
                test_placements[other_id] = candidate_word
                if not self.check_intersection(test_placements):
                    continue

                candidate_exists = True
                break

            if candidate_exists:
                score += 1

        return score

    def solve(self, slot_ids):
        """Backtracking solver with confidence-based heuristic"""
        placements = {}
        used_words = set()
        self.current_placements = placements

        def backtrack(slot_idx):
            if slot_idx >= len(slot_ids):
                return True

            slot_id = slot_ids[slot_idx]
            slot = self.slots[slot_id]

            # Collect all valid candidates with their confidence scores
            candidates = []
            for word in self.words:
                if len(word) != slot["length"]:
                    continue

                if word in used_words:
                    continue

                if slot_id in self.constraints:
                    if not self.matches_pattern(word, self.constraints[slot_id]):
                        continue

                confidence = self.calculate_confidence(
                    slot_id, word, placements, used_words
                )
                candidates.append((confidence, word))

            # Sort by confidence (highest first)
            candidates.sort(reverse=True, key=lambda x: x[0])

            # Try candidates in order of confidence
            for confidence, word in candidates:
                # FIX: Only prune zero-confidence if there are intersections
                if confidence == 0 and len(candidates) > 1:
                    has_intersections = any(
                        self.get_intersection(slot, self.slots[other_id]) is not None
                        for other_id in slot_ids
                        if other_id != slot_id and other_id not in placements
                    )
                    if has_intersections:
                        continue

                placements[slot_id] = word
                used_words.add(word)

                if self.check_intersection(placements):
                    if self.step_callback:
                        self.step_callback(slot_id, word, slot, True, placements.copy())

                    if backtrack(slot_idx + 1):
                        return True

                if self.step_callback:
                    self.step_callback(slot_id, word, slot, False, placements.copy())

                del placements[slot_id]
                used_words.remove(word)

            return False

        if backtrack(0):
            return [(sid, placements[sid]) for sid in slot_ids if sid in placements]
        return None


class CrosswordDrawer:
    """Beautiful crossword drawer with modern UI (FIXED)"""

    def __init__(self, grid, cell_size=60, animate=True):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.cell_size = cell_size
        self.animate = animate
        self.slots_storage = []
        self.current_placements = {}

        # Color scheme
        self.colors = {
            "background": "#2C3E50",
            "cell_bg": "#FFFFFF",  # Pure white for empty cells
            "cell_border": "#34495E",
            "blocked": "#34495E",
            "letter": "#2C3E50",
            "letter_placing": "#27AE60",
            "letter_backtrack": "#E74C3C",
            "title": "#ECF0F1",
        }

        width = max(900, self.cols * cell_size + 200)
        height = max(700, self.rows * cell_size + 250)

        self.screen = turtle.Screen()
        self.screen.setup(width=width, height=height)
        self.screen.title("Crossword Puzzle Solver - Fixed Version")
        self.screen.bgcolor(self.colors["background"])

        self.pen = turtle.Turtle()
        self.pen.speed(0)
        self.pen.hideturtle()

        self.info = turtle.Turtle()
        self.info.speed(0)
        self.info.hideturtle()
        self.info.penup()

        if animate:
            self.screen.tracer(1)
        else:
            self.screen.tracer(0)

    def draw_grid(self):
        """Draw beautiful crossword grid (FIXED: lighter empty cells)"""
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2

        # Draw outer border
        self.pen.penup()
        self.pen.goto(start_x - 5, start_y + 5)
        self.pen.pendown()
        self.pen.pensize(8)
        self.pen.color(self.colors["cell_border"])
        for _ in range(4):
            if _ % 2 == 0:
                self.pen.forward(self.cols * self.cell_size + 10)
            else:
                self.pen.forward(self.rows * self.cell_size + 10)
            self.pen.right(90)

        # Draw cells (FIX: Don't fill empty cells, only draw borders)
        self.pen.pensize(2)
        for row in range(self.rows):
            for col in range(self.cols):
                x = start_x + col * self.cell_size
                y = start_y - row * self.cell_size

                self.pen.penup()
                self.pen.goto(x, y)
                self.pen.pendown()

                if self.grid[row][col] == "X":
                    # Blocked cells: filled
                    self.pen.fillcolor(self.colors["blocked"])
                    self.pen.color(self.colors["blocked"])
                    self.pen.begin_fill()
                    for _ in range(4):
                        self.pen.forward(self.cell_size)
                        self.pen.right(90)
                    self.pen.end_fill()
                else:
                    # Empty cells: white background with border only
                    self.pen.fillcolor(self.colors["cell_bg"])
                    self.pen.color(self.colors["cell_border"])
                    self.pen.begin_fill()
                    for _ in range(4):
                        self.pen.forward(self.cell_size)
                        self.pen.right(90)
                    self.pen.end_fill()

        self.screen.update()

    def draw_letter(self, x, y, letter, color=None):
        """Draw a letter with beautiful typography"""
        if color is None:
            color = self.colors["letter"]

        self.pen.penup()
        self.pen.goto(x + self.cell_size / 2, y - self.cell_size * 0.72)
        self.pen.color(color)
        self.pen.write(
            letter.upper(),
            align="center",
            font=("Arial", int(self.cell_size * 0.55), "bold"),
        )

    def clear_cell(self, row, col):
        """Clear a cell and redraw any intersecting letters (FIXED)"""
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2
        x = start_x + col * self.cell_size
        y = start_y - row * self.cell_size

        self.pen.penup()
        self.pen.goto(x, y)
        self.pen.pendown()
        self.pen.pensize(2)

        # Redraw cell background
        if self.grid[row][col] == "X":
            self.pen.fillcolor(self.colors["blocked"])
            self.pen.color(self.colors["blocked"])
        else:
            self.pen.fillcolor(self.colors["cell_bg"])
            self.pen.color(self.colors["cell_border"])

        self.pen.begin_fill()
        for _ in range(4):
            self.pen.forward(self.cell_size)
            self.pen.right(90)
        self.pen.end_fill()

        # FIX: Redraw any letter that should still be there from other placements
        for sid, word in self.current_placements.items():
            if sid >= len(self.slots_storage):
                continue
            slot = self.slots_storage[sid]

            if slot["direction"] == "across":
                if (
                    row == slot["row"]
                    and slot["col"] <= col < slot["col"] + slot["length"]
                ):
                    letter_idx = col - slot["col"]
                    if letter_idx < len(word):
                        self.draw_letter(x, y, word[letter_idx])
            else:  # down
                if (
                    col == slot["col"]
                    and slot["row"] <= row < slot["row"] + slot["length"]
                ):
                    letter_idx = row - slot["row"]
                    if letter_idx < len(word):
                        self.draw_letter(x, y, word[letter_idx])

    def update_info(self, message, color=None):
        """Update info message at top"""
        if color is None:
            color = self.colors["title"]

        self.info.clear()
        self.info.goto(0, (self.rows * self.cell_size) / 2 + 70)
        self.info.color(color)
        self.info.write(message, align="center", font=("Arial", 18, "bold"))

    def animate_word_placement(self, slot_id, word, slot, is_placing, placements):
        """Animate placing or removing a word (FIXED with placements tracking)"""
        self.current_placements = placements
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2

        row, col = slot["row"], slot["col"]
        direction = slot["direction"]

        if is_placing:
            self.update_info(
                f"Trying: {word.upper()} ({direction}) at slot {slot_id}",
                self.colors["letter_placing"],
            )

            for i, letter in enumerate(word):
                if direction == "across":
                    x = start_x + (col + i) * self.cell_size
                    y = start_y - row * self.cell_size
                else:
                    x = start_x + col * self.cell_size
                    y = start_y - (row + i) * self.cell_size

                self.draw_letter(x, y, letter, self.colors["letter_placing"])
                self.screen.update()
                time.sleep(0.05)  # Slightly faster
        else:
            self.update_info(
                f"Backtracking: {word.upper()} from slot {slot_id}",
                self.colors["letter_backtrack"],
            )

            for i in range(len(word)):
                if direction == "across":
                    r, c = row, col + i
                else:
                    r, c = row + i, col

                self.clear_cell(r, c)
                self.screen.update()
                time.sleep(0.03)

    def draw_solution(self, slots, solution, animated=False):
        """Draw the complete solved crossword"""
        self.slots_storage = slots
        self.draw_grid()

        if not solution:
            self.update_info("NO SOLUTION FOUND", self.colors["letter_backtrack"])
            self.screen.update()
            print("\n✗ No solution found")
            print("✓ Close window to exit")
            turtle.done()
            return

        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2

        # Draw final solution
        for slot_id, word in solution:
            slot = slots[slot_id]
            row, col = slot["row"], slot["col"]
            direction = slot["direction"]

            if animated:
                self.update_info(
                    f"Final: {word.upper()} ({direction})", self.colors["letter"]
                )
                time.sleep(0.2)

            for i, letter in enumerate(word):
                if direction == "across":
                    x = start_x + (col + i) * self.cell_size
                    y = start_y - row * self.cell_size
                else:
                    x = start_x + col * self.cell_size
                    y = start_y - (row + i) * self.cell_size

                self.draw_letter(x, y, letter, self.colors["letter"])

        self.screen.update()
        self.update_info("✓ PUZZLE SOLVED!", self.colors["letter_placing"])
        print("\n✓ Close window to exit")
        turtle.done()


def print_ascii_grid(grid, solution, slots):
    """Print the crossword grid in ASCII format"""
    rows = len(grid)
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
    wordlist = sys.argv[1] if len(sys.argv) > 1 else "word.txt"
    grid = sys.argv[2] if len(sys.argv) > 2 else "grid.txt"

    print("=" * 70)
    print("   CROSSWORD PUZZLE SOLVER (FIXED VERSION)")
    print("   Confidence-Based Heuristic + Full Backtracking")
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

    print("\n[3] Initializing graphics...")
    drawer = CrosswordDrawer(grid_reader.grid, cell_size=60, animate=True)
    drawer.draw_grid()

    print("\n[4] Initializing solver...")
    solver = PythonCrosswordSolver()
    solver.set_step_callback(drawer.animate_word_placement)

    print("\n[5] Loading data...")
    solver.load_slots(slots)
    solver.load_words(str(wordlist))
    solver.load_constraints("constraints.txt", slots)

    print("\n[6] Solving with confidence-based heuristic...")

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
        drawer.draw_solution(slots, solution, animated=True)
    else:
        print(
            f"\n✗ No solution found (searched for {end_time - start_time:.2f} seconds)"
        )
        drawer.draw_solution(slots, None)


if __name__ == "__main__":
    main()
