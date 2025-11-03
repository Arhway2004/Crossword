"""
============================================================================
CROSSWORD PUZZLE SOLVER - PYTHON MAIN PROGRAM
Beautiful UI with Full Backtracking & Intersection Detection
============================================================================
Run: python crossword_solver.py

Requirements:
- Python 3.x
- turtle (built-in)
- pyswip (optional, for Prolog solver): pip install pyswip
- SWI-Prolog installed (for Prolog solver)

Files needed:
- crossword_solver.pl (Prolog rules - will be auto-created if missing)
- grid.txt (crossword grid)
- word.txt or words.txt (word list)
- constraints.txt (optional pattern constraints)
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
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Check if grid uses pipe separators
                if '|' in content:
                    print("  Grid normalization: Detected pipe-separated format")
                    rows = [line.strip() for line in content.split('|') if line.strip()]
                else:
                    print("  Grid normalization: Using line-per-row format")
                    rows = [line.strip() for line in content.split('\n') if line.strip()]
                
                # Normalize characters: * -> X, # -> X, space -> _
                normalized_rows = []
                for row in rows:
                    normalized = row.replace('*', 'X').replace('#', 'X').replace(' ', '_')
                    normalized_rows.append(list(normalized))
                
                # Validate only X and _ remain
                for i, row in enumerate(normalized_rows):
                    for j, cell in enumerate(row):
                        if cell not in ['X', '_']:
                            raise ValueError(f"Invalid character '{cell}' at row {i}, col {j}. Only 'X', '#' and '_' allowed after normalization.")
                
                # Ensure all rows have same length
                if normalized_rows:
                    max_len = max(len(row) for row in normalized_rows)
                    for i, row in enumerate(normalized_rows):
                        if len(row) < max_len:
                            deficit = max_len - len(row)
                            if deficit > 2:
                                raise ValueError(f"Row {i} has length {len(row)}, expected {max_len} (difference too large)")
                            print(f"  Warning: Padding row {i} with {deficit} 'X' cells")
                            row.extend(['X'] * deficit)
                
                print(f"  Grid normalized: {len(normalized_rows)} rows x {len(normalized_rows[0]) if normalized_rows else 0} cols")
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
        with open("grid.txt", 'w') as f:
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
                if self.grid[row_idx][col] == '_' and (col == 0 or self.grid[row_idx][col-1] == 'X'):
                    start_col = col
                    length = 0
                    # Extend right while empty
                    while col < self.cols and self.grid[row_idx][col] == '_':
                        length += 1
                        col += 1
                    # Record if length >= 2
                    if length >= 2:
                        slots.append({
                            'id': slot_id,
                            'direction': 'across',
                            'row': row_idx,
                            'col': start_col,
                            'length': length
                        })
                        slot_id += 1
                else:
                    col += 1
        
        # Find DOWN slots (vertical)
        for col_idx in range(self.cols):
            row = 0
            while row < self.rows:
                # Start slot if current cell is empty AND (at top edge OR top cell is blocked)
                if self.grid[row][col_idx] == '_' and (row == 0 or self.grid[row-1][col_idx] == 'X'):
                    start_row = row
                    length = 0
                    # Extend down while empty
                    while row < self.rows and self.grid[row][col_idx] == '_':
                        length += 1
                        row += 1
                    # Record if length >= 2
                    if length >= 2:
                        slots.append({
                            'id': slot_id,
                            'direction': 'down',
                            'row': start_row,
                            'col': col_idx,
                            'length': length
                        })
                        slot_id += 1
                else:
                    row += 1
        
        print(f"  Slot detection: Found {len(slots)} slots")
        return slots


class PrologCrosswordSolver:
    """Interface to Prolog solver for constraint satisfaction"""
    
    def __init__(self, prolog_file="crossword_solver.pl"):
        if not PROLOG_AVAILABLE:
            raise ImportError("pyswip not available")
        
        self.prolog = Prolog()
        
        # Check if Prolog file exists, if not create it
        if not os.path.exists(prolog_file):
            print(f"WARNING: {prolog_file} not found!")
            print("Please make sure crossword_solver.pl is in the same directory.")
            raise FileNotFoundError(f"{prolog_file} not found")
        
        try:
            self.prolog.consult(prolog_file)
            print(f"✓ Solver choice: Using Prolog solver from {prolog_file}")
        except Exception as e:
            print(f"ERROR loading Prolog: {e}")
            raise
    
    def load_slots(self, slots):
        """Load slot definitions into Prolog"""
        for slot in slots:
            query = f"assertz(slot({slot['id']}, {slot['direction']}, " \
                   f"{slot['row']}, {slot['col']}, {slot['length']}))"
            list(self.prolog.query(query))
    
    def load_words(self, filename):
        """Load word list from file"""
        # Try word.txt first, then words.txt
        filenames = [filename, 'word.txt', 'words.txt']
        loaded = False
        
        for fname in filenames:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            # Handle apostrophes by replacing with underscore
                            word = word.replace("'", "_")
                            query = f"assertz(word('{word}'))"
                            list(self.prolog.query(query))
                            count += 1
                    print(f"✓ Loaded {count} words from {fname}")
                    loaded = True
                    break
            except FileNotFoundError:
                continue
        
        if not loaded:
            print(f"WARNING: No word file found, creating sample...")
            self.create_sample_words()
            self.load_words('word.txt')
    
    def create_sample_words(self):
        """Create sample word list"""
        words = ["dog", "doctor", "rat"]
        with open("word.txt", 'w') as f:
            f.write('\n'.join(words))
        print("✓ Created sample word.txt")
    
    def load_constraints(self, filename, slots):
        """Load pattern constraints (e.g., dog***, o*o***)"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                count = 0
                mismatches = 0
                for line_idx, line in enumerate(f):
                    line = line.strip().lower()
                    if not line:
                        continue
                    
                    # Support both "slot_id:pattern" and bare "pattern"
                    if ':' in line:
                        parts = line.split(':', 1)
                        try:
                            slot_id = int(parts[0].strip())
                            pattern = parts[1].strip()
                        except ValueError:
                            print(f"  Warning: Invalid slot_id in line {line_idx+1}, skipping")
                            continue
                    else:
                        # Fallback: use line index as slot_id
                        slot_id = line_idx
                        pattern = line
                    
                    # Verify pattern length matches slot length
                    slot_match = None
                    for slot in slots:
                        if slot['id'] == slot_id:
                            slot_match = slot
                            break
                    
                    if slot_match:
                        if len(pattern) != slot_match['length']:
                            print(f"  Warning: Constraint for slot {slot_id} has length {len(pattern)}, but slot length is {slot_match['length']} - ignoring")
                            mismatches += 1
                            continue
                        
                        query = f"assertz(constraint({slot_id}, '{pattern}'))"
                        list(self.prolog.query(query))
                        print(f"  Constraint slot {slot_id}: {pattern}")
                        count += 1
                    else:
                        print(f"  Warning: Constraint references non-existent slot {slot_id}, ignoring")
                
                if count > 0:
                    print(f"✓ Loaded {count} constraints from {filename}")
                if mismatches > 0:
                    print(f"  ({mismatches} constraint(s) ignored due to length mismatch)")
        except FileNotFoundError:
            print(f"  No {filename} found (constraints are optional)")
    
    def solve(self, slot_ids):
        """Solve using Prolog backtracking"""
        list(self.prolog.query("clear_placements"))
        slot_list = str(slot_ids).replace(' ', '')
        query = f"solve_crossword({slot_list}, Solution)"
        
        print(f"  Prolog query: {query}")
        
        try:
            solutions = list(self.prolog.query(query))
            if solutions:
                solution = solutions[0]['Solution']
                parsed = []
                for item in solution:
                    slot_id = int(str(item.args[0]))
                    word = str(item.args[1])
                    parsed.append((slot_id, word))
                return parsed
            return None
        except Exception as e:
            print(f"Prolog error: {e}")
            return None


class PythonCrosswordSolver:
    """Fallback Python-based solver with FULL BACKTRACKING"""
    
    def __init__(self):
        self.words = []
        self.constraints = {}
        self.slots = []
        self.step_callback = None
        print("✓ Solver choice: Using Python solver with full backtracking")
    
    def set_step_callback(self, callback):
        """Set callback function to visualize each step"""
        self.step_callback = callback
    
    def load_slots(self, slots):
        self.slots = slots
    
    def load_words(self, filename):
        # Try word.txt first, then words.txt
        filenames = [filename, 'word.txt', 'words.txt']
        loaded = False
        
        for fname in filenames:
            try:
                with open(fname, 'r') as f:
                    self.words = []
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            # Handle apostrophes by replacing with underscore
                            word = word.replace("'", "_")
                            self.words.append(word)
                    print(f"✓ Loaded {len(self.words)} words from {fname}")
                    loaded = True
                    break
            except FileNotFoundError:
                continue
        
        if not loaded:
            self.words = ["dog", "doctor", "rat"]
            with open("word.txt", 'w') as f:
                f.write('\n'.join(self.words))
            print("✓ Created sample word.txt with default words")
    
    def load_constraints(self, filename, slots):
        try:
            with open(filename, 'r') as f:
                count = 0
                mismatches = 0
                for line_idx, line in enumerate(f):
                    line = line.strip().lower()
                    if not line:
                        continue
                    
                    # Support both "slot_id:pattern" and bare "pattern"
                    if ':' in line:
                        parts = line.split(':', 1)
                        try:
                            slot_id = int(parts[0].strip())
                            pattern = parts[1].strip()
                        except ValueError:
                            print(f"  Warning: Invalid slot_id in line {line_idx+1}, skipping")
                            continue
                    else:
                        # Fallback: use line index as slot_id
                        slot_id = line_idx
                        pattern = line
                    
                    # Verify pattern length matches slot length
                    slot_match = None
                    for slot in slots:
                        if slot['id'] == slot_id:
                            slot_match = slot
                            break
                    
                    if slot_match:
                        if len(pattern) != slot_match['length']:
                            print(f"  Warning: Constraint for slot {slot_id} has length {len(pattern)}, but slot length is {slot_match['length']} - ignoring")
                            mismatches += 1
                            continue
                        
                        self.constraints[slot_id] = pattern
                        print(f"  Constraint slot {slot_id}: {pattern}")
                        count += 1
                    else:
                        print(f"  Warning: Constraint references non-existent slot {slot_id}, ignoring")
                
                if count > 0:
                    print(f"✓ Loaded {count} constraints from {filename}")
                if mismatches > 0:
                    print(f"  ({mismatches} constraint(s) ignored due to length mismatch)")
        except FileNotFoundError:
            print(f"  No {filename} found (constraints are optional)")
    
    def matches_pattern(self, word, pattern):
        """Check if word matches pattern (e.g., dog*** matches doctor)"""
        if len(word) != len(pattern):
            return False
        for w, p in zip(word, pattern):
            if p != '*' and p != w:
                return False
        return True
    
    def get_intersection(self, slot1, slot2):
        """Get intersection point between two slots if it exists"""
        if slot1['direction'] == slot2['direction']:
            return None
            
        if slot1['direction'] == 'across' and slot2['direction'] == 'down':
            # ACROSS: row R1, columns C1 to C1+Len1-1
            # DOWN: column C2, rows R2 to R2+Len2-1
            # Intersection if R1 is in [R2, R2+Len2-1] and C2 is in [C1, C1+Len1-1]
            if (slot2['row'] <= slot1['row'] < slot2['row'] + slot2['length'] and
                slot1['col'] <= slot2['col'] < slot1['col'] + slot1['length']):
                pos1 = slot2['col'] - slot1['col']  # Position in across word
                pos2 = slot1['row'] - slot2['row']  # Position in down word
                return (pos1, pos2)
                
        elif slot1['direction'] == 'down' and slot2['direction'] == 'across':
            # DOWN: column C1, rows R1 to R1+Len1-1
            # ACROSS: row R2, columns C2 to C2+Len2-1
            # Intersection if R2 is in [R1, R1+Len1-1] and C1 is in [C2, C2+Len2-1]
            if (slot1['row'] <= slot2['row'] < slot1['row'] + slot1['length'] and
                slot2['col'] <= slot1['col'] < slot2['col'] + slot2['length']):
                pos1 = slot2['row'] - slot1['row']  # Position in down word
                pos2 = slot1['col'] - slot2['col']  # Position in across word
                return (pos1, pos2)
        
        return None
    
    def check_intersection(self, placements):
        """Check if current placements have valid intersections"""
        slot_ids = list(placements.keys())
        
        for i, slot1_id in enumerate(slot_ids):
            for slot2_id in slot_ids[i+1:]:
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
    
    def solve(self, slot_ids):
        """Backtracking solver with FULL BACKTRACKING (can go back to root)"""
        placements = {}
        used_words = set()
        
        def backtrack(slot_idx):
            if slot_idx >= len(slot_ids):
                return True
            
            slot_id = slot_ids[slot_idx]
            slot = self.slots[slot_id]
            
            for word in self.words:
                # Check word length
                if len(word) != slot['length']:
                    continue
                    
                # Check if word already used
                if word in used_words:
                    continue
                    
                # Check constraint if exists
                if slot_id in self.constraints:
                    if not self.matches_pattern(word, self.constraints[slot_id]):
                        continue
                
                # Try this word
                placements[slot_id] = word
                used_words.add(word)
                
                # Check intersections with ALL previously placed words
                if self.check_intersection(placements):
                    # Visualize this step
                    if self.step_callback:
                        self.step_callback(slot_id, word, slot, True)
                    
                    # Recursively try to fill remaining slots
                    if backtrack(slot_idx + 1):
                        return True
                
                # BACKTRACK: Remove this word and try next
                if self.step_callback:
                    self.step_callback(slot_id, word, slot, False)
                
                del placements[slot_id]
                used_words.remove(word)
            
            # No word worked for this slot, backtrack further
            return False
        
        if backtrack(0):
            return [(sid, placements[sid]) for sid in slot_ids if sid in placements]
        return None


class CrosswordDrawer:
    """Beautiful crossword drawer with modern UI"""
    
    def __init__(self, grid, cell_size=60, animate=True):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.cell_size = cell_size
        self.animate = animate
        self.slots_storage = []
        
        # Color scheme (dark blue-gray theme)
        self.colors = {
            'background': '#2C3E50',
            'cell_bg': '#ECF0F1',
            'cell_border': '#34495E',
            'blocked': '#34495E',
            'letter': '#2C3E50',
            'letter_placing': '#27AE60',
            'letter_backtrack': '#E74C3C',
            'title': '#ECF0F1'
        }
        
        width = max(900, self.cols * cell_size + 200)
        height = max(700, self.rows * cell_size + 250)
        
        self.screen = turtle.Screen()
        self.screen.setup(width=width, height=height)
        self.screen.title("Crossword Puzzle Solver - Full Backtracking")
        self.screen.bgcolor(self.colors['background'])
        
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
        """Draw beautiful crossword grid"""
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2
        
        # Draw outer border (thick)
        self.pen.penup()
        self.pen.goto(start_x - 5, start_y + 5)
        self.pen.pendown()
        self.pen.pensize(8)
        self.pen.color(self.colors['cell_border'])
        for _ in range(4):
            if _ % 2 == 0:
                self.pen.forward(self.cols * self.cell_size + 10)
            else:
                self.pen.forward(self.rows * self.cell_size + 10)
            self.pen.right(90)
        
        # Draw cells
        self.pen.pensize(3)
        for row in range(self.rows):
            for col in range(self.cols):
                x = start_x + col * self.cell_size
                y = start_y - row * self.cell_size
                
                self.pen.penup()
                self.pen.goto(x, y)
                self.pen.pendown()
                
                # Only 'X' is blocked (solid), '_' is empty white cell
                if self.grid[row][col] == 'X':
                    self.pen.fillcolor(self.colors['blocked'])
                    self.pen.color(self.colors['blocked'])
                else:  # '_' is empty white cell
                    self.pen.fillcolor(self.colors['cell_bg'])
                    self.pen.color(self.colors['cell_border'])
                
                self.pen.begin_fill()
                for _ in range(4):
                    self.pen.forward(self.cell_size)
                    self.pen.right(90)
                self.pen.end_fill()
        
        self.screen.update()
    
    def draw_letter(self, x, y, letter, color=None):
        """Draw a letter with beautiful typography"""
        if color is None:
            color = self.colors['letter']
        
        self.pen.penup()
        self.pen.goto(x + self.cell_size / 2, y - self.cell_size * 0.72)
        self.pen.color(color)
        self.pen.write(letter.upper(), align="center", 
                      font=("Arial", int(self.cell_size * 0.55), "bold"))
    
    def clear_cell(self, row, col):
        """Clear a cell by redrawing it based on original grid"""
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2
        x = start_x + col * self.cell_size
        y = start_y - row * self.cell_size
        
        self.pen.penup()
        self.pen.goto(x, y)
        self.pen.pendown()
        self.pen.pensize(3)
        
        # Redraw based on original grid state
        if self.grid[row][col] == 'X':
            self.pen.fillcolor(self.colors['blocked'])
            self.pen.color(self.colors['blocked'])
        else:
            self.pen.fillcolor(self.colors['cell_bg'])
            self.pen.color(self.colors['cell_border'])
        
        self.pen.begin_fill()
        for _ in range(4):
            self.pen.forward(self.cell_size)
            self.pen.right(90)
        self.pen.end_fill()
    
    def update_info(self, message, color=None):
        """Update info message at top"""
        if color is None:
            color = self.colors['title']
        
        self.info.clear()
        self.info.goto(0, (self.rows * self.cell_size) / 2 + 70)
        self.info.color(color)
        self.info.write(message, align="center",
                       font=("Arial", 18, "bold"))
    
    def animate_word_placement(self, slot_id, word, slot, is_placing):
        """Animate placing or removing a word"""
        start_x = -(self.cols * self.cell_size) / 2
        start_y = (self.rows * self.cell_size) / 2
        
        row, col = slot['row'], slot['col']
        direction = slot['direction']
        
        if is_placing:
            self.update_info(f"Trying: {word.upper()} ({direction}) at slot {slot_id}", 
                           self.colors['letter_placing'])
            
            for i, letter in enumerate(word):
                if direction == 'across':
                    x = start_x + (col + i) * self.cell_size
                    y = start_y - row * self.cell_size
                else:
                    x = start_x + col * self.cell_size
                    y = start_y - (row + i) * self.cell_size
                
                self.draw_letter(x, y, letter, self.colors['letter_placing'])
                self.screen.update()
                time.sleep(0.08)
        else:
            self.update_info(f"Backtracking: {word.upper()} from slot {slot_id}", 
                           self.colors['letter_backtrack'])
            
            for i, letter in enumerate(word):
                if direction == 'across':
                    r, c = row, col + i
                else:
                    r, c = row + i, col
                
                self.clear_cell(r, c)
                self.screen.update()
                time.sleep(0.04)
    
    def draw_solution(self, slots, solution, animated=False):
        """Draw the complete solved crossword"""
        self.slots_storage = slots
        self.draw_grid()
        
        if not solution:
            self.update_info("NO SOLUTION FOUND", self.colors['letter_backtrack'])
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
            row, col = slot['row'], slot['col']
            direction = slot['direction']
            
            if animated:
                self.update_info(f"Final: {word.upper()} ({direction})", 
                               self.colors['letter'])
                time.sleep(0.2)
            
            for i, letter in enumerate(word):
                if direction == 'across':
                    x = start_x + (col + i) * self.cell_size
                    y = start_y - row * self.cell_size
                else:
                    x = start_x + col * self.cell_size
                    y = start_y - (row + i) * self.cell_size
                
                self.draw_letter(x, y, letter, self.colors['letter'])
        
        self.update_info("✓ PUZZLE SOLVED!", self.colors['letter_placing'])
        self.screen.update()
        print("\n✓ Close window to exit")
        turtle.done()


def print_ascii_grid(grid, solution, slots):
    """Print the crossword grid in ASCII format with letters placed correctly"""
    # Create a copy of the grid for filling
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    filled_grid = [row[:] for row in grid]  # Deep copy
    
    # Place all solution words into the grid
    for slot_id, word in solution:
        slot = slots[slot_id]
        row, col = slot['row'], slot['col']
        direction = slot['direction']
        
        for i, letter in enumerate(word):
            if direction == 'across':
                filled_grid[row][col + i] = letter.upper()
            else:  # down
                filled_grid[row + i][col] = letter.upper()
    
    # Print the grid in ASCII format
    print("\n" + "=" * (cols * 4 + 1))
    print("SOLVED CROSSWORD GRID (ASCII)")
    print("=" * (cols * 4 + 1))
    
    for row in filled_grid:
        line = "|"
        for cell in row:
            if cell == 'X':
                line += "###|"
            elif cell == '_':
                line += "   |"
            else:
                line += f" {cell} |"
        print(line)
        print("-" * (cols * 4 + 1))
    
    print()


def main():
    """Main program entry point"""
    print("=" * 70)
    print("   CROSSWORD PUZZLE SOLVER")
    print("   Full Backtracking + Intersection Detection")
    print("=" * 70)
    
    # Read grid
    print("\n[1] Reading grid...")
    grid_reader = GridReader("grid.txt")
    print(f"    Grid: {grid_reader.rows} x {grid_reader.cols}")
    
    # Find slots
    print("\n[2] Finding slots...")
    slots = grid_reader.find_slots()
    print(f"    Found {len(slots)} slots:")
    for slot in slots:
        print(f"      Slot {slot['id']}: {slot['direction']:6} at ({slot['row']},{slot['col']}) len={slot['length']}")
    
    # Create drawer
    print("\n[3] Initializing graphics...")
    drawer = CrosswordDrawer(grid_reader.grid, cell_size=60, animate=True)
    drawer.draw_grid()
    
    # Choose solver
    print("\n[4] Initializing solver...")
    use_prolog = False
    if PROLOG_AVAILABLE:
        try:
            solver = PrologCrosswordSolver("crossword_solver.pl")
            use_prolog = True
        except:
            print("    Falling back to Python solver...")
            solver = PythonCrosswordSolver()
    else:
        solver = PythonCrosswordSolver()
    
    # Set up step callback for Python solver
    if not use_prolog:
        solver.set_step_callback(drawer.animate_word_placement)
    
    # Load data
    print("\n[5] Loading data...")
    solver.load_slots(slots)
    solver.load_words("word.txt")
    solver.load_constraints("constraints.txt", slots)
    
    # Solve
    print("\n[6] Solving with full backtracking...")
    print("    Watch the window to see backtracking in action!")
    slot_ids = [s['id'] for s in slots]
    solution = solver.solve(slot_ids)
    
    if solution:
        print(f"\n✓ Solution found!")
        print("\nFinal solution:")
        for slot_id, word in solution:
            slot = slots[slot_id]
            print(f"  Slot {slot_id} ({slot['direction']:6} at {slot['row']},{slot['col']}): {word.upper()}")
        
        # Print ASCII grid with solution
        print_ascii_grid(grid_reader.grid, solution, slots)
        
        # Always call draw_solution with animation
        drawer.draw_solution(slots, solution, animated=True)
    else:
        print("\n✗ No solution found")
        drawer.draw_solution(slots, None)


if __name__ == "__main__":
    main()