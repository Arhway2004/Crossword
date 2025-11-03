"""
Crossword Puzzle Solver with Backtracking and Turtle Visualization
Author: AI Assistant
Description: Reads grid format, word lists, and constraints, then solves
             using backtracking algorithm and visualizes with Turtle graphics.
"""

import turtle
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Word direction in crossword"""
    ACROSS = "across"
    DOWN = "down"


@dataclass
class Slot:
    """Represents a word slot in the crossword grid"""
    number: int  # Clue number
    row: int  # Starting row
    col: int  # Starting column
    length: int  # Word length
    direction: Direction  # Across or Down
    constraint: Optional[str] = None  # Constraint pattern (e.g., "o*o***")
    
    def __repr__(self):
        return f"Slot({self.number}, {self.direction.value}, r{self.row}c{self.col}, len={self.length})"


class GridParser:
    """Parses the crossword grid from file"""
    
    def __init__(self):
        self.grid = []
        self.rows = 0
        self.cols = 0
    
    def parse_grid(self, filename: str) -> List[List[str]]:
        """
        Parse grid file and convert to 2D array.
        X = black cell, _ = empty cell
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            self.grid = []
            for line in lines:
                line = line.strip()
                if line:
                    # Remove any | separators and convert to list
                    row = [c for c in line if c in ['X', '_']]
                    if row:
                        self.grid.append(row)
            
            self.rows = len(self.grid)
            self.cols = len(self.grid[0]) if self.rows > 0 else 0
            
            print(f"Grid loaded: {self.rows} x {self.cols}")
            return self.grid
            
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return []
    
    def find_word_slots(self) -> List[Slot]:
        """
        Find all horizontal and vertical word slots.
        A slot is a sequence of 2+ consecutive empty cells.
        """
        slots = []
        slot_number = 1
        numbered_cells = {}  # Track which cells get numbers
        
        # First pass: identify cells that start words
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == '_':
                    starts_across = (c == 0 or self.grid[r][c-1] == 'X') and \
                                   (c < self.cols - 1 and self.grid[r][c+1] == '_')
                    starts_down = (r == 0 or self.grid[r-1][c] == 'X') and \
                                 (r < self.rows - 1 and self.grid[r+1][c] == '_')
                    
                    if starts_across or starts_down:
                        if (r, c) not in numbered_cells:
                            numbered_cells[(r, c)] = slot_number
                            slot_number += 1
        
        # Second pass: create slots
        # Find across slots
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if self.grid[r][c] == '_':
                    # Start of potential word
                    start_col = c
                    length = 0
                    while c < self.cols and self.grid[r][c] == '_':
                        length += 1
                        c += 1
                    
                    if length >= 2:  # Valid word slot
                        number = numbered_cells.get((r, start_col), 0)
                        slots.append(Slot(number, r, start_col, length, Direction.ACROSS))
                else:
                    c += 1
        
        # Find down slots
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if self.grid[r][c] == '_':
                    # Start of potential word
                    start_row = r
                    length = 0
                    while r < self.rows and self.grid[r][c] == '_':
                        length += 1
                        r += 1
                    
                    if length >= 2:  # Valid word slot
                        number = numbered_cells.get((start_row, c), 0)
                        slots.append(Slot(number, start_row, c, length, Direction.DOWN))
                else:
                    r += 1
        
        # Sort by number, then direction
        slots.sort(key=lambda s: (s.number, s.direction.value))
        
        print(f"Found {len(slots)} word slots:")
        for slot in slots:
            print(f"  {slot}")
        
        return slots


class ConstraintParser:
    """Parses word constraints from file"""
    
    def parse_constraints(self, filename: str) -> List[str]:
        """
        Parse constraints file.
        Each line represents a constraint pattern:
        - Letters = fixed positions
        - * = any letter
        Example: "o*o***" means length 6, 'o' at pos 0 and 2
        """
        try:
            with open(filename, 'r') as f:
                constraints = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(constraints)} constraints")
            return constraints
            
        except FileNotFoundError:
            print(f"Error: {filename} not found. Using no constraints.")
            return []


class CrosswordSolver:
    """Solves crossword puzzle using backtracking"""
    
    def __init__(self, grid: List[List[str]], slots: List[Slot], 
                 words: List[str], constraints: List[str]):
        self.grid = [row[:] for row in grid]  # Deep copy
        self.slots = slots
        self.words = words
        self.constraints = constraints
        self.solution = {}  # slot_index -> word
        self.used_words = set()  # Track used words
        self.attempts = 0
        self.backtracks = 0
        
        # Apply constraints to slots
        self._apply_constraints()
        
        # Create word dictionary by length for faster lookup
        self.words_by_length = {}
        for word in words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word.upper())
    
    def _apply_constraints(self):
        """Apply constraint patterns to corresponding slots"""
        for i, slot in enumerate(self.slots):
            if i < len(self.constraints):
                slot.constraint = self.constraints[i]
                print(f"Constraint for {slot}: {slot.constraint}")
    
    def solve(self) -> Optional[Dict[int, str]]:
        """
        Main solving method using backtracking.
        Returns solution dictionary or None if no solution.
        """
        print("\nStarting backtracking solver...")
        
        # Sort slots by constraint complexity (most constrained first)
        # This improves efficiency (MRV heuristic)
        sorted_slots = sorted(
            enumerate(self.slots),
            key=lambda x: self._count_fixed_letters(x[1].constraint) if x[1].constraint else 0,
            reverse=True
        )
        
        if self._backtrack(0, sorted_slots):
            print(f"\nSolution found! Attempts: {self.attempts}, Backtracks: {self.backtracks}")
            return self.solution
        else:
            print(f"\nNo solution exists. Attempts: {self.attempts}, Backtracks: {self.backtracks}")
            return None
    
    def _count_fixed_letters(self, constraint: Optional[str]) -> int:
        """Count fixed letters in constraint"""
        if not constraint:
            return 0
        return sum(1 for c in constraint if c != '*')
    
    def _backtrack(self, slot_idx: int, sorted_slots: List[Tuple[int, Slot]]) -> bool:
        """
        Recursive backtracking algorithm.
        Try to fill each slot with valid words.
        """
        # Base case: all slots filled
        if slot_idx >= len(sorted_slots):
            return True
        
        original_idx, slot = sorted_slots[slot_idx]
        
        # Get candidate words for this slot
        candidates = self._get_candidates(slot)
        
        for word in candidates:
            self.attempts += 1
            
            if word in self.used_words:
                continue
            
            if self._is_valid_placement(word, slot, original_idx):
                # Place word
                self._place_word(word, slot)
                self.solution[original_idx] = word
                self.used_words.add(word)
                
                # Recurse
                if self._backtrack(slot_idx + 1, sorted_slots):
                    return True
                
                # Backtrack
                self.backtracks += 1
                self._remove_word(slot)
                del self.solution[original_idx]
                self.used_words.remove(word)
        
        return False
    
    def _get_candidates(self, slot: Slot) -> List[str]:
        """Get candidate words for a slot based on length and constraint"""
        candidates = self.words_by_length.get(slot.length, [])
        
        if slot.constraint:
            # Filter by constraint
            filtered = []
            for word in candidates:
                if self._matches_constraint(word, slot.constraint):
                    filtered.append(word)
            return filtered
        
        return candidates
    
    def _matches_constraint(self, word: str, constraint: str) -> bool:
        """Check if word matches constraint pattern"""
        if len(word) != len(constraint):
            return False
        
        for i, c in enumerate(constraint):
            if c != '*' and c.upper() != word[i].upper():
                return False
        
        return True
    
    def _is_valid_placement(self, word: str, slot: Slot, slot_idx: int) -> bool:
        """
        Check if word can be validly placed in slot.
        Checks length, constraints, and intersections.
        """
        if len(word) != slot.length:
            return False
        
        # Check constraint match
        if slot.constraint and not self._matches_constraint(word, slot.constraint):
            return False
        
        # Check intersections with already placed words
        for other_idx, other_word in self.solution.items():
            other_slot = self.slots[other_idx]
            
            # Check if slots intersect
            intersection = self._find_intersection(slot, other_slot)
            if intersection:
                word_pos, other_pos = intersection
                if word[word_pos].upper() != other_word[other_pos].upper():
                    return False
        
        return True
    
    def _find_intersection(self, slot1: Slot, slot2: Slot) -> Optional[Tuple[int, int]]:
        """
        Find intersection point between two slots.
        Returns (pos_in_slot1, pos_in_slot2) or None.
        """
        if slot1.direction == slot2.direction:
            return None  # Parallel slots don't intersect
        
        if slot1.direction == Direction.ACROSS:
            across, down = slot1, slot2
        else:
            across, down = slot2, slot1
        
        # Check if they intersect
        if (down.col >= across.col and 
            down.col < across.col + across.length and
            across.row >= down.row and 
            across.row < down.row + down.length):
            
            across_pos = down.col - across.col
            down_pos = across.row - down.row
            
            if slot1.direction == Direction.ACROSS:
                return (across_pos, down_pos)
            else:
                return (down_pos, across_pos)
        
        return None
    
    def _place_word(self, word: str, slot: Slot):
        """Place word in grid"""
        for i, letter in enumerate(word):
            if slot.direction == Direction.ACROSS:
                self.grid[slot.row][slot.col + i] = letter.upper()
            else:
                self.grid[slot.row + i][slot.col] = letter.upper()
    
    def _remove_word(self, slot: Slot):
        """Remove word from grid (restore to empty)"""
        for i in range(slot.length):
            if slot.direction == Direction.ACROSS:
                self.grid[slot.row][slot.col + i] = '_'
            else:
                self.grid[slot.row + i][slot.col] = '_'


class CrosswordVisualizer:
    """Visualizes crossword solution using Turtle graphics"""
    
    def __init__(self, grid: List[List[str]], slots: List[Slot]):
        self.grid = grid
        self.slots = slots
        self.cell_size = 40
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
        # Setup turtle
        self.screen = turtle.Screen()
        self.screen.setup(width=self.cols * self.cell_size + 100, 
                         height=self.rows * self.cell_size + 100)
        self.screen.title("Crossword Puzzle Solution")
        self.screen.bgcolor("white")
        
        self.pen = turtle.Turtle()
        self.pen.speed(0)
        self.pen.hideturtle()
    
    def draw_grid(self):
        """Draw complete crossword grid with solution"""
        # Calculate starting position (center the grid)
        start_x = -(self.cols * self.cell_size) // 2
        start_y = (self.rows * self.cell_size) // 2
        
        # Create number dictionary for cells
        cell_numbers = {}
        for slot in self.slots:
            if slot.number > 0:
                key = (slot.row, slot.col)
                if key not in cell_numbers:
                    cell_numbers[key] = slot.number
        
        # Draw each cell
        for r in range(self.rows):
            for c in range(self.cols):
                x = start_x + c * self.cell_size
                y = start_y - r * self.cell_size
                
                is_black = self.grid[r][c] == 'X'
                self._draw_cell(x, y, is_black)
                
                if not is_black:
                    # Draw cell number if exists
                    if (r, c) in cell_numbers:
                        self._draw_number(x, y, cell_numbers[(r, c)])
                    
                    # Draw letter if not empty
                    if self.grid[r][c] != '_':
                        self._draw_letter(x, y, self.grid[r][c])
        
        self.screen.update()
        print("\nVisualization complete! Click to close.")
        self.screen.exitonclick()
    
    def _draw_cell(self, x: float, y: float, is_black: bool):
        """Draw individual cell"""
        self.pen.penup()
        self.pen.goto(x, y)
        self.pen.pendown()
        
        if is_black:
            self.pen.fillcolor("black")
        else:
            self.pen.fillcolor("white")
        
        self.pen.begin_fill()
        for _ in range(4):
            self.pen.forward(self.cell_size)
            self.pen.right(90)
        self.pen.end_fill()
    
    def _draw_letter(self, x: float, y: float, letter: str):
        """Draw letter centered in cell"""
        self.pen.penup()
        self.pen.goto(x + self.cell_size // 2, y - self.cell_size * 0.7)
        self.pen.color("black")
        self.pen.write(letter, align="center", font=("Arial", 16, "bold"))
    
    def _draw_number(self, x: float, y: float, number: int):
        """Draw clue number in top-left corner"""
        self.pen.penup()
        self.pen.goto(x + 3, y - 12)
        self.pen.color("blue")
        self.pen.write(str(number), align="left", font=("Arial", 8, "normal"))


def create_sample_files():
    """Create sample input files for testing"""
    
    # Sample grid
    grid_content = """____X___
________
__X_____
________"""
    
    with open("grid.txt", "w") as f:
        f.write(grid_content)
    
    # Sample words
    words_content = """code
test
exit
open
note
date
rest
rode
"""
    
    with open("words.txt", "w") as f:
        f.write(words_content)
    
    # Sample constraints (one per slot)
    constraints_content = """****
****
****
****
****
****
"""
    
    with open("constraints.txt", "w") as f:
        f.write(constraints_content)
    
    print("Sample files created: grid.txt, words.txt, constraints.txt")


def main():
    """Main program"""
    print("=" * 60)
    print("CROSSWORD PUZZLE SOLVER WITH BACKTRACKING")
    print("=" * 60)
    
    # Create sample files if they don't exist
    try:
        with open("grid.txt"):
            pass
    except FileNotFoundError:
        print("\nSample files not found. Creating them...")
        create_sample_files()
        print()
    
    # Parse grid
    grid_parser = GridParser()
    grid = grid_parser.parse_grid("grid.txt")
    if not grid:
        print("Failed to load grid!")
        return
    
    slots = grid_parser.find_word_slots()
    if not slots:
        print("No word slots found!")
        return
    
    # Parse constraints
    constraint_parser = ConstraintParser()
    constraints = constraint_parser.parse_constraints("constraints.txt")
    
    # Load words
    try:
        with open("words.txt", 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(words)} words")
    except FileNotFoundError:
        print("Error: words.txt not found")
        return
    
    # Solve puzzle
    solver = CrosswordSolver(grid, slots, words, constraints)
    solution = solver.solve()
    
    if solution:
        print("\n" + "=" * 60)
        print("SOLUTION FOUND!")
        print("=" * 60)
        for idx, word in solution.items():
            print(f"{slots[idx].direction.value.upper():>6} {slots[idx].number:>2}: {word}")
        
        print("\nSolved Grid:")
        for row in solver.grid:
            print(" ".join(row))
        
        # Visualize with turtle
        print("\nLaunching visualization...")
        visualizer = CrosswordVisualizer(solver.grid, slots)
        visualizer.draw_grid()
    else:
        print("\nNo solution exists with the given words and constraints.")


if __name__ == "__main__":
    main()