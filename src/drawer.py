import turtle
import time


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
