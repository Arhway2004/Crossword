"""
============================================================================
CROSSWORD SOLVER - SIMPLIFIED UI
Upload Image → Input Words → Solve with Visualization
============================================================================
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import time

# Import modules
try:
    from src.ocr import CrosswordExtractor

    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    print("Warning: OCR.py not found")

try:
    from src.crossword_solver import PrologCrosswordSolver, GridReader, CrosswordDrawer

    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    print("Warning: crossword_solver.py not found")


class StepViewer:
    """Play/Pause/Step viewer for Prolog trace"""

    def __init__(self, parent, grid, slots, steps, solution):
        self.parent = parent
        self.grid0 = [row[:] for row in grid]
        self.slots = slots
        self.steps = steps or []
        self.solution = solution or []
        self.idx = -1
        self.timer = None
        self.playing = False

        self.win = tk.Toplevel(parent)
        self.win.title("Solver Steps")
        self.win.geometry("900x700")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=10)

        self.btn_play = tk.Button(top, text="▶ Play", command=self.play)
        self.btn_pause = tk.Button(top, text="⏸ Pause", command=self.pause, state=tk.DISABLED)
        self.btn_step = tk.Button(top, text="Step ▶", command=self.step)
        self.btn_reset = tk.Button(top, text="⟲ Reset", command=self.reset)

        self.btn_play.pack(side=tk.LEFT, padx=5)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        self.btn_step.pack(side=tk.LEFT, padx=5)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(top, text="Ready")
        self.status.pack(side=tk.RIGHT, padx=5)

        mid = tk.Frame(self.win)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Canvas grid
        left = tk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cell_size = 36
        self.margin = 10
        self.step_colors = {
            "start": "#0078D4",
            "select_slot": "#2563eb",
            "try": "#64748b",
            "place": "#16a34a",
            "fail_forward": "#dc2626",
            "backtrack": "#e11d48",
            "solution": "#b45309",
        }
        rows = len(self.grid0)
        cols = len(self.grid0[0]) if rows else 0
        cw = self.margin * 2 + cols * self.cell_size
        ch = self.margin * 2 + rows * self.cell_size
        self.canvas = tk.Canvas(left, width=cw, height=ch, bg="#ffffff")
        self.canvas.pack(fill=tk.BOTH, expand=False)

        # Info panel under canvas
        info = tk.Frame(left)
        info.pack(fill=tk.X, padx=4, pady=(8, 0))
        self.info_label = tk.Label(info, text="", anchor="w", justify="left")
        self.info_label.pack(fill=tk.X)

        # Right: step list
        right = tk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)

        # Legend
        legend = tk.Frame(right)
        legend.pack(fill=tk.X, pady=(0, 6))
        tk.Label(legend, text="Legend:").pack(anchor="w")
        for key in ["select_slot", "try", "place", "fail_forward", "backtrack", "solution"]:
            row = tk.Frame(legend)
            row.pack(anchor="w")
            sw = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            sw.create_rectangle(1, 1, 13, 13, fill=self.step_colors.get(key, "#888"), outline="")
            sw.pack(side=tk.LEFT)
            tk.Label(row, text=f" {key}").pack(side=tk.LEFT)

        tk.Label(right, text="Steps").pack()
        self.listbox = tk.Listbox(right, width=40, height=28)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        for s in self.steps:
            txt = self._fmt_step(s)
            self.listbox.insert(tk.END, txt)
            # Colorize by type if supported
            try:
                idx = self.listbox.size() - 1
                color = self.step_colors.get(s.get("type"), "#111")
                self.listbox.itemconfig(idx, fg=color)
            except Exception:
                pass

        # If no steps, make it clear and disable play
        if not self.steps:
            self.status.config(text="No steps to show")
            self.btn_play.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.DISABLED)

        self.render()

        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.pause()
        self.win.destroy()

    def _fmt_step(self, s):
        t = s.get("type")
        sid = s.get("slot")
        w = s.get("word")
        w = (w.upper() if isinstance(w, str) else "")
        i = s.get("i", 0)
        if t == "select_slot":
            return f"{i:04d}  SELECT slot {sid}"
        if t == "try":
            return f"{i:04d}  TRY slot {sid}: {w}"
        if t == "place":
            return f"{i:04d}  PLACE slot {sid}: {w}"
        if t == "backtrack":
            return f"{i:04d}  BACKTRACK slot {sid}: {w}"
        if t == "fail_forward":
            return f"{i:04d}  FORWARD-FAIL slot {sid}: {w}"
        if t == "solution":
            return f"{i:04d}  SOLUTION"
        if t == "start":
            return f"{i:04d}  START"
        return f"{i:04d}  {t} slot {sid}: {w}"

    def play(self):
        if self.playing:
            return
        self.playing = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self._tick()

    def pause(self):
        self.playing = False
        self.btn_play.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        if self.timer:
            try:
                self.win.after_cancel(self.timer)
            except Exception:
                pass
            self.timer = None

    def reset(self):
        self.pause()
        self.idx = -1
        self.render()

    def step(self):
        if self.idx + 1 < len(self.steps):
            self.idx += 1
            self.render()

    def _tick(self):
        if not self.playing:
            return
        if self.idx + 1 < len(self.steps):
            self.idx += 1
            self.render()
            self.timer = self.win.after(300, self._tick)  # 0.3s per step
        else:
            self.pause()

    def render(self):
        # Highlight current step
        self.listbox.selection_clear(0, tk.END)
        if 0 <= self.idx < len(self.steps):
            self.listbox.selection_set(self.idx)
            self.listbox.see(self.idx)
            cur = self.steps[self.idx]
            self.status.config(text=self._fmt_step(cur))
            # Update info pane
            sid = cur.get("slot")
            slot = None
            if isinstance(sid, int) and 0 <= sid < len(self.slots):
                slot = self.slots[sid]
            direction = slot["direction"] if slot else "-"
            length = slot["length"] if slot else "-"
            word = cur.get("word") or ""
            note = cur.get("note") or ""
            self.info_label.config(
                text=f"Step: {cur.get('type')}\nSlot: {sid}  Dir: {direction}  Len: {length}\nWord: {word.upper()}  {note}"
            )
        else:
            self.status.config(text="Ready")
            self.info_label.config(text="")

        # Rebuild placement map by replaying steps up to idx
        placements = {}
        for k in range(0, self.idx + 1):
            s = self.steps[k]
            sid = s.get("slot")
            w = s.get("word")
            if s.get("type") == "place" and w:
                placements[sid] = w
            elif s.get("type") in ("backtrack", "fail_forward") and w and sid in placements:
                placements.pop(sid, None)

        # Draw canvas grid with current placements
        self._draw_canvas(placements)

    def _draw_canvas(self, placements):
        self.canvas.delete("all")
        rows = len(self.grid0)
        cols = len(self.grid0[0]) if rows else 0
        cs = self.cell_size
        m = self.margin

        # Determine current slot for highlight
        cur_slot = None
        if 0 <= self.idx < len(self.steps):
            cur_slot = self.steps[self.idx].get("slot")

        # Precompute cells for each slot for drawing slot numbers and highlights
        def slot_cells(slot):
            cells = []
            r, c = slot["row"], slot["col"]
            L = slot["length"]
            if slot["direction"] == "across":
                for i in range(L):
                    cells.append((r, c + i))
            else:
                for i in range(L):
                    cells.append((r + i, c))
            return cells

        # Map for quick slot->cells
        slot_to_cells = {s["id"]: slot_cells(s) for s in self.slots}

        # Background and cells
        for r in range(rows):
            for c in range(cols):
                x0 = m + c * cs
                y0 = m + r * cs
                x1 = x0 + cs
                y1 = y0 + cs
                is_block = (self.grid0[r][c] == "X")

                fill = "#000000" if is_block else "#ffffff"
                outline = "#C0C0C0"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=1)

        # Draw placed letters
        for sid, word in placements.items():
            if sid not in slot_to_cells:
                continue
            cells = slot_to_cells[sid]
            for i, ch in enumerate((word or "").upper()):
                if i >= len(cells):
                    break
                r, c = cells[i]
                if self.grid0[r][c] == "X":
                    continue
                x = m + c * cs + cs // 2
                y = m + r * cs + cs // 2
                self.canvas.create_text(x, y, text=ch, font=("Helvetica", int(cs*0.45), "bold"), fill="#1f2a44")

        # Slot numbers at starting cells
        for slot in self.slots:
            sid = slot["id"]
            r, c = slot["row"], slot["col"]
            if 0 <= r < rows and 0 <= c < cols and self.grid0[r][c] != "X":
                x = m + c * cs + 4
                y = m + r * cs + 4
                self.canvas.create_text(x, y, text=str(sid), anchor="nw", font=("Helvetica", int(cs*0.28), "bold"), fill="#888888")

        # Highlight current slot cells (if any) using step color
        if isinstance(cur_slot, int) and cur_slot in slot_to_cells:
            step_type = None
            if 0 <= self.idx < len(self.steps):
                step_type = self.steps[self.idx].get("type")
            hl = self.step_colors.get(step_type, "#ff6a00")
            for (r, c) in slot_to_cells[cur_slot]:
                x0 = m + c * cs
                y0 = m + r * cs
                x1 = x0 + cs
                y1 = y0 + cs
                self.canvas.create_rectangle(x0+1, y0+1, x1-1, y1-1, outline=hl, width=2)


class SimpleCrosswordUI:
    """Simplified 3-step crossword solver UI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crossword Solver")
        self.root.geometry("1000x800")
        self.root.configure(bg="#AAAAAA")

        # State
        self.image_path = None
        self.grid_extracted = False
        self.words_ready = False
        self.drawer = None

        # Colors
        self.colors = {
            "bg": "#AAAAAA",
            "fg": "#000000",
            "button": "#3498DB",
            "button_hover": "#000000",
            "success": "#27AE60",
            "error": "#E74C3C",
            "warning": "#F39C12",
            "panel_bg": "#000000",
        }

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI"""
        # Title
        title = tk.Label(
            self.root,
            text="🧩 CROSSWORD SOLVER",
            font=("Arial", 28, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        )
        title.pack(pady=20)

        # Main container
        container = tk.Frame(self.root, bg=self.colors["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Left panel - Actions
        left_panel = tk.Frame(
            container, bg=self.colors["panel_bg"], relief=tk.RAISED, bd=3
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))

        tk.Label(
            left_panel,
            text="ACTIONS",
            font=("Arial", 16, "bold"),
            bg=self.colors["panel_bg"],
            fg=self.colors["fg"],
        ).pack(pady=20)

        # Step 1: Upload & Extract
        self.btn_upload = self.create_big_button(
            left_panel,
            "UPLOAD IMAGE\n& EXTRACT GRID",
            self.upload_and_extract,
            self.colors["panel_bg"],
        )
        self.btn_upload.pack(padx=20, pady=15)

        tk.Label(
            left_panel,
            text="↓",
            font=("Arial", 20, "bold"),
            bg=self.colors["panel_bg"],
            fg=self.colors["warning"],
        ).pack()

        # Step 2: Input Words
        self.btn_words = self.create_big_button(
            left_panel,
            "INPUT WORDS",
            self.open_word_input,
            self.colors["panel_bg"],
            state=tk.DISABLED,
        )
        self.btn_words.pack(padx=20, pady=15)

        tk.Label(
            left_panel,
            text="↓",
            font=("Arial", 20, "bold"),
            bg=self.colors["panel_bg"],
            fg=self.colors["warning"],
        ).pack()

        # Step 3: Solve & Visualize
        self.btn_solve = self.create_big_button(
            left_panel,
            "SOLVE &\nVISUALIZE",
            self.solve_and_visualize,
            self.colors["panel_bg"],
            state=tk.DISABLED,
        )
        self.btn_solve.pack(padx=20, pady=15)

        # Reset button
        tk.Label(left_panel, text="", bg=self.colors["panel_bg"], height=2).pack()
        tk.Button(
            left_panel,
            text="RESET",
            command=self.reset_all,
            font=("Arial", 11, "bold"),
            bg=self.colors["error"],
            fg="red",
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
        ).pack(padx=20, pady=10)

        # Right panel - Output
        right_panel = tk.Frame(
            container, bg=self.colors["panel_bg"], relief=tk.RAISED, bd=3
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_panel,
            text="OUTPUT LOG",
            font=("Arial", 16, "bold"),
            bg=self.colors["panel_bg"],
            fg=self.colors["fg"],
        ).pack(pady=20)

        # Output text area
        self.output = scrolledtext.ScrolledText(
            right_panel,
            font=("Courier", 10),
            bg="#ECF0F1",
            fg="#2C3E50",
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD,
        )
        self.output.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Initial message
        self.log("Welcome to Crossword Solver!\n\n")
        self.log("Simple 3-step process:\n")
        self.log("1. Upload crossword image (auto-extracts grid)\n")
        self.log("2. Input your word list\n")
        self.log("3. Solve with step-by-step visualization\n\n")
        self.log("Click 'UPLOAD IMAGE & EXTRACT GRID' to begin.\n")

        # Status bar
        self.status = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 11, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            pady=10,
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def create_big_button(self, parent, text, command, bg, state=tk.NORMAL):
        """Create a large styled button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 14, "bold"),
            bg=bg,
            fg="#000000",
            activebackground=self.colors["button_hover"],
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=25,
            cursor="hand2" if state == tk.NORMAL else "arrow",
            state=state,
            width=18,
            height=3,
        )
        return btn

    def lock_button(self, btn):
        """Disable a button to prevent further clicks until reset."""
        try:
            btn.config(state=tk.DISABLED, cursor="arrow")
        except Exception:
            # Fail-safe: ignore if widget is already destroyed or unavailable
            pass

    # ========================================================================
    # STEP 1: UPLOAD & EXTRACT
    # ========================================================================

    def upload_and_extract(self):
        """Upload image and extract grid automatically"""
        self.clear_output()
        self.log("═" * 60 + "\n")
        self.log("STEP 1: UPLOAD IMAGE & EXTRACT GRID\n", "title")
        self.log("═" * 60 + "\n\n")

        # File dialog
        file_path = filedialog.askopenfilename(
            title="Select Crossword Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            self.log("✗ No image selected\n", "error")
            self.update_status("No image selected", "error")
            return

        self.image_path = file_path
        self.log(f"📂 Image: {os.path.basename(file_path)}\n\n")

        # Check OCR module
        if not EXTRACTOR_AVAILABLE:
            self.log("✗ ERROR: OCR.py not found!\n", "error")
            self.log("Please ensure OCR.py is in the same folder.\n")
            self.update_status("OCR module missing", "error")
            return

        # Extract grid
        self.log("🔄 Processing image with OCR...\n")
        self.update_status("Extracting grid...", "warning")
        self.root.update()

        try:
            extractor = CrosswordExtractor(
                rows=None, cols=None, image_path=file_path, debug=True
            )

            self.log("   Detecting grid region...\n")
            grid = extractor.extract_grid()

            self.log("   Analyzing cells...\n")
            extractor.save_grid("grid.txt")

            self.log(
                f"\nSUCCESS! Grid extracted: {len(grid)}×{len(grid[0])}\n\n",
                "success",
            )

            # Display grid
            self.log("Extracted Grid:\n")
            self.log("─" * 50 + "\n")
            for row in grid:
                self.log("".join(row) + "\n")
            self.log("─" * 50 + "\n\n")

            self.log("Grid saved to: grid.txt\n\n")
            self.log("✓ Ready for next step!\n", "success")

            # Enable next button
            self.grid_extracted = True
            self.btn_words.config(bg=self.colors["button"], state=tk.NORMAL)
            self.btn_upload.config(bg=self.colors["success"])
            # Lock this step to prevent re-clicks until reset
            self.lock_button(self.btn_upload)
            self.update_status("Grid extracted successfully ✓", "success")

        except Exception as e:
            self.log(f"\n✗ ERROR: {str(e)}\n", "error")
            self.update_status("Extraction failed", "error")

    # ========================================================================
    # STEP 2: INPUT WORDS
    # ========================================================================

    def open_word_input(self):
        """Open word input dialog"""
        self.clear_output()
        self.log("═" * 60 + "\n")
        self.log("STEP 2: INPUT WORDS\n", "title")
        self.log("═" * 60 + "\n\n")

        # Create input window
        word_window = tk.Toplevel(self.root)
        word_window.title("Input Words")
        word_window.geometry("600x700")
        word_window.configure(bg=self.colors["bg"])
        word_window.transient(self.root)
        word_window.grab_set()

        tk.Label(
            word_window,
            text="📝 Enter Your Word List",
            font=("Arial", 18, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        ).pack(pady=20)

        tk.Label(
            word_window,
            text="Enter one word per line (case insensitive)",
            font=("Arial", 11),
            bg=self.colors["bg"],
            fg=self.colors["warning"],
        ).pack(pady=5)

        # Text area
        text_frame = tk.Frame(word_window, bg=self.colors["bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

        text_widget = scrolledtext.ScrolledText(
            text_frame, font=("Courier", 12), relief=tk.FLAT, padx=10, pady=10
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Load existing words
        if os.path.exists("word.txt"):
            try:
                with open("word.txt", "r") as f:
                    text_widget.insert(tk.END, f.read())
            except:
                pass
        else:
            text_widget.insert(tk.END, "doctor\nrat\ndog\ncat\nrow\nend\n")

        # Word count label
        count_label = tk.Label(
            word_window,
            text="Words: 0",
            font=("Arial", 10),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        )
        count_label.pack(pady=5)

        def update_count(*args):
            content = text_widget.get("1.0", tk.END)
            words = [w.strip() for w in content.split("\n") if w.strip()]
            count_label.config(text=f"Words: {len(words)}")

        text_widget.bind("<KeyRelease>", update_count)
        update_count()

        # Save button
        def save_words():
            content = text_widget.get("1.0", tk.END).strip()
            words = [w.strip().lower() for w in content.split("\n") if w.strip()]

            if not words:
                messagebox.showwarning("Warning", "Please enter at least one word!")
                return

            with open("word.txt", "w") as f:
                f.write("\n".join(words))

            self.log(f"✅ Saved {len(words)} words\n\n", "success")
            self.log("Words: " + ", ".join(words[:10]))
            if len(words) > 10:
                self.log(f"... and {len(words) - 10} more")
            self.log("\n\n✓ Ready to solve!\n", "success")

            # Enable solve button
            self.words_ready = True
            self.btn_solve.config(bg=self.colors["button"], state=tk.NORMAL)
            self.btn_words.config(bg=self.colors["success"])
            # Lock this step to prevent re-clicks until reset
            self.lock_button(self.btn_words)
            self.update_status(f"{len(words)} words loaded ✓", "success")
            word_window.destroy()

        tk.Button(
            word_window,
            text="💾 SAVE & CONTINUE",
            command=save_words,
            font=("Arial", 14, "bold"),
            bg=self.colors["success"],
            fg="white",
            relief=tk.FLAT,
            padx=40,
            pady=15,
            cursor="hand2",
        ).pack(pady=20)

    # ========================================================================
    # STEP 3: SOLVE & VISUALIZE
    # ========================================================================

    def solve_and_visualize(self):
        """Solve puzzle with step-by-step visualization (trace replay, no turtle)"""
        self.clear_output()
        self.log("═" * 60 + "\n")
        self.log("STEP 3: SOLVE & VISUALIZE\n", "title")
        self.log("═" * 60 + "\n\n")

        if not SOLVER_AVAILABLE:
            self.log("ERROR: crossword_solver.py not found!\n", "error")
            self.update_status("Solver module missing", "error")
            return

        self.log("Initializing solver...\n")
        self.update_status("Solving puzzle...", "warning")
        self.root.update()

        # Lock this step immediately to avoid double-clicks; requires Reset to start over
        self.lock_button(self.btn_solve)

        try:
            # Read grid
            grid_reader = GridReader("grid.txt")
            slots = grid_reader.find_slots()

            self.log(f"   Grid: {grid_reader.rows}×{grid_reader.cols}\n")
            self.log(f"   Slots found: {len(slots)}\n\n")

            # Initialize solver
            solver = PrologCrosswordSolver()
            solver.load_slots(slots)
            solver.load_words("word.txt")

            self.root.update()

            self.log("\n🔍 Solving with backtracking (capturing trace)...\n\n")

            # Solve with trace
            slot_ids = [s["id"] for s in slots]
            start_time = time.time()
            solution, steps = solver.solve_with_trace(slot_ids)
            end_time = time.time()

            self.log(f"Trace steps: {len(steps)}\n")

            if solution:
                self.log(f"\n{'=' * 60}\n")
                self.log(f"✅ SOLVED in {end_time - start_time:.2f}s!\n", "success")
                self.log(f"{'=' * 60}\n\n")

                for slot_id, word in solution:
                    slot = slots[slot_id]
                    self.log(f"  {slot_id}. {word.upper()} ({slot['direction']})\n")

                self.update_status("✅ Puzzle solved! Open trace window to replay.", "success")
                self.btn_solve.config(bg=self.colors["success"])
            else:
                self.log("\nNo solution found (showing search trace).\n", "error")
                self.update_status("No solution found (trace available)", "warning")

            # Open the trace viewer window
            StepViewer(self.root, grid_reader.grid, slots, steps, solution)

        except Exception as e:
            self.log(f"\n✗ ERROR: {str(e)}\n", "error")
            import traceback

            self.log(f"\n{traceback.format_exc()}\n")
            self.update_status("Solving failed", "error")

    # ========================================================================
    # HELPERS
    # ========================================================================

    def clear_output(self):
        """Clear output text"""
        self.output.delete("1.0", tk.END)

    def log(self, message, style="normal"):
        """Log message"""
        self.output.insert(tk.END, message)
        self.output.see(tk.END)
        self.root.update()

    def update_status(self, message, status_type="normal"):
        """Update status bar"""
        colors = {
            "success": self.colors["success"],
            "error": self.colors["error"],
            "warning": self.colors["warning"],
            "normal": self.colors["fg"],
        }
        self.status.config(text=message, fg=colors.get(status_type, self.colors["fg"]))

    def reset_all(self):
        """Reset everything"""
        if messagebox.askyesno("Reset", "Reset all progress?"):
            # No turtle canvas used in trace viewer mode
            pass

            self.image_path = None
            self.grid_extracted = False
            self.words_ready = False

            # Re-enable first step button; other steps remain disabled until progressed again
            self.btn_upload.config(
                bg=self.colors["button"], state=tk.NORMAL, cursor="hand2"
            )
            self.btn_words.config(bg=self.colors["panel_bg"], state=tk.DISABLED)
            self.btn_solve.config(bg=self.colors["panel_bg"], state=tk.DISABLED)

            self.clear_output()
            self.log("Reset complete. Start with 'UPLOAD IMAGE'.\n")
            self.update_status("Ready", "normal")

    def run(self):
        """Start application"""
        self.root.mainloop()
