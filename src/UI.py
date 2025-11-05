"""
============================================================================
CROSSWORD SOLVER - SIMPLIFIED UI
Upload Image → Input Words → Solve with Visualization
============================================================================
"""

import turtle
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import time

# Import modules
try:
    from OCR import CrosswordExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    print("Warning: OCR.py not found")

try:
    from crossword_solver import PrologCrosswordSolver, GridReader, CrosswordDrawer
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    print("Warning: crossword_solver.py not found")


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
        
        # Colors
        self.colors = {
            'bg': '#AAAAAA',
            'fg': '#000000',
            'button': '#3498DB',
            'button_hover': '#000000',
            'success': '#27AE60',
            'error': '#E74C3C',
            'warning': '#F39C12',
            'panel_bg': "#000000"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Title
        title = tk.Label(
            self.root,
            text="🧩 CROSSWORD SOLVER",
            font=("Arial", 28, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        )
        title.pack(pady=20)
        
        # Main container
        container = tk.Frame(self.root, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Left panel - Actions
        left_panel = tk.Frame(container, bg=self.colors['panel_bg'], relief=tk.RAISED, bd=3)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        
        tk.Label(
            left_panel,
            text="ACTIONS",
            font=("Arial", 16, "bold"),
            bg=self.colors['panel_bg'],
            fg=self.colors['fg']
        ).pack(pady=20)
        
        # Step 1: Upload & Extract
        self.btn_upload = self.create_big_button(
            left_panel,
            "UPLOAD IMAGE\n& EXTRACT GRID",
            self.upload_and_extract,
            self.colors['panel_bg']
        )
        self.btn_upload.pack(padx=20, pady=15)
        
        tk.Label(
            left_panel,
            text="↓",
            font=("Arial", 20, "bold"),
            bg=self.colors['panel_bg'],
            fg=self.colors['warning']
        ).pack()
        
        # Step 2: Input Words
        self.btn_words = self.create_big_button(
            left_panel,
            "INPUT WORDS",
            self.open_word_input,
            self.colors['panel_bg'],
            state=tk.DISABLED
        )
        self.btn_words.pack(padx=20, pady=15)
        
        tk.Label(
            left_panel,
            text="↓",
            font=("Arial", 20, "bold"),
            bg=self.colors['panel_bg'],
            fg=self.colors['warning']
        ).pack()
        
        # Step 3: Solve & Visualize
        self.btn_solve = self.create_big_button(
            left_panel,
            "SOLVE &\nVISUALIZE",
            self.solve_and_visualize,
            self.colors['panel_bg'],
            state=tk.DISABLED
        )
        self.btn_solve.pack(padx=20, pady=15)
        
        # Reset button
        tk.Label(left_panel, text="", bg=self.colors['panel_bg'], height=2).pack()
        tk.Button(
            left_panel,
            text="RESET",
            command=self.reset_all,
            font=("Arial", 11, "bold"),
            bg=self.colors['error'],
            fg='red',
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2"
        ).pack(padx=20, pady=10)
        
        # Right panel - Output
        right_panel = tk.Frame(container, bg=self.colors['panel_bg'], relief=tk.RAISED, bd=3)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_panel,
            text="OUTPUT LOG",
            font=("Arial", 16, "bold"),
            bg=self.colors['panel_bg'],
            fg=self.colors['fg']
        ).pack(pady=20)
        
        # Output text area
        self.output = scrolledtext.ScrolledText(
            right_panel,
            font=("Courier", 10),
            bg='#ECF0F1',
            fg='#2C3E50',
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD
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
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            pady=10
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
            fg='#AAAAAA',
            activebackground=self.colors['button_hover'],
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=25,
            cursor="hand2" if state == tk.NORMAL else "arrow",
            state=state,
            width=18,
            height=3
        )
        return btn
    
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
                ("All files", "*.*")
            ]
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
                rows=None,
                cols=None,
                image_path=file_path,
                debug=True
            )
            
            self.log("   Detecting grid region...\n")
            grid = extractor.extract_grid()
            
            self.log("   Analyzing cells...\n")
            extractor.save_grid("grid.txt")
            
            self.log(f"\n✅ SUCCESS! Grid extracted: {len(grid)}×{len(grid[0])}\n\n", "success")
            
            # Display grid
            self.log("Extracted Grid:\n")
            self.log("─" * 50 + "\n")
            for row in grid:
                self.log(''.join(row) + '\n')
            self.log("─" * 50 + "\n\n")
            
            self.log("Grid saved to: grid.txt\n\n")
            self.log("✓ Ready for next step!\n", "success")
            
            # Enable next button
            self.grid_extracted = True
            self.btn_words.config(bg=self.colors['button'], state=tk.NORMAL)
            self.btn_upload.config(bg=self.colors['success'])
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
        word_window.configure(bg=self.colors['bg'])
        word_window.transient(self.root)
        word_window.grab_set()
        
        tk.Label(
            word_window,
            text="📝 Enter Your Word List",
            font=("Arial", 18, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        ).pack(pady=20)
        
        tk.Label(
            word_window,
            text="Enter one word per line (case insensitive)",
            font=("Arial", 11),
            bg=self.colors['bg'],
            fg=self.colors['warning']
        ).pack(pady=5)
        
        # Text area
        text_frame = tk.Frame(word_window, bg=self.colors['bg'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame,
            font=("Courier", 12),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Load existing words
        if os.path.exists("word.txt"):
            try:
                with open("word.txt", 'r') as f:
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
            bg=self.colors['bg'],
            fg=self.colors['fg']
        )
        count_label.pack(pady=5)
        
        def update_count(*args):
            content = text_widget.get("1.0", tk.END)
            words = [w.strip() for w in content.split('\n') if w.strip()]
            count_label.config(text=f"Words: {len(words)}")
        
        text_widget.bind('<KeyRelease>', update_count)
        update_count()
        
        # Save button
        def save_words():
            content = text_widget.get("1.0", tk.END).strip()
            words = [w.strip().lower() for w in content.split('\n') if w.strip()]
            
            if not words:
                messagebox.showwarning("Warning", "Please enter at least one word!")
                return
            
            with open("word.txt", 'w') as f:
                f.write('\n'.join(words))
            
            self.log(f"✅ Saved {len(words)} words\n\n", "success")
            self.log("Words: " + ", ".join(words[:10]))
            if len(words) > 10:
                self.log(f"... and {len(words)-10} more")
            self.log("\n\n✓ Ready to solve!\n", "success")
            
            # Enable solve button
            self.words_ready = True
            self.btn_solve.config(bg=self.colors['button'], state=tk.NORMAL)
            self.btn_words.config(bg=self.colors['success'])
            self.update_status(f"{len(words)} words loaded ✓", "success")
            word_window.destroy()
        
        tk.Button(
            word_window,
            text="💾 SAVE & CONTINUE",
            command=save_words,
            font=("Arial", 14, "bold"),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            padx=40,
            pady=15,
            cursor="hand2"
        ).pack(pady=20)
    
    # ========================================================================
    # STEP 3: SOLVE & VISUALIZE
    # ========================================================================
    
    def solve_and_visualize(self):
        """Solve puzzle with step-by-step visualization"""
        self.clear_output()
        self.log("═" * 60 + "\n")
        self.log("STEP 3: SOLVE & VISUALIZE\n", "title")
        self.log("═" * 60 + "\n\n")
        
        if not SOLVER_AVAILABLE:
            self.log("✗ ERROR: crossword_solver.py not found!\n", "error")
            self.update_status("Solver module missing", "error")
            return
        
        self.log("🔄 Initializing solver...\n")
        self.update_status("Solving puzzle...", "warning")
        self.root.update()
        
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
            
            # Create visualization
            self.log("🎨 Opening visualization window...\n")
            self.root.update()
            
            drawer = CrosswordDrawer(grid_reader.grid, cell_size=50, animate=True)
            drawer.draw_grid()
            drawer.draw_slot_numbers(slots)
            drawer.slots_storage = slots
            
            # Set callback for step-by-step visualization
            def visualize_step(slot_id, word, slot, is_placing, placements):
                self.log(f"{'  ▶' if is_placing else '  ◀'} Slot {slot_id}: {word.upper()}\n")
                self.root.update()
            
            solver.set_step_callback(drawer.animate_word_placement)
            
            self.log("\n🔍 Solving with backtracking...\n\n")
            
            # Solve
            slot_ids = [s['id'] for s in slots]
            start_time = time.time()
            solution = solver.solve(slot_ids)
            end_time = time.time()
            
            if solution:
                self.log(f"\n{'='*60}\n")
                self.log(f"✅ SOLVED in {end_time - start_time:.2f}s!\n", "success")
                self.log(f"{'='*60}\n\n")
                
                for slot_id, word in solution:
                    slot = slots[slot_id]
                    self.log(f"  {slot_id}. {word.upper()} ({slot['direction']})\n")
                
                # Draw final solution
                drawer.draw_solution(slots, solution, animated=False)
                
                self.update_status("✅ Puzzle solved!", "success")
                self.btn_solve.config(bg=self.colors['success'])
                
            else:
                self.log("\n✗ No solution found\n", "error")
                self.log("\nTry adding more words or checking the grid.\n")
                self.update_status("No solution found", "error")
                drawer.draw_solution(slots, None)
                
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
            "success": self.colors['success'],
            "error": self.colors['error'],
            "warning": self.colors['warning'],
            "normal": self.colors['fg']
        }
        self.status.config(text=message, fg=colors.get(status_type, self.colors['fg']))
    
    def reset_all(self):
        """Reset everything"""
        if messagebox.askyesno("Reset", "Reset all progress?"):
            self.image_path = None
            self.grid_extracted = False
            self.words_ready = False
            
            self.btn_upload.config(bg=self.colors['button'])
            self.btn_words.config(bg=self.colors['panel_bg'], state=tk.DISABLED)
            self.btn_solve.config(bg=self.colors['panel_bg'], state=tk.DISABLED)
            
            self.clear_output()
            self.log("Reset complete. Start with 'UPLOAD IMAGE'.\n")
            self.update_status("Ready", "normal")
    
    def run(self):
        """Start application"""
        self.root.mainloop()


def main():
    """Main entry"""
    print("=" * 70)
    print("   CROSSWORD SOLVER - SIMPLIFIED UI")
    print("=" * 70)
    
    app = SimpleCrosswordUI()
    app.run()


if __name__ == "__main__":
    main()