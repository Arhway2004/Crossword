import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
from typing import List, Optional, Dict
from enum import Enum
from collections import defaultdict


class GridFormat(Enum):
    """Types of crossword grid formats"""

    STANDARD = "standard"  # Rectangular grid with black cells
    IRREGULAR = "irregular"  # Scattered cell layout
    UNKNOWN = "unknown"


class EnhancedCrosswordExtractor:
    """Extract crossword grid from both standard and irregular formats"""

    def __init__(self, image_path: str, debug: bool = False):
        self.image_path = image_path
        self.debug = debug
        self.image = None
        self.grid = None
        self.grid_format = GridFormat.UNKNOWN
        self.cells = []  # Individual detected cells

        self._load_image()

    def _load_image(self):
        """Load and validate the input image"""
        print(f"Loading image: {self.image_path}")

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        print(f"Image loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")

    def detect_grid_format(self) -> GridFormat:
        """
        Automatically detect which format the crossword uses
        Returns: GridFormat enum
        """
        print("Detecting grid format...")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for rectangular contours (cells)
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > self.image.shape[0] * self.image.shape[1] * 0.8:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Check if roughly square (cell-like)
                if 0.7 < aspect_ratio < 1.3 and w > 20 and h > 20:
                    rectangles.append((x, y, w, h, area))

        if len(rectangles) < 4:
            print("Warning: Too few cells detected")
            return GridFormat.UNKNOWN

        print(f"Detected {len(rectangles)} potential cells")

        # Analyze cell distribution
        rectangles.sort(key=lambda r: (r[1], r[0]))  # Sort by y, then x

        # Check if cells form a regular grid (standard format)
        y_positions = [r[1] for r in rectangles]
        x_positions = [r[0] for r in rectangles]

        # Count unique rows and columns (with tolerance)
        def count_unique_positions(positions, tolerance=15):
            if not positions:
                return 0
            unique = [positions[0]]
            for pos in positions[1:]:
                if all(abs(pos - u) > tolerance for u in unique):
                    unique.append(pos)
            return len(unique)

        unique_rows = count_unique_positions(y_positions)
        unique_cols = count_unique_positions(x_positions)

        # If cells align in clear rows/columns, it's standard format
        expected_cells = unique_rows * unique_cols
        actual_cells = len(rectangles)

        alignment_ratio = actual_cells / expected_cells if expected_cells > 0 else 0

        print(f"Analysis: {unique_rows} rows x {unique_cols} cols")
        print(f"Expected cells: {expected_cells}, Actual: {actual_cells}")
        print(f"Alignment ratio: {alignment_ratio:.2f}")

        if alignment_ratio > 0.5 and unique_rows >= 3 and unique_cols >= 3:
            self.grid_format = GridFormat.STANDARD
            print("✓ Detected: STANDARD rectangular grid format")
        else:
            self.grid_format = GridFormat.IRREGULAR
            print("✓ Detected: IRREGULAR scattered grid format")

        return self.grid_format

    def extract_cells_irregular(self) -> List[Dict]:
        """
        Extract individual cells for irregular format
        Returns list of cell dictionaries with position and connections
        """
        print("Extracting irregular grid cells...")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        cells = []
        cell_id = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 200 or area > self.image.shape[0] * self.image.shape[1] * 0.5:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                if 0.6 < aspect_ratio < 1.4 and w > 20 and h > 20:
                    # Extract cell number using OCR
                    cell_img = gray[y : y + h, x : x + w]
                    cell_number = self._extract_cell_number(cell_img)

                    cells.append(
                        {
                            "id": cell_id,
                            "number": cell_number,
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w // 2,
                            "center_y": y + h // 2,
                        }
                    )
                    cell_id += 1

        # Sort by position (top to bottom, left to right)
        cells.sort(key=lambda c: (c["y"], c["x"]))

        print(f"Extracted {len(cells)} cells")

        if self.debug:
            debug_img = self.image.copy()
            for cell in cells:
                cv2.rectangle(
                    debug_img,
                    (cell["x"], cell["y"]),
                    (cell["x"] + cell["width"], cell["y"] + cell["height"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    debug_img,
                    str(cell.get("number", "?")),
                    (cell["x"] + 5, cell["y"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
            cv2.imwrite("debug_irregular_cells.png", debug_img)

        self.cells = cells
        return cells

    def _extract_cell_number(self, cell_img: np.ndarray) -> Optional[int]:
        """Extract cell number using OCR"""
        try:
            # Focus on top-left corner where numbers typically appear
            h, w = cell_img.shape
            corner = cell_img[0 : h // 3, 0 : w // 3]

            # Enhance for OCR
            corner = cv2.resize(corner, None, fx=2, fy=2)
            _, corner = cv2.threshold(corner, 127, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(corner, config="--psm 10 digits")
            text = text.strip()

            if text.isdigit():
                return int(text)
        except:
            pass
        return None

    def build_irregular_grid_structure(self, cells: List[Dict]) -> List[List[str]]:
        """
        Build grid structure from irregular cells by detecting adjacency
        """
        print("Building irregular grid structure...")

        if not cells:
            return []

        # Find connected cells (horizontally and vertically adjacent)
        def are_adjacent(c1, c2, tolerance=10):
            """Check if two cells are adjacent"""
            # Horizontal adjacency
            if abs(c1["center_y"] - c2["center_y"]) < tolerance:
                gap = abs(c1["x"] + c1["width"] - c2["x"])
                if gap < tolerance:
                    return "right"
                gap = abs(c2["x"] + c2["width"] - c1["x"])
                if gap < tolerance:
                    return "left"

            # Vertical adjacency
            if abs(c1["center_x"] - c2["center_x"]) < tolerance:
                gap = abs(c1["y"] + c1["height"] - c2["y"])
                if gap < tolerance:
                    return "down"
                gap = abs(c2["y"] + c2["height"] - c1["y"])
                if gap < tolerance:
                    return "up"

            return None

        # Build adjacency map
        for cell in cells:
            cell["neighbors"] = {"left": None, "right": None, "up": None, "down": None}

        for i, c1 in enumerate(cells):
            for j, c2 in enumerate(cells):
                if i != j:
                    direction = are_adjacent(c1, c2)
                    if direction:
                        c1["neighbors"][direction] = j

        # Find grid bounds
        min_x = min(c["x"] for c in cells)
        max_x = max(c["x"] + c["width"] for c in cells)
        min_y = min(c["y"] for c in cells)
        max_y = max(c["y"] + c["height"] for c in cells)

        # Group cells into rows
        rows = defaultdict(list)
        avg_height = np.mean([c["height"] for c in cells])

        for cell in cells:
            row_idx = int((cell["y"] - min_y) / avg_height)
            rows[row_idx].append(cell)

        # Sort cells in each row by x position
        for row_idx in rows:
            rows[row_idx].sort(key=lambda c: c["x"])

        # Build grid
        max_cols = max(len(rows[r]) for r in rows)
        grid = []

        for row_idx in sorted(rows.keys()):
            row = ["X"] * max_cols
            for col_idx, cell in enumerate(rows[row_idx]):
                row[col_idx] = "_"
            grid.append(row)

        print(f"Built {len(grid)} x {max_cols} grid from irregular format")
        return grid

    def extract_standard_grid(self) -> List[List[str]]:
        """
        Extract grid from standard rectangular format
        """
        print("Extracting standard grid format...")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        if self.debug:
            cv2.imwrite("debug_standard_thresh.png", thresh)

        # Detect grid lines using Hough transform
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        if lines is None:
            print("Warning: No lines detected, using morphology approach")
            return self._extract_standard_grid_morphology(thresh)

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(y2 - y1) < 10:  # Horizontal line
                h_lines.append(y1)
            elif abs(x2 - x1) < 10:  # Vertical line
                v_lines.append(x1)

        # Cluster lines
        h_lines = self._cluster_lines(h_lines)
        v_lines = self._cluster_lines(v_lines)

        print(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines")

        if len(h_lines) < 2 or len(v_lines) < 2:
            print("Warning: Insufficient lines, using morphology approach")
            return self._extract_standard_grid_morphology(thresh)

        # Build grid based on detected lines
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1
        grid = []

        for r in range(rows):
            row = []
            for c in range(cols):
                y1 = h_lines[r] + 5
                y2 = h_lines[r + 1] - 5
                x1 = v_lines[c] + 5
                x2 = v_lines[c + 1] - 5

                # Check if cell is black/blocked
                cell = gray[y1:y2, x1:x2]
                if cell.size == 0:
                    row.append("X")
                    continue

                avg_intensity = np.mean(cell)

                # If mostly dark, it's blocked
                if avg_intensity < 100:
                    row.append("X")
                else:
                    row.append("_")

            grid.append(row)

        print(f"Extracted {rows} x {cols} standard grid")

        if self.debug:
            self._visualize_grid(grid)

        return grid

    def _extract_standard_grid_morphology(self, binary: np.ndarray) -> List[List[str]]:
        """Fallback method using morphological operations"""
        h, w = binary.shape

        # Detect lines using morphology
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        # Find line positions
        h_projection = np.sum(horizontal, axis=1)
        v_projection = np.sum(vertical, axis=0)

        h_lines = self._find_peaks(h_projection, h // 20)
        v_lines = self._find_peaks(v_projection, w // 20)

        if len(h_lines) < 2 or len(v_lines) < 2:
            # Use estimated grid
            rows = 7  # Default
            cols = 7
            h_lines = [int(i * h / rows) for i in range(rows + 1)]
            v_lines = [int(i * w / cols) for i in range(cols + 1)]

        rows = len(h_lines) - 1
        cols = len(v_lines) - 1

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grid = []

        for r in range(rows):
            row = []
            for c in range(cols):
                y1 = h_lines[r] + 3
                y2 = h_lines[r + 1] - 3
                x1 = v_lines[c] + 3
                x2 = v_lines[c + 1] - 3

                cell = gray[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

                if cell.size == 0:
                    row.append("X")
                    continue

                if np.mean(cell) < 100:
                    row.append("X")
                else:
                    row.append("_")

            grid.append(row)

        return grid

    def _cluster_lines(self, lines: List[int], tolerance: int = 15) -> List[int]:
        """Cluster nearby line positions"""
        if not lines:
            return []

        lines = sorted(lines)
        clusters = [[lines[0]]]

        for line in lines[1:]:
            if abs(line - np.mean(clusters[-1])) < tolerance:
                clusters[-1].append(line)
            else:
                clusters.append([line])

        return [int(np.mean(cluster)) for cluster in clusters]

    def _find_peaks(self, signal: np.ndarray, min_distance: int) -> List[int]:
        """Find peaks in 1D signal"""
        threshold = np.mean(signal) + 0.5 * np.std(signal)

        peaks = []
        i = 0
        while i < len(signal):
            if signal[i] > threshold:
                window_end = min(i + min_distance, len(signal))
                local_max_idx = i + np.argmax(signal[i:window_end])
                peaks.append(local_max_idx)
                i = local_max_idx + min_distance
            else:
                i += 1

        return peaks

    def _visualize_grid(self, grid: List[List[str]]):
        """Debug visualization of extracted grid"""
        print("\nExtracted Grid:")
        for row in grid:
            print("".join(row))

    def extract_grid(self) -> List[List[str]]:
        """
        Main method to extract grid - automatically handles both formats
        """
        # Detect format
        fmt = self.detect_grid_format()

        if fmt == GridFormat.IRREGULAR:
            cells = self.extract_cells_irregular()
            self.grid = self.build_irregular_grid_structure(cells)
        elif fmt == GridFormat.STANDARD:
            self.grid = self.extract_standard_grid()
        else:
            print("Warning: Unknown format, attempting standard extraction")
            self.grid = self.extract_standard_grid()

        return self.grid

    def save_grid(self, filename: str = "grid.txt"):
        """Save extracted grid to file"""
        if self.grid is None:
            print("Warning: No grid to save")
            return

        print(f"\nSaving grid to {filename}...")

        with open(filename, "w") as f:
            for row in self.grid:
                f.write("".join(row) + "\n")

        print(f"✓ Grid saved: {len(self.grid)}x{len(self.grid[0]) if self.grid else 0}")

    def extract_word_list(self, filename: str = "word.txt") -> List[str]:
        """
        Extract word list from image (if available in Word Box)
        """
        print("\nExtracting word list...")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Look for text regions (typically at bottom)
        h, w = gray.shape
        text_region = gray[int(h * 0.7) : h, 0:w]

        # OCR for words
        text = pytesseract.image_to_string(text_region)

        # Parse words
        words = []
        for line in text.split("\n"):
            # Extract individual words
            line_words = re.findall(r"\b[a-z]{2,}\b", line.lower())
            words.extend(line_words)

        # Remove duplicates
        words = list(set(words))

        if words:
            print(f"✓ Extracted {len(words)} words")
            with open(filename, "w") as f:
                for word in sorted(words):
                    f.write(word + "\n")
        else:
            print("No words extracted from image")

        return words


def main():
    """Test the enhanced extractor"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Crossword Extractor")
    parser.add_argument("--image", required=True, help="Path to crossword image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output", default="grid.txt", help="Output grid file")

    args = parser.parse_args()

    print("=" * 70)
    print("ENHANCED CROSSWORD EXTRACTOR")
    print("Supports both standard and irregular grid formats")
    print("=" * 70)

    extractor = EnhancedCrosswordExtractor(args.image, debug=args.debug)

    grid = extractor.extract_grid()
    extractor.save_grid(args.output)

    # Try to extract words
    extractor.extract_word_list()

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
