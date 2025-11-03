"""
Crossword Puzzle Image Extractor - Improved Grid Detection
Extracts grid layout and clues from crossword puzzle images using OCR
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
from typing import Tuple, List, Optional
import argparse


class CrosswordExtractor:
    """Extract crossword grid and clues from images"""
    
    def __init__(self, rows, cols, image_path: str, debug: bool = False):
        self.rows = rows
        self.cols = cols
        self.image_path = image_path
        self.debug = debug
        self.image = None
        self.grid = None
        self.clues = []
        
        # Load and validate image
        self._load_image()
        
    def _load_image(self):
        """Load and validate the input image"""
        print(f"Loading image: {self.image_path}")
        
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        print(f"Image loaded successfully: {self.image.shape[1]}x{self.image.shape[0]} pixels")
    
    def _detect_grid_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the crossword grid region in the image
        
        Returns:
            Tuple of (x, y, width, height) for grid region, or None if not found
        """
        print("Detecting grid region...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        if self.debug:
            cv2.imwrite("debug_threshold.png", thresh)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find the largest rectangular contour (likely the grid)
        max_area = 0
        grid_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Check if contour is roughly square/rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4 and area > 1000:  # Minimum area threshold
                    max_area = area
                    grid_contour = contour
        
        if grid_contour is None:
            print("Warning: Could not detect grid region automatically")
            return None
        
        x, y, w, h = cv2.boundingRect(grid_contour)
        print(f"Grid region detected: {w}x{h} pixels at position ({x}, {y})")
        
        if self.debug:
            debug_img = self.image.copy()
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imwrite("debug_grid_region.png", debug_img)
        
        return (x, y, w, h)
    
    def _rotate_and_align(self, binary: np.ndarray) -> np.ndarray:
        """
        Detect and correct rotation of the grid
        
        Args:
            binary: Binary image of grid
            
        Returns:
            Rotation-corrected binary image
        """
        h, w = binary.shape[:2]
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) == 0:
            return binary
        
        angle = cv2.minAreaRect(coords)[-1]
        
        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only rotate if angle is significant
        if abs(angle) > 0.5:
            print(f"Rotating grid by {angle:.2f} degrees")
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            if self.debug:
                cv2.imwrite("debug_rotated.png", binary)
        
        return binary
    
    def _detect_grid_lines_morphology(self, binary: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect grid lines using morphological operations
        
        Args:
            binary: Binary image of grid
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines) as lists of y and x coordinates
        """
        h, w = binary.shape[:2]
        
        # Create kernels for detecting lines
        # Horizontal lines: wide and short
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w//10, 1))
        # Vertical lines: tall and narrow
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//10))
        
        # Extract horizontal and vertical lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        
        if self.debug:
            cv2.imwrite("debug_horizontal_lines.png", horizontal)
            cv2.imwrite("debug_vertical_lines.png", vertical)
        
        # Find line positions using projection profiles
        h_projection = np.sum(horizontal, axis=1)
        v_projection = np.sum(vertical, axis=0)
        
        # Find peaks in projections
        h_lines = self._find_peaks(h_projection, min_distance=h//20)
        v_lines = self._find_peaks(v_projection, min_distance=w//20)
        
        print(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
        
        return h_lines, v_lines
    
    def _find_peaks(self, signal: np.ndarray, min_distance: int) -> List[int]:
        """
        Find peaks in a 1D signal with minimum distance constraint
        
        Args:
            signal: 1D array of signal values
            min_distance: Minimum distance between peaks
            
        Returns:
            List of peak positions
        """
        # Threshold: peaks should be above mean
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        
        peaks = []
        i = 0
        while i < len(signal):
            if signal[i] > threshold:
                # Find local maximum in window
                window_end = min(i + min_distance, len(signal))
                local_max_idx = i + np.argmax(signal[i:window_end])
                peaks.append(local_max_idx)
                i = local_max_idx + min_distance
            else:
                i += 1
        
        return peaks
    
    def _estimate_grid_dimensions(self, h_lines: List[int], v_lines: List[int], 
                                  w: int, h: int) -> Tuple[int, int, List[int], List[int]]:
        """
        Estimate grid dimensions and regularize line positions
        
        Args:
            h_lines: Detected horizontal line positions
            v_lines: Detected vertical line positions
            w: Width of grid image
            h: Height of grid image
            
        Returns:
            Tuple of (rows, cols, regularized_h_lines, regularized_v_lines)
        """
        # If we have good line detection, use it
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            rows = len(h_lines) - 1
            cols = len(v_lines) - 1
            
            # Check if detected dimensions are close to expected
            if self.rows and abs(rows - self.rows) > 3:
                print(f"Warning: Detected {rows} rows but expected {self.rows}")
            if self.cols and abs(cols - self.cols) > 3:
                print(f"Warning: Detected {cols} cols but expected {self.cols}")
            
            print(f"Using detected grid dimensions: {rows}x{cols}")
            return rows, cols, h_lines, v_lines
        
        # Fallback: use provided dimensions or default
        rows = self.rows if self.rows else 15
        cols = self.cols if self.cols else 15
        
        print(f"Using fallback grid dimensions: {rows}x{cols}")
        
        # Create evenly spaced grid lines
        cell_height = h / rows
        cell_width = w / cols
        
        h_lines = [int(i * cell_height) for i in range(rows + 1)]
        v_lines = [int(i * cell_width) for i in range(cols + 1)]
        
        return rows, cols, h_lines, v_lines
    
    def _extract_grid_layout(self, grid_region: Tuple[int, int, int, int]) -> List[List[str]]:
        """
        Extract the grid layout from the detected region
        
        Args:
            grid_region: Tuple of (x, y, width, height)
            
        Returns:
            2D list representing the grid
        """
        print("Extracting grid layout...")
        
        x, y, w, h = grid_region
        grid_img = self.image[y:y+h, x:x+w].copy()
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold with optimized parameters
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        if self.debug:
            cv2.imwrite("debug_grid_binary.png", binary)
        
        # Rotate and align if needed
        binary = self._rotate_and_align(binary)
        
        # Detect grid lines using morphology
        h_lines, v_lines = self._detect_grid_lines_morphology(binary)
        
        # Estimate dimensions and regularize lines
        rows, cols, h_lines, v_lines = self._estimate_grid_dimensions(
            h_lines, v_lines, w, h
        )
        
        # Debug: draw detected lines
        if self.debug:
            debug_lines = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for y_pos in h_lines:
                cv2.line(debug_lines, (0, y_pos), (w, y_pos), (0, 255, 0), 1)
            for x_pos in v_lines:
                cv2.line(debug_lines, (x_pos, 0), (x_pos, h), (255, 0, 0), 1)
            cv2.imwrite("debug_lines.png", debug_lines)
        
        # Extract cell states
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                # Get cell boundaries with padding
                y1 = h_lines[r] + 3
                y2 = h_lines[r + 1] - 3
                x1 = v_lines[c] + 3
                x2 = v_lines[c + 1] - 3
                
                # Ensure valid bounds
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)
                
                if y2 <= y1 or x2 <= x1:
                    row.append('_')
                    continue
                
                # Extract cell from original grayscale (not binary)
                cell = gray[y1:y2, x1:x2]
                
                if cell.size == 0:
                    row.append('_')
                    continue
                
                # Calculate darkness using median for robustness
                dark_ratio = np.mean(cell < 128)
                
                # Threshold: if more than 60% dark, it's filled
                if dark_ratio > 0.6:
                    row.append('X')
                else:
                    row.append('_')
            
            grid.append(row)
        
        # Post-processing: check if we need to invert
        total_cells = rows * cols
        filled_cells = sum(row.count('X') for row in grid)
        filled_ratio = filled_cells / total_cells
        
        if filled_ratio > 0.6:
            print(f"Warning: {filled_ratio*100:.1f}% cells are filled. Inverting grid...")
            grid = [['_' if cell == 'X' else 'X' for cell in row] for row in grid]
        
        if self.debug:
            print("\nGrid preview:")
            for row in grid:
                print(''.join(row))
            print(f"\nFilled cells: {sum(row.count('X') for row in grid)}/{total_cells}")
        
        return grid
    
    def extract_grid(self) -> List[List[str]]:
        """
        Main method to extract grid layout
        
        Returns:
            2D list representing the grid
        """
        # Detect grid region
        grid_region = self._detect_grid_region()
        
        if grid_region is None:
            # Fallback: use entire image or central portion
            h, w = self.image.shape[:2]
            grid_region = (w//8, h//8, w*3//4, h*3//4)
            print("Using fallback grid region")
        
        # Extract grid layout
        self.grid = self._extract_grid_layout(grid_region)
        return self.grid
    
    def _extract_text_regions(self) -> str:
        """
        Extract text from the image using OCR
        
        Returns:
            Raw text extracted from image
        """
        print("Extracting text using OCR...")
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing for better OCR
        # Increase contrast
        enhanced = cv2.equalizeHist(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        if self.debug:
            cv2.imwrite("debug_ocr_preprocessed.png", denoised)
        
        # Perform OCR
        custom_config = r'--oem 3 --psm 6'  # PSM 6: Assume uniform block of text
        text = pytesseract.image_to_string(denoised, config=custom_config)
        
        if self.debug:
            print("Raw OCR output:")
            print(text)
            print("-" * 50)
        
        return text
    
    def _parse_clues(self, text: str) -> List[str]:
        """
        Parse and clean clue text
        
        Args:
            text: Raw OCR text
            
        Returns:
            List of cleaned clue strings
        """
        print("Parsing and cleaning clues...")
        
        # Split into lines
        lines = text.split('\n')
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Filter out header lines and non-clue text
        header_keywords = [
            'across', 'down', 'crossword', 'puzzle', 'clues',
            'copyright', 'by', 'edited', 'page', 'www', 'http'
        ]
        
        clues = []
        for line in lines:
            line_lower = line.lower()
            
            # Skip header lines
            if any(keyword in line_lower for keyword in header_keywords):
                # Exception: if line starts with a number, it might be a valid clue
                if not re.match(r'^\d+\.', line):
                    continue
            
            # Check if line starts with a number followed by period
            # Pattern: "NUMBER. CLUE TEXT"
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            
            if match:
                number = match.group(1)
                clue_text = match.group(2).strip()
                
                # Clean up common OCR errors
                clue_text = self._clean_ocr_text(clue_text)
                
                # Reconstruct clue
                clues.append(f"{number}. {clue_text}")
        
        print(f"Extracted {len(clues)} clues")
        
        if self.debug and clues:
            print("Sample clues:")
            for clue in clues[:5]:
                print(f"  {clue}")
        
        return clues
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean common OCR errors
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Common OCR substitutions
        replacements = {
            '|': 'I',
            '0': 'O',  # Context-dependent, but often correct in words
            '§': 'S',
            '€': 'C',
            '@': 'a',
        }
        
        cleaned = text
        for old, new in replacements.items():
            # Only replace in word contexts
            cleaned = re.sub(f'{old}(?=[a-z])', new, cleaned)
        
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def extract_clues(self) -> List[str]:
        """
        Main method to extract clues
        
        Returns:
            List of clue strings
        """
        # Extract text
        raw_text = self._extract_text_regions()
        
        # Parse clues
        self.clues = self._parse_clues(raw_text)
        
        return self.clues
    
    def save_grid(self, filename: str = "grid.txt"):
        """
        Save grid to file
        
        Args:
            filename: Output filename
        """
        if self.grid is None:
            print("Warning: No grid to save. Run extract_grid() first.")
            return
        
        print(f"Saving grid to {filename}...")
        
        with open(filename, 'w') as f:
            for row in self.grid:
                f.write(''.join(row) + '\n')
        
        print(f"Grid saved successfully ({len(self.grid)}x{len(self.grid[0])} cells)")
    
    def save_clues(self, filename: str = "word.txt"):
        """
        Save clues to file
        
        Args:
            filename: Output filename
        """
        if not self.clues:
            print("Warning: No clues to save. Run extract_clues() first.")
            return
        
        print(f"Saving clues to {filename}...")
        
        with open(filename, 'w') as f:
            for clue in self.clues:
                f.write(clue + '\n')
        
        print(f"Clues saved successfully ({len(self.clues)} clues)")
    
    def save_results(self, grid_file: str = "grid.txt", clues_file: str = "word.txt"):
        """
        Extract and save both grid and clues
        
        Args:
            grid_file: Output filename for grid
            clues_file: Output filename for clues
        """
        print("=" * 60)
        print("CROSSWORD EXTRACTION STARTED")
        print("=" * 60)
        
        try:
            # Extract grid
            self.extract_grid()
            self.save_grid(grid_file)
            
            print()
            
            # Extract clues
            self.extract_clues()
            self.save_clues(clues_file)
            
            print()
            print("=" * 60)
            print("EXTRACTION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Grid saved to: {grid_file}")
            print(f"Clues saved to: {clues_file}")
            
        except Exception as e:
            print()
            print("=" * 60)
            print("ERROR DURING EXTRACTION")
            print("=" * 60)
            print(f"Error: {str(e)}")
            raise


def main():
    """Run crossword extraction with optional size parameters"""
    parser = argparse.ArgumentParser(description="Crossword Puzzle Image Extractor")
    parser.add_argument("--image", type=str, required=True, help="Path to crossword image")
    parser.add_argument("--rows", type=int, help="Number of rows in grid (default: auto-detect)")
    parser.add_argument("--cols", type=int, help="Number of columns in grid (default: auto-detect)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results")

    args = parser.parse_args()

    extractor = CrosswordExtractor(args.rows, args.cols, args.image, debug=args.debug)

    extractor.save_results(
        os.path.join(args.outdir, "grid.txt"),
        os.path.join(args.outdir, "word.txt")
    )


if __name__ == "__main__":
    main()