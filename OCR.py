"""
Crossword Puzzle Image Extractor
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
        grid_img = self.image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        if self.debug:
            cv2.imwrite("debug_grid_binary.png", binary)
        
        # Detect grid lines using Hough transform
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w//4, maxLineGap=10)
        
        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:  # Horizontal
                    h_lines.append((min(y1, y2), max(y1, y2)))
                elif 80 < angle < 100:  # Vertical
                    v_lines.append((min(x1, x2), max(x1, x2)))
        
        # Cluster lines to find grid structure
        h_lines = sorted(set([y[0] for y in h_lines]))
        v_lines = sorted(set([x[0] for x in v_lines]))
        
        # Estimate grid dimensions
        if len(h_lines) > 1 and len(v_lines) > 1:
            rows = len(h_lines) - 1
            cols = len(v_lines) - 1
        else:
            # Fallback: try to estimate based on image size
            cell_size_w = w // getattr(self, "cols", self.cols)
            cell_size_h = h // getattr(self, "rows", self.rows)
            rows = getattr(self, "rows", self.rows)
            cols = getattr(self, "cols", self.cols)

            h_lines = [i * cell_size_h for i in range(rows + 1)]
            v_lines = [i * cell_size_w for i in range(cols + 1)]

        
        print(f"Detected grid dimensions: {rows}x{cols}")
        
        # Extract cell states (filled or empty)
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                # Get cell boundaries
                y1 = h_lines[r] + 2  # Small padding to avoid borders
                y2 = h_lines[r + 1] - 2
                x1 = v_lines[c] + 2
                x2 = v_lines[c + 1] - 2
                
                # Extract cell
                cell = binary[y1:y2, x1:x2]
                
                if cell.size == 0:
                    row.append('_')
                    continue
                
                # Calculate darkness ratio
                darkness = np.sum(cell == 255) / cell.size
                
                # Threshold: if more than 60% dark, it's filled
                if darkness > 0.6:
                    row.append('X')
                else:
                    row.append('_')
            
            grid.append(row)
        
        if self.debug:
            print("Grid preview:")
            for row in grid:
                print(''.join(row))
        
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
    parser.add_argument("--rows", type=int, help="Number of rows in grid (default: 15)")
    parser.add_argument("--cols", type=int, help="Number of columns in grid (default: 15)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results")

    args = parser.parse_args()

    # extractor = CrosswordExtractor(args.image, debug=args.debug)
    extractor = CrosswordExtractor(args.rows, args.cols, args.image, debug=args.debug)


    # Pass dynamic size to grid extraction
    extractor.save_results(
        os.path.join(args.outdir, "grid.txt"),
        os.path.join(args.outdir, "word.txt")
    )


if __name__ == "__main__":
    main()