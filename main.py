import argparse
import os


from src.crossword_solver import entry
from src.ocr import CrosswordExtractor


def main():
    parser = argparse.ArgumentParser(description="Crossword Puzzle Image Extractor")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to crossword image"
    )
    parser.add_argument(
        "--rows", type=int, help="Number of rows in grid (default: auto-detect)"
    )
    parser.add_argument(
        "--cols", type=int, help="Number of columns in grid (default: auto-detect)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--outdir", type=str, default=".", help="Output directory for results"
    )
    parser.add_argument(
        "--nogui", default=False, action="store_true", help="Run without GUI display"
    )
    parser.add_argument(
        "--wordfile", type=str, help="Path to word list file for solver"
    )

    args = parser.parse_args()

    extractor = CrosswordExtractor(args.rows, args.cols, args.image, debug=args.debug)

    gridfile = os.path.join(args.outdir, "grid.txt")
    wordfile = os.path.join(args.outdir, "word.txt")
    extractor.save_results(gridfile, wordfile)

    nogui = args.nogui
    entry(gridfile, wordfile, nogui)


if __name__ == "__main__":
    main()
