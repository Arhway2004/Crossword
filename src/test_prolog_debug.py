#!/usr/bin/env python3
"""
Debug script to test Prolog solver without GUI
"""

# Check if pyswip and SWI-Prolog are available
try:
    from pyswip import Prolog

    PROLOG_AVAILABLE = True
    print("✓ pyswip imported successfully")
except ImportError:
    print("✗ pyswip not installed")
    exit(1)
except Exception as e:
    print(f"✗ SWI-Prolog error: {e}")
    exit(1)


def test_prolog_solver():
    """Test the Prolog solver step by step"""

    print("\n=== PROLOG CROSSWORD SOLVER DEBUG ===\n")

    # Initialize Prolog
    try:
        prolog = Prolog()
        print("✓ Prolog instance created")
    except Exception as e:
        print(f"✗ Failed to create Prolog instance: {e}")
        return

    # Load the Prolog file
    try:
        prolog.consult("crossword_solver.pl")
        print("✓ crossword_solver.pl loaded")
    except Exception as e:
        print(f"✗ Failed to load crossword_solver.pl: {e}")
        return

    # Test basic Prolog functionality
    print("\n--- Testing basic Prolog queries ---")

    # Clear any existing data
    list(prolog.query("retractall(slot(_, _, _, _, _))"))
    list(prolog.query("retractall(word(_))"))
    list(prolog.query("retractall(placement(_, _))"))
    list(prolog.query("retractall(constraint(_, _))"))
    print("✓ Cleared existing data")

    # Add some test slots
    test_slots = [(0, "across", 0, 0, 5), (1, "down", 0, 1, 3), (2, "across", 2, 0, 3)]

    print("\n--- Adding test slots ---")
    for slot_id, direction, row, col, length in test_slots:
        query = f"assertz(slot({slot_id}, {direction}, {row}, {col}, {length}))"
        list(prolog.query(query))
        print(f"  Added slot {slot_id}: {direction} at ({row},{col}) len={length}")

    # Add some test words
    test_words = ["hello", "cat", "dog", "rat", "box"]
    print("\n--- Adding test words ---")
    for word in test_words:
        query = f"assertz(word({word}))"
        list(prolog.query(query))
        print(f"  Added word: {word}")

    # Test slot queries
    print("\n--- Testing slot queries ---")
    slots = list(prolog.query("slot(ID, Dir, R, C, Len)"))
    for slot in slots:
        print(
            f"  Slot {slot['ID']}: {slot['Dir']} at ({slot['R']},{slot['C']}) len={slot['Len']}"
        )

    # Test word queries
    print("\n--- Testing word queries ---")
    words = list(prolog.query("word(W)"))
    word_list = [w["W"] for w in words]
    print(f"  Words: {word_list}")

    # Test can_place_word
    print("\n--- Testing can_place_word ---")
    for word in ["hello", "cat", "dog"]:
        for slot_id in [0, 1, 2]:
            results = list(prolog.query(f"can_place_word({word}, {slot_id})"))
            can_place = len(results) > 0
            print(f"  can_place_word({word}, {slot_id}): {can_place}")

    # Test confidence calculation
    print("\n--- Testing confidence calculation ---")
    for word in ["hello", "cat"]:
        for slot_id in [0, 1]:
            try:
                results = list(prolog.query(f"confidence({slot_id}, {word}, Score)"))
                if results:
                    score = results[0]["Score"]
                    print(f"  confidence({slot_id}, {word}): {score}")
                else:
                    print(f"  confidence({slot_id}, {word}): failed")
            except Exception as e:
                print(f"  confidence({slot_id}, {word}): error - {e}")

    # Test choose_word_with_confidence
    print("\n--- Testing choose_word_with_confidence ---")
    for slot_id in [0, 1, 2]:
        try:
            results = list(
                prolog.query(f"choose_word_with_confidence({slot_id}, Word)")
            )
            if results:
                words = [r["Word"] for r in results]
                print(f"  Slot {slot_id} candidates: {words}")
            else:
                print(f"  Slot {slot_id}: no candidates")
        except Exception as e:
            print(f"  Slot {slot_id}: error - {e}")

    # Test simple solve
    print("\n--- Testing simple solve ---")
    try:
        # Clear placements
        list(prolog.query("clear_placements"))

        # Try to solve with just slot 2 (length 3)
        results = list(prolog.query("solve_crossword([2], Solution)"))
        if results:
            print(f"  Simple solve result: {results[0]['Solution']}")
        else:
            print("  Simple solve: no solution")

        # Try to solve with slots 1 and 2
        list(prolog.query("clear_placements"))
        results = list(prolog.query("solve_crossword([1, 2], Solution)"))
        if results:
            print(f"  Two-slot solve result: {results[0]['Solution']}")
        else:
            print("  Two-slot solve: no solution")

    except Exception as e:
        print(f"  Solve test error: {e}")

    print("\n--- Testing with real crossword data ---")

    # Load real grid data
    try:
        # Clear existing data
        list(prolog.query("retractall(slot(_, _, _, _, _))"))
        list(prolog.query("retractall(word(_))"))

        # Read and load real slots from grid.txt
        from crossword_solver import GridReader

        grid_reader = GridReader("grid.txt")
        slots = grid_reader.find_slots()

        print(f"✓ Found {len(slots)} real slots")

        # Add slots to Prolog
        for slot in slots:
            slot_id = slot["id"]
            direction = slot["direction"]
            row = slot["row"]
            col = slot["col"]
            length = slot["length"]
            query = f"assertz(slot({slot_id}, {direction}, {row}, {col}, {length}))"
            list(prolog.query(query))

        # Load real words
        with open("word.txt", "r") as f:
            real_words = [line.strip().lower() for line in f if line.strip()]

        print(f"✓ Loading {len(real_words)} real words")
        for word in real_words:
            query = f"assertz(word({word}))"
            list(prolog.query(query))

        # Test with first few slots
        test_slot_ids = [s["id"] for s in slots[:3]]
        print(f"\n--- Testing solve with real data (slots {test_slot_ids}) ---")

        list(prolog.query("clear_placements"))
        query = f"solve_crossword({test_slot_ids}, Solution)"
        print(f"Query: {query}")

        results = list(prolog.query(query))
        if results:
            solution = results[0]["Solution"]
            print(f"✓ Solution found: {solution}")

            # Parse solution
            print("\n--- Parsing solution ---")
            if hasattr(solution, "__iter__"):
                for i, item in enumerate(solution):
                    print(f"  Item {i}: {item} (type: {type(item)})")
                    if hasattr(item, "functor"):
                        print(f"    Functor: {item.functor}")
                        if hasattr(item, "args"):
                            print(f"    Args: {item.args}")
            else:
                print(f"  Solution is not iterable: {solution}")
        else:
            print("✗ No solution found with real data")

    except Exception as e:
        print(f"✗ Real data test error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_prolog_solver()
