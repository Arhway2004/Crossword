class PythonCrosswordSolver:
    """Python-based solver with confidence heuristic"""

    def __init__(self):
        self.words = []
        self.constraints = {}
        self.slots = []
        self.step_callback = None
        self.current_placements = {}  # Track current placements for visualization
        print("✓ Solver choice: Using Python solver with confidence heuristic")

    def set_step_callback(self, callback):
        """Set callback function to visualize each step"""
        self.step_callback = callback

    def load_slots(self, slots):
        self.slots = slots

    def load_words(self, filename):
        loaded = False

        if filename == "":
            raise ValueError("No wordlist filename provided")

        try:
            with open(filename, "r") as f:
                self.words = []
                for line in f:
                    word = line.strip().lower()
                    if word:
                        word = word.replace("'", "_")
                        self.words.append(word)
                print(f"✓ Loaded {len(self.words)} words from {filename}")
                loaded = True
        except FileNotFoundError:
            print(f"ERROR: {filename} not found!")

    def load_constraints(self, filename, slots):
        try:
            with open(filename, "r") as f:
                count = 0
                mismatches = 0
                for line_idx, line in enumerate(f):
                    line = line.strip().lower()
                    if not line:
                        continue

                    if ":" in line:
                        parts = line.split(":", 1)
                        try:
                            slot_id = int(parts[0].strip())
                            pattern = parts[1].strip()
                        except ValueError:
                            print(
                                f"  Warning: Invalid slot_id in line {line_idx + 1}, skipping"
                            )
                            continue
                    else:
                        slot_id = line_idx
                        pattern = line

                    slot_match = None
                    for slot in slots:
                        if slot["id"] == slot_id:
                            slot_match = slot
                            break

                    if slot_match:
                        if len(pattern) != slot_match["length"]:
                            print(
                                f"  Warning: Constraint for slot {slot_id} has length {len(pattern)}, but slot length is {slot_match['length']} - ignoring"
                            )
                            mismatches += 1
                            continue

                        self.constraints[slot_id] = pattern
                        print(f"  Constraint slot {slot_id}: {pattern}")
                        count += 1
                    else:
                        print(
                            f"  Warning: Constraint references non-existent slot {slot_id}, ignoring"
                        )

                if count > 0:
                    print(f"✓ Loaded {count} constraints from {filename}")
                if mismatches > 0:
                    print(
                        f"  ({mismatches} constraint(s) ignored due to length mismatch)"
                    )
        except FileNotFoundError:
            print(f"  No {filename} found (constraints are optional)")

    def matches_pattern(self, word, pattern):
        """Check if word matches pattern (e.g., dog*** matches doctor)"""
        if len(word) != len(pattern):
            return False
        for w, p in zip(word, pattern):
            if p != "*" and p != w:
                return False
        return True

    def get_intersection(self, slot1, slot2):
        """Get intersection point between two slots if it exists"""
        if slot1["direction"] == slot2["direction"]:
            return None

        if slot1["direction"] == "across" and slot2["direction"] == "down":
            if (
                slot2["row"] <= slot1["row"] < slot2["row"] + slot2["length"]
                and slot1["col"] <= slot2["col"] < slot1["col"] + slot1["length"]
            ):
                pos1 = slot2["col"] - slot1["col"]
                pos2 = slot1["row"] - slot2["row"]
                return (pos1, pos2)

        elif slot1["direction"] == "down" and slot2["direction"] == "across":
            if (
                slot1["row"] <= slot2["row"] < slot1["row"] + slot1["length"]
                and slot2["col"] <= slot1["col"] < slot2["col"] + slot2["length"]
            ):
                pos1 = slot2["row"] - slot1["row"]
                pos2 = slot1["col"] - slot2["col"]
                return (pos1, pos2)

        return None

    def check_intersection(self, placements):
        """Check if current placements have valid intersections"""
        slot_ids = list(placements.keys())

        for i, slot1_id in enumerate(slot_ids):
            for slot2_id in slot_ids[i + 1 :]:
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

    def calculate_confidence(self, slot_id, word, placements, used_words):
        """
        Calculate confidence score for placing a word in a slot.
        Score = number of intersecting slots that still have valid candidates.
        """
        slot = self.slots[slot_id]
        score = 0

        for other_slot in self.slots:
            other_id = other_slot["id"]

            if other_id == slot_id or other_id in placements:
                continue

            if slot["direction"] == other_slot["direction"]:
                continue

            intersection = self.get_intersection(slot, other_slot)
            if not intersection:
                continue

            pos_in_slot, pos_in_other = intersection
            letter = word[pos_in_slot]

            candidate_exists = False
            for candidate_word in self.words:
                if len(candidate_word) != other_slot["length"]:
                    continue

                if candidate_word in used_words:
                    continue

                if candidate_word[pos_in_other] != letter:
                    continue

                if other_id in self.constraints:
                    if not self.matches_pattern(
                        candidate_word, self.constraints[other_id]
                    ):
                        continue

                # Check if this candidate would be compatible with existing placements
                test_placements = placements.copy()
                test_placements[other_id] = candidate_word
                if not self.check_intersection(test_placements):
                    continue

                candidate_exists = True
                break

            if candidate_exists:
                score += 1

        return score

    def solve(self, slot_ids):
        """Backtracking solver with confidence-based heuristic"""
        placements = {}
        used_words = set()
        self.current_placements = placements

        def backtrack(slot_idx):
            if slot_idx >= len(slot_ids):
                return True

            slot_id = slot_ids[slot_idx]
            slot = self.slots[slot_id]

            # Collect all valid candidates with their confidence scores
            candidates = []
            for word in self.words:
                if len(word) != slot["length"]:
                    continue

                if word in used_words:
                    continue

                if slot_id in self.constraints:
                    if not self.matches_pattern(word, self.constraints[slot_id]):
                        continue

                confidence = self.calculate_confidence(
                    slot_id, word, placements, used_words
                )
                candidates.append((confidence, word))

            # Sort by confidence (highest first)
            candidates.sort(reverse=True, key=lambda x: x[0])

            # Try candidates in order of confidence
            for confidence, word in candidates:
                # FIX: Only prune zero-confidence if there are intersections
                if confidence == 0 and len(candidates) > 1:
                    has_intersections = any(
                        self.get_intersection(slot, self.slots[other_id]) is not None
                        for other_id in slot_ids
                        if other_id != slot_id and other_id not in placements
                    )
                    if has_intersections:
                        continue

                placements[slot_id] = word
                used_words.add(word)

                if self.check_intersection(placements):
                    if self.step_callback:
                        self.step_callback(slot_id, word, slot, True, placements.copy())

                    if backtrack(slot_idx + 1):
                        return True

                if self.step_callback:
                    self.step_callback(slot_id, word, slot, False, placements.copy())

                del placements[slot_id]
                used_words.remove(word)

            return False

        if backtrack(0):
            return [(sid, placements[sid]) for sid in slot_ids if sid in placements]
        return None
