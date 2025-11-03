% ============================================================================
% CROSSWORD PUZZLE SOLVER - PROLOG MODULE
% Full Backtracking with Intersection Detection
% ============================================================================

% Dynamic predicates for storing puzzle state
:- dynamic slot/5.        % slot(ID, Direction, Row, Col, Length)
:- dynamic word/1.        % word(Word)
:- dynamic placement/2.   % placement(SlotID, Word)
:- dynamic constraint/2.  % constraint(SlotID, Pattern)

% ============================================================================
% PATTERN MATCHING
% ============================================================================

% Check if word matches a pattern constraint
% Pattern uses '*' as wildcard, exact letters must match
% Example: 'dog***' matches 'doctor'
matches_pattern(Word, Pattern) :-
    atom_chars(Word, WordChars),
    atom_chars(Pattern, PatternChars),
    match_chars(WordChars, PatternChars).

% Recursive pattern matching
% Base case: empty lists match
match_chars([], []).

% Wildcard case: '*' matches any character
match_chars([_|Ws], ['*'|Ps]) :- 
    match_chars(Ws, Ps).

% Exact match case: characters must be identical
match_chars([C|Ws], [C|Ps]) :- 
    match_chars(Ws, Ps).

% ============================================================================
% WORD PLACEMENT VALIDATION
% ============================================================================

% Check if a word can be placed in a slot
can_place_word(Word, SlotID) :-
    word(Word),                              % Word exists in dictionary
    slot(SlotID, _, _, _, Length),           % Slot exists
    atom_length(Word, Length),               % Word fits slot length
    \+ placement(_, Word),                   % Word not already used
    % Check pattern constraint if one exists for this slot
    (constraint(SlotID, Pattern) -> matches_pattern(Word, Pattern) ; true).

% ============================================================================
% LETTER EXTRACTION
% ============================================================================

% Get the Nth letter (0-indexed) from a word
letter_at(Word, Pos, Letter) :-
    atom_chars(Word, Chars),
    nth0(Pos, Chars, Letter).

% ============================================================================
% INTERSECTION CALCULATION
% ============================================================================

% Calculate intersection position between ACROSS and DOWN slots
% Returns the position in each word where they intersect
find_intersection(R1, C1, across, R2, C2, down, Len1, Len2, PosInAcross, PosInDown) :-
    % Compute positions from formulas first
    PosInDown is R1 - R2,
    PosInAcross is C2 - C1,
    % Then check if they're within valid range
    Len2Minus1 is Len2 - 1,
    Len1Minus1 is Len1 - 1,
    between(0, Len2Minus1, PosInDown),
    between(0, Len1Minus1, PosInAcross).

% Calculate intersection position between DOWN and ACROSS slots
find_intersection(R1, C1, down, R2, C2, across, Len1, Len2, PosInDown, PosInAcross) :-
    % Compute positions from formulas first
    PosInDown is R2 - R1,
    PosInAcross is C1 - C2,
    % Then check if they're within valid range
    Len1Minus1 is Len1 - 1,
    Len2Minus1 is Len2 - 1,
    between(0, Len1Minus1, PosInDown),
    between(0, Len2Minus1, PosInAcross).

% ============================================================================
% INTERSECTION VALIDATION
% ============================================================================

% Check if two placed words intersect correctly
% Words must have matching letters at intersection points
check_intersection(Slot1, Word1, Slot2, Word2) :-
    slot(Slot1, Dir1, R1, C1, Len1),
    slot(Slot2, Dir2, R2, C2, Len2),
    % Only check if different directions (across vs down)
    (Dir1 \= Dir2 ->
        % Try to find intersection point
        (find_intersection(R1, C1, Dir1, R2, C2, Dir2, Len1, Len2, Pos1, Pos2) ->
            % Get letters at intersection positions
            letter_at(Word1, Pos1, Letter1),
            letter_at(Word2, Pos2, Letter2),
            % Letters must match
            Letter1 = Letter2
        ; true)  % No intersection found, that's OK
    ; true).     % Same direction, no intersection possible

% ============================================================================
% CONSTRAINT CHECKING
% ============================================================================

% Check all existing placements for conflicts with new placement
% This validates that the new word doesn't violate any intersection constraints
check_all_constraints(SlotID, Word) :-
    forall(
        placement(OtherSlotID, OtherWord),
        (OtherSlotID \= SlotID -> 
            check_intersection(SlotID, Word, OtherSlotID, OtherWord)
        ; true)
    ).

% ============================================================================
% MAIN SOLVING ALGORITHM - FULL BACKTRACKING
% ============================================================================

% Base case: no more slots to fill
solve_crossword([], []).

% Recursive case: try to place a word in the current slot
solve_crossword([SlotID|RestSlots], [(SlotID, Word)|RestPlacements]) :-
    % Find a valid word for this slot
    can_place_word(Word, SlotID),
    % Temporarily place the word
    assertz(placement(SlotID, Word)),
    % Check if this placement is compatible with all existing placements
    (check_all_constraints(SlotID, Word) ->
        % Valid placement, continue solving
        (solve_crossword(RestSlots, RestPlacements) ->
            % Solution found, cut to prevent unnecessary backtracking
            !
        ;
            % Failed deeper, retract and backtrack
            retract(placement(SlotID, Word)),
            fail
        )
    ;
        % Intersection check failed, retract and try next word
        retract(placement(SlotID, Word)),
        fail
    ).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

% Clear all placements (for multiple solve attempts)
clear_placements :-
    retractall(placement(_, _)).

% Get all current placements as a list
get_all_placements(Placements) :-
    findall((SlotID, Word), placement(SlotID, Word), Placements).

% ============================================================================
% USAGE EXAMPLE
% ============================================================================
% 
% To use this solver:
% 1. Load slots: assertz(slot(0, across, 0, 0, 3))
% 2. Load words: assertz(word(dog))
% 3. Load constraints (optional): assertz(constraint(0, 'dog***'))
% 4. Solve: solve_crossword([0,1,2], Solution)
% 
% ============================================================================