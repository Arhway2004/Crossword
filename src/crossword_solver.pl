:- dynamic slot/5.
:- dynamic word/1.
:- dynamic placement/2.
:- dynamic constraint/2.

% Check if word matches pattern constraint
matches_pattern(Word, Pattern) :-
    atom_chars(Word, WordChars),
    atom_chars(Pattern, PatternChars),
    match_chars(WordChars, PatternChars).

match_chars([], []).
match_chars([_|Ws], ['*'|Ps]) :- match_chars(Ws, Ps).
match_chars([C|Ws], [C|Ps]) :- match_chars(Ws, Ps).

% Check if word can be placed in slot
can_place_word(Word, SlotID) :-
    word(Word),
    slot(SlotID, _, _, _, Length),
    atom_length(Word, Length),
    \+ placement(_, Word),
    (constraint(SlotID, Pattern) -> matches_pattern(Word, Pattern) ; true).

% Get letter at position in word (0-indexed)
letter_at(Word, Pos, Letter) :-
    atom_chars(Word, Chars),
    nth0(Pos, Chars, Letter).

% Calculate intersection between across and down slots
% Returns positions in each word where they intersect
find_intersection(R1, C1, across, R2, C2, down, LenA, LenD, PosInA, PosInD) :-
    R2 =< R1,
    R1 < R2 + LenD,
    C1 =< C2,
    C2 < C1 + LenA,
    PosInA is C2 - C1,
    PosInD is R1 - R2.

find_intersection(R1, C1, down, R2, C2, across, LenD, LenA, PosInD, PosInA) :-
    R1 =< R2,
    R2 < R1 + LenD,
    C2 =< C1,
    C1 < C2 + LenA,
    PosInD is R2 - R1,
    PosInA is C1 - C2.

% Check if two placements are compatible at their intersection
check_intersection(Slot1, Word1, Slot2, Word2) :-
    slot(Slot1, Dir1, R1, C1, Len1),
    slot(Slot2, Dir2, R2, C2, Len2),
    (Dir1 \= Dir2 ->
        (find_intersection(R1, C1, Dir1, R2, C2, Dir2, Len1, Len2, Pos1, Pos2) ->
            letter_at(Word1, Pos1, Letter),
            letter_at(Word2, Pos2, Letter)
        ; true)
    ; true).

% Check all existing placements are compatible with this new placement
check_all_constraints(SlotID, Word) :-
    forall(
        placement(OtherSlot, OtherWord),
        (OtherSlot \= SlotID -> check_intersection(SlotID, Word, OtherSlot, OtherWord) ; true)
    ).

% ============================
% FIXED CONFIDENCE-BASED HEURISTIC
% ============================

% neighbors(+SlotA, -SlotB, -PosInA, -PosInB)
% Find all opposite-direction slots that intersect with SlotA
neighbors(SlotA, SlotB, PosInA, PosInB) :-
    slot(SlotA, DirA, R1, C1, LenA),
    slot(SlotB, DirB, R2, C2, LenB),
    SlotA \= SlotB,
    DirA \= DirB,
    ( DirA = across, DirB = down ->
        find_intersection(R1, C1, across, R2, C2, down, LenA, LenB, PosInA, PosInB)
    ; DirA = down, DirB = across ->
        find_intersection(R1, C1, down, R2, C2, across, LenA, LenB, PosInA, PosInB)
    ).

% FIX: Check if there exists a valid candidate for the other slot
% that is compatible with ALL existing placements
candidate_exists_for_other(OtherSlot, PosInOther, Letter) :-
    slot(OtherSlot, _, _, _, LenO),
    word(W),
    atom_length(W, LenO),
    letter_at(W, PosInOther, Letter),
    \+ placement(_, W),
    % Check constraint if exists
    ( constraint(OtherSlot, Pattern) ->
        matches_pattern(W, Pattern)
    ; true ),
    % CRITICAL FIX: Verify this candidate is compatible with ALL existing placements
    forall(
        placement(ExistingSlot, ExistingWord),
        check_intersection(OtherSlot, W, ExistingSlot, ExistingWord)
    ).

% FIX: confidence now only counts unfilled neighboring slots
% Score = number of unfilled intersecting slots that have valid candidates
confidence(SlotID, Word, Score) :-
    % First verify this word doesn't conflict with existing placements
    check_all_constraints(SlotID, Word),
    % Count viable neighbor candidates (only for unfilled slots)
    findall(1,
        ( neighbors(SlotID, OtherSlot, PosInThis, PosInOther),
          \+ placement(OtherSlot, _),  % Only check unfilled slots
          letter_at(Word, PosInThis, Letter),
          candidate_exists_for_other(OtherSlot, PosInOther, Letter)
        ),
        Ones),
    length(Ones, Score).

% choose_word_with_confidence(+SlotID, -Word)
% Generate words that fit SlotID, ordered by descending confidence
choose_word_with_confidence(SlotID, Word) :-
    % Collect all (Conf-Word) candidates
    findall(Conf-Word,
        ( word(Word),
          slot(SlotID, _, _, _, Len),
          atom_length(Word, Len),
          \+ placement(_, Word),
          ( constraint(SlotID, Pattern) -> matches_pattern(Word, Pattern) ; true ),
          confidence(SlotID, Word, Conf)
        ),
        Pairs),
    % Sort by descending confidence (highest first)
    sort(0, @>=, Pairs, Sorted),
    % Nondeterministically yield words
    member(_Conf-Word, Sorted).

% Main solving predicate with confidence heuristic
solve_crossword([], []).
solve_crossword([SlotID|Rest], [(SlotID, Word)|RestPlacements]) :-
    % Try words ordered by confidence
    choose_word_with_confidence(SlotID, Word),
    assertz(placement(SlotID, Word)),
    ( check_all_constraints(SlotID, Word),
      solve_crossword(Rest, RestPlacements)
    -> !
    ;  retract(placement(SlotID, Word)),
       fail
    ).

% Clear all placements
clear_placements :-
    retractall(placement(_, _)).

% Helper predicates for debugging
print_confidence(SlotID, Word, Conf) :-
    confidence(SlotID, Word, Conf),
    format('  Word: ~w, Confidence: ~d~n', [Word, Conf]).

show_candidates(SlotID) :-
    format('Candidates for slot ~d:~n', [SlotID]),
    forall(
        choose_word_with_confidence(SlotID, Word),
        print_confidence(SlotID, Word, _)
    ).