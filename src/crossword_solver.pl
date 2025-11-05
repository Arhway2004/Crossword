:- dynamic slot/5.
:- dynamic word/1.
:- dynamic placement/2.
:- dynamic constraint/2.

% ============================================================================
% INITIALIZATION
% ============================================================================

clear_all :-
    retractall(slot(_, _, _, _, _)),
    retractall(word(_)),
    retractall(placement(_, _)),
    retractall(constraint(_, _)),
    retractall(generated_word(_)).

clear_placements :- 
    retractall(placement(_, _)).

% ============================================================================
% PATTERN MATCHING
% ============================================================================

matches_pattern(Word, Pattern) :-
    atom_length(Word, WLen),
    atom_length(Pattern, PLen),
    WLen =:= PLen,
    atom_chars(Word, WordChars),
    atom_chars(Pattern, PatternChars),
    match_chars(WordChars, PatternChars).

match_chars([], []).
match_chars([_|Ws], ['*'|Ps]) :- match_chars(Ws, Ps).
match_chars([C|Ws], [C|Ps]) :- match_chars(Ws, Ps).

letter_at(Word, Pos, Letter) :-
    atom_chars(Word, Chars),
    nth0(Pos, Chars, Letter).

% ============================================================================
% INTERSECTION CHECKING - RELAXED
% ============================================================================

find_intersection(R1, C1, across, R2, C2, down, LenA, LenD, PosInA, PosInD) :-
    R2 =< R1, R1 < R2 + LenD,
    C1 =< C2, C2 < C1 + LenA,
    PosInA is C2 - C1,
    PosInD is R1 - R2.

find_intersection(R1, C1, down, R2, C2, across, LenD, LenA, PosInD, PosInA) :-
    R1 =< R2, R2 < R1 + LenD,
    C2 =< C1, C1 < C2 + LenA,
    PosInD is R2 - R1,
    PosInA is C1 - C2.

% Check if two placements are compatible
placements_compatible(Slot1, Word1, Slot2, Word2) :-
    slot(Slot1, Dir1, R1, C1, Len1),
    slot(Slot2, Dir2, R2, C2, Len2),
    (Dir1 \= Dir2 ->
        (find_intersection(R1, C1, Dir1, R2, C2, Dir2, Len1, Len2, Pos1, Pos2) ->
            (letter_at(Word1, Pos1, L1),
             letter_at(Word2, Pos2, L2),
             L1 = L2)
        ; true)  % No intersection
    ; true).  % Same direction

% Check word is compatible with all PLACED words
check_placed_compatible(SlotID, Word) :-
    forall(
        (placement(OtherSlot, OtherWord), OtherSlot \= SlotID),
        placements_compatible(SlotID, Word, OtherSlot, OtherWord)
    ).

% ============================================================================
% DOMAIN GENERATION - OPTIMIZED
% ============================================================================

get_slot_domain(SlotID, Domain) :-
    slot(SlotID, _, _, _, Len),
    findall(W,
        ( word(W),
          atom_length(W, Len),
          \+ placement(_, W),  % Word not used
          (constraint(SlotID, Pattern) -> matches_pattern(W, Pattern) ; true),
          check_placed_compatible(SlotID, W)
        ),
        Domain).

% ============================================================================
% HEURISTICS
% ============================================================================

slot_candidate_count(SlotID, Count) :-
    get_slot_domain(SlotID, Domain),
    length(Domain, Count).

% MRV: Choose slot with fewest candidates
select_slot_mrv([SlotID|Rest], Selected, Remaining) :-
    findall(Count-S,
            (member(S, [SlotID|Rest]), 
             slot_candidate_count(S, Count),
             Count > 0),
            Pairs),
    (Pairs = [] -> fail  % No valid slots
    ; sort(Pairs, [_-Selected|_]),
      select(Selected, [SlotID|Rest], Remaining)
    ).

% ============================================================================
% BACKTRACKING SOLVER WITH FORWARD CHECKING
% ============================================================================

solve_backtrack([], []).
solve_backtrack(Slots, [(SlotID, Word)|Rest]) :-
    select_slot_mrv(Slots, SlotID, Remaining),
    get_slot_domain(SlotID, Domain),
    Domain \= [],  % Fail if no candidates
    member(Word, Domain),
    assertz(placement(SlotID, Word)),
    
    % Forward check: ensure remaining slots still have options
    (check_forward_viable(Remaining) ->
        (solve_backtrack(Remaining, Rest) -> true
        ; retract(placement(SlotID, Word)), fail)
    ; retract(placement(SlotID, Word)), fail
    ).

% Check if all remaining slots have at least one candidate
check_forward_viable([]).
check_forward_viable([SlotID|Rest]) :-
    get_slot_domain(SlotID, Domain),
    Domain \= [],
    check_forward_viable(Rest).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

solve :-
    clear_placements,
    
    findall(L, slot(_, _, _, _, L), Lengths),
    sort(Lengths, UniqueLengths),
    forall(
        member(Len, UniqueLengths),
        (findall(S, slot(S, _, _, _, Len), Slots),
         length(Slots, SC),
         findall(W, (word(W), atom_length(W, Len)), Words),
         length(Words, WC)
    )), 
    
    % Solve
    findall(SlotID, slot(SlotID, _, _, _, _), AllSlots),
    
    (solve_backtrack(AllSlots, Solution) ->
        print_solution(Solution)
        ; analyze_failure,
          fail
    ).

print_solution([]).
print_solution([(Slot, Word)|Rest]) :-
    slot(Slot, Dir, R, C, Len),
    print_solution(Rest).

analyze_failure :-
    format('~nFailure analysis:~n'),
    findall(C-S,
        (slot(S, _, _, _, _),
         \+ placement(S, _),
         slot_candidate_count(S, C)),
        Counts),
    sort(Counts, Sorted),
    forall(member(Count-Slot, Sorted),
           (format('  Slot ~w: ~w candidates~n', [Slot, Count]),
            (Count =:= 0 -> false ; true))).

% ============================================================================
% PYTHON INTERFACE
% ============================================================================

solve_crossword(SlotOrder, Solution) :-
    clear_placements,
    solve_backtrack(SlotOrder, Solution).