:- dynamic slot/5.
:- dynamic word/1.
:- dynamic placement/2.
:- dynamic constraint/2.
:- dynamic step/5.            % step(Id, Type, SlotID, Word, Note)
:- dynamic step_counter/1.

% ============================================================================
% INITIALIZATION
% ============================================================================

clear_all :-
    retractall(slot(_, _, _, _, _)),
    retractall(word(_)),
    retractall(placement(_, _)),
    retractall(constraint(_, _)),
    retractall(generated_word(_)),
    retractall(step(_,_,_,_,_)),
    retractall(step_counter(_)).

clear_placements :- 
    retractall(placement(_, _)).

% ----------------------------------------------------------------------------
% STEP LOGGING SUPPORT
% ----------------------------------------------------------------------------

clear_steps :-
    retractall(step(_,_,_,_,_)), retractall(step_counter(_)), assertz(step_counter(0)).
next_step_id(Id) :-
    ( retract(step_counter(C0)) -> true ; C0 = 0 ),
    C1 is C0 + 1,
    assertz(step_counter(C1)),
    Id = C1.

% Type is one of: start | select_slot | try | place | fail_forward | backtrack | solution
log_step(Type, SlotID, Word, Note) :-
    ( var(Word) -> WordAtom = '' ; WordAtom = Word ),
    next_step_id(Id),
    assertz(step(Id, Type, SlotID, WordAtom, Note)).

get_steps(Steps) :-
    findall(step(I,T,S,W,N), step(I,T,S,W,N), Steps).

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

solve_backtrack([], []) :-
    log_step(solution, -1, '', 'All slots assigned').
solve_backtrack(Slots, [(SlotID, Word)|Rest]) :-
    select_slot_mrv(Slots, SlotID, Remaining),
    log_step(select_slot, SlotID, '', ''),
    get_slot_domain(SlotID, Domain),
    Domain \= [],  % Fail if no candidates
    member(Word, Domain),
    log_step(try, SlotID, Word, ''),
    assertz(placement(SlotID, Word)),
    log_step(place, SlotID, Word, ''),
    
    % Forward check: ensure remaining slots still have options
    ( check_forward_viable(Remaining, Culprit) ->
        ( solve_backtrack(Remaining, Rest) -> true
        ;   retract(placement(SlotID, Word)),
            log_step(backtrack, SlotID, Word, 'Recursion failed'),
            fail
        )
    ;   retract(placement(SlotID, Word)),
        ( nonvar(Culprit)
        -> format(atom(Note), 'Forward check failed at slot ~w', [Culprit])
        ;  Note = 'Forward check failed'
        ),
        log_step(fail_forward, SlotID, Word, Note),
        fail
    ).

% Check if all remaining slots have at least one candidate
% Succeeds if all remaining slots have non-empty domains.
% On failure, binds Culprit to a slot whose domain is empty.
check_forward_viable(Slots, Culprit) :-
    ( empty_domain_slot(Slots, Culprit) -> fail ; true ).

empty_domain_slot([S|Rest], Culprit) :-
    get_slot_domain(S, D),
    ( D = [] -> Culprit = S
    ; empty_domain_slot(Rest, Culprit)
    ).
empty_domain_slot([], _) :- fail.

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

solve :-
    clear_placements,
    clear_steps,
    log_step(start, -1, '', 'Starting solve'),
    
    findall(L, slot(_, _, _, _, L), Lengths),
    sort(Lengths, UniqueLengths),
    forall(
        member(Len, UniqueLengths),
        (findall(S, slot(S, _, _, _, Len), Slots),
         length(Slots, _),
         findall(W, (word(W), atom_length(W, Len)), Words),
         length(Words, _)
    )), 
    
    % Solve
    findall(SlotID, slot(SlotID, _, _, _, _), AllSlots),
    
    ( solve_backtrack(AllSlots, Solution) ->
        print_solution(Solution)
    ;   log_step(no_solution, -1, '', 'No solution'),
        analyze_failure,
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
           format('  Slot ~w: ~w candidates~n', [Slot, Count])).

% ============================================================================
% PYTHON INTERFACE
% ============================================================================

solve_crossword(SlotOrder, Solution) :-
    clear_placements,
    solve_backtrack(SlotOrder, Solution).

% Solve with step capture for Python: returns Solution and Steps facts
trace_solve(SlotOrder, Solution, Steps) :-
    clear_placements,
    clear_steps,
    log_step(start, -1, '', 'Starting trace_solve'),
    ( solve_backtrack(SlotOrder, Solution) -> true
    ; log_step(no_solution, -1, '', 'No solution')
    ),
    get_steps(Steps).