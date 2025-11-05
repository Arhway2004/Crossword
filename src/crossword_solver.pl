:- dynamic slot/5.
:- dynamic word/1.
:- dynamic placement/2.
:- dynamic constraint/2.
:- dynamic generated_word/1.  % Track auto-generated words

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
% WORD GENERATION - Auto-generate missing words
% ============================================================================

% Generate realistic English-like words of specific length
generate_word(Length, Word) :-
    Length >= 3,
    Length =< 10,
    generate_word_chars(Length, Chars),
    atom_chars(Word, Chars).

% Common letter patterns for realistic words
consonant(C) :- member(C, [b,c,d,f,g,h,j,k,l,m,n,p,r,s,t,v,w,y,z]).
vowel(V) :- member(V, [a,e,i,o,u]).
common_letter(L) :- member(L, [a,e,i,o,u,r,s,t,n,l]).

generate_word_chars(1, [C]) :- common_letter(C).
generate_word_chars(Len, [C|Rest]) :-
    Len > 1,
    (Len mod 2 =:= 0 -> consonant(C) ; vowel(C)),
    Len1 is Len - 1,
    generate_word_chars_alt(Len1, Rest, C).

generate_word_chars_alt(1, [L], Prev) :- 
    (consonant(Prev) -> vowel(L) ; common_letter(L)).
generate_word_chars_alt(Len, [L|Rest], Prev) :-
    Len > 1,
    (consonant(Prev) -> vowel(L) ; consonant(L)),
    Len1 is Len - 1,
    generate_word_chars_alt(Len1, Rest, L).

% Generate N unique words of specific length
generate_words(Length, Count, Words) :-
    findall(W,
        (between(1, Count, _),
         generate_word(Length, W),
         \+ word(W),
         \+ generated_word(W)),
        AllWords),
    sort(AllWords, Sorted),
    length(Needed, Count),
    append(Needed, _, Sorted),
    Needed = Words.

% Auto-supplement word list based on grid requirements
supplement_words :-
    % Analyze what we need
    findall(Len, slot(_, _, _, _, Len), AllLengths),
    count_by_length(AllLengths, RequiredCounts),
    
    % For each required length, ensure we have enough words
    forall(
        member(Len-RequiredCount, RequiredCounts),
        ensure_words_for_length(Len, RequiredCount)
    ).

count_by_length([], []).
count_by_length([L|Rest], Counts) :-
    count_by_length(Rest, RestCounts),
    (member(L-C, RestCounts) ->
        C1 is C + 1,
        select(L-C, RestCounts, L-C1, Counts)
    ;   Counts = [L-1|RestCounts]
    ).

ensure_words_for_length(Length, Required) :-
    findall(W, (word(W), atom_length(W, Length)), Existing),
    length(Existing, ExistingCount),
    (ExistingCount < Required ->
        Needed is Required - ExistingCount + 2,  % Add 2 extra for flexibility
        format('  Generating ~w words of length ~w...~n', [Needed, Length]),
        generate_words(Length, Needed, NewWords),
        forall(member(W, NewWords),
               (atom_string(W, WStr),
                string_upper(WStr, WUpper),
                atom_string(WAtom, WUpper),
                assertz(word(WAtom)),
                assertz(generated_word(WAtom))))
    ;   true).

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
    
    format('~n╔════════════════════════════════════╗~n'),
    format('║   CROSSWORD SOLVER - FIXED        ║~n'),
    format('╚════════════════════════════════════╝~n'),
    
    % Analyze and supplement words
    format('~nAnalyzing word requirements...~n'),
    supplement_words,
    
    % Show statistics
    findall(L, slot(_, _, _, _, L), Lengths),
    sort(Lengths, UniqueLengths),
    format('~nSlot lengths: ~w~n', [UniqueLengths]),
    forall(
        member(Len, UniqueLengths),
        (findall(S, slot(S, _, _, _, Len), Slots),
         length(Slots, SC),
         findall(W, (word(W), atom_length(W, Len)), Words),
         length(Words, WC),
         format('  Length ~w: ~w slots, ~w words~n', [Len, SC, WC]))
    ),
    
    % Solve
    format('~nSolving...~n'),
    findall(SlotID, slot(SlotID, _, _, _, _), AllSlots),
    
    (solve_backtrack(AllSlots, Solution) ->
        format('~n✓ SOLVED!~n~n'),
        print_solution(Solution),
        format('~nSolution: ~w~n', [Solution])
    ;   format('~n✗ No solution found~n'),
        analyze_failure,
        fail
    ).

print_solution([]).
print_solution([(Slot, Word)|Rest]) :-
    slot(Slot, Dir, R, C, Len),
    format('Slot ~w (~w at ~w,~w len ~w): ~w~n', [Slot, Dir, R, C, Len, Word]),
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
            (Count =:= 0 -> format('    ⚠️  ZERO candidates!~n') ; true))).

% ============================================================================
% PYTHON INTERFACE
% ============================================================================

solve_crossword(SlotOrder, Solution) :-
    clear_placements,
    supplement_words,  % Ensure we have enough words
    solve_backtrack(SlotOrder, Solution).