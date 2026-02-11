/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "movepick.h"

#include <cassert>
#include <limits>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "position.h"
#include "tune.h"

namespace Stockfish {

// Tactical observation theory bonuses (SPSA-tunable).
// When a theory fires, the move is promoted to a priority tier searched
// before ALL non-theory quiets.  The bonus value orders moves WITHIN the
// theory tier.  Zero disables that specific theory.
//
// The tier promotion constant (1 << 28) guarantees theory moves sort above
// any history/check/threat score.  SPSA tunes the per-theory bonus only.

// Quiet theories — move exploits a detected tactical pattern
int TactObs_PinPressure      = 250;   // move attacks a pinned enemy piece
int TactObs_DiscoveredAttack = 300;   // discoverer moves off the king ray
int TactObs_ForkWithKing     = 500;   // knight fork including king
int TactObs_Fork             = 350;   // knight fork on 2+ high-value pieces
int TactObs_BackRank         = 250;   // R/Q to back rank when enemy king is there

// Capture theory
int TactObs_PinnedCapture    = 300;   // capturing a pinned piece (can't recapture)

TUNE(SetRange(0, 1500), TactObs_PinPressure);
TUNE(SetRange(0, 1500), TactObs_DiscoveredAttack);
TUNE(SetRange(0, 1500), TactObs_ForkWithKing);
TUNE(SetRange(0, 1500), TactObs_Fork);
TUNE(SetRange(0, 1500), TactObs_BackRank);
TUNE(SetRange(0, 1500), TactObs_PinnedCapture);

namespace {

// Theory moves are promoted to this tier so they sort before all
// non-theory quiet moves, regardless of history scores.
constexpr int TheoryTier = 1 << 28;

enum Stages {
    // generate main search moves
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};


// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = pos.attacks_by<QUEEN>(~us) | threatByLesser[QUEEN];
    }

    // ── Pre-compute tactical observations (once per scoring call) ──
    // These reuse bitboards Stockfish already maintains — zero extra cost.
    [[maybe_unused]] Bitboard pinnedEnemy, discoverers;
    [[maybe_unused]] Square   enemyKingSq;
    [[maybe_unused]] Rank     backRank;
    [[maybe_unused]] bool     kingOnBackRank;
    [[maybe_unused]] Bitboard occ;

    if constexpr (Type == CAPTURES || Type == QUIETS)
    {
        pinnedEnemy    = pos.blockers_for_king(~us) & pos.pieces(~us);
        discoverers    = pos.blockers_for_king(~us) & pos.pieces(us);
        enemyKingSq   = pos.square<KING>(~us);
        backRank       = (us == WHITE) ? RANK_8 : RANK_1;
        kingOnBackRank = rank_of(enemyKingSq) == backRank;
        occ            = pos.pieces();
    }

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);
        const Piece     capturedPiece = pos.piece_on(to);

        if constexpr (Type == CAPTURES)
        {
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
                    + 7 * int(PieceValue[capturedPiece]);

            // ExploitPinTheory (capture): pinned piece can't escape or
            // recapture effectively — safe material gain.
            if (TactObs_PinnedCapture && (pinnedEnemy & to))
                m.value += TactObs_PinnedCapture;
        }

        else if constexpr (Type == QUIETS)
        {
            // ── Standard history scoring (unchanged) ──
            m.value = 2 * (*mainHistory)[us][m.raw()];
            m.value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            m.value += (bool(pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            int v = threatByLesser[pt] & to ? -19 : 20 * bool(threatByLesser[pt] & from);
            m.value += PieceValue[pt] * v;

            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);

            // ── Tactical theory scoring ──
            // Each theory detects a positional pattern and identifies moves
            // that exploit it.  When ANY theory fires, the move is promoted
            // to a priority tier searched before all non-theory quiets.
            // The per-theory bonus orders moves within the theory tier.

            int theoryBonus = 0;

            // Compute attacks from the destination square (one lookup
            // for non-sliders, one magic lookup for sliders).
            Bitboard moveAtks = attacks_bb(pc, to, occ);

            // ExploitPinTheory (pressure): ANY move whose destination
            // attacks a pinned enemy piece.  The pinned piece cannot move
            // to defend itself (absolute pin) or shouldn't (relative pin),
            // so adding attackers wins material.
            if (TactObs_PinPressure && pinnedEnemy && (moveAtks & pinnedEnemy))
                theoryBonus += TactObs_PinPressure;

            // DiscoveredAttackTheory: our piece blocks a slider ray to
            // the enemy king.  Moving it OFF that ray reveals the attack.
            // Moves along the ray don't create a discovery — filter them.
            if (TactObs_DiscoveredAttack && (discoverers & from))
            {
                Bitboard ray = line_bb(from, enemyKingSq);
                if (!(ray & to))   // moving OFF the ray → discovery
                    theoryBonus += TactObs_DiscoveredAttack;
            }

            // ExecuteForkTheory: knight moves to a square attacking 2+
            // high-value enemy pieces (K/Q/R).  Fork with king forces
            // the king to move, guaranteeing material gain.
            if (pt == KNIGHT)
            {
                Bitboard knightAtks = attacks_bb<KNIGHT>(to);
                Bitboard highValue  = knightAtks & pos.pieces(~us)
                                    & (pos.pieces(QUEEN) | pos.pieces(ROOK));
                Bitboard hitsKing   = knightAtks & enemyKingSq;
                if (hitsKing)
                    highValue |= hitsKing;
                if (more_than_one(highValue))
                    theoryBonus += hitsKing ? TactObs_ForkWithKing
                                            : TactObs_Fork;
            }

            // BackRankAttackTheory: R/Q to enemy's back rank when the
            // king is confined there — threatens back rank mate.
            if (TactObs_BackRank && kingOnBackRank
                && (pt == ROOK || pt == QUEEN) && rank_of(to) == backRank)
                theoryBonus += TactObs_BackRank;

            // Promote theory moves to priority tier
            if (theoryBonus > 0)
                m.value += TheoryTier + theoryBonus;
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
        }
    }
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;
        endCur = endCaptures = score<CAPTURES>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        goto top;
    }

    case GOOD_CAPTURE :
        if (select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                std::swap(*endBadCaptures++, *cur);
                return false;
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);

            partial_insertion_sort(cur, endCur, -3560 * depth);
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad captures
        cur    = moves;
        endCur = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over quiets again
        cur    = endCaptures;
        endCur = endGenerated;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

        return Move::none();

    case EVASION_INIT : {
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        [[fallthrough]];
    }

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish
