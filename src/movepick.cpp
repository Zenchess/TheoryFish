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

// Tactical observation theories v4 (SPSA-tunable).
// Based on "Chess Tactics from Scratch" (Chapters 1-12).
//
// Bonuses are ADDITIVE to history scores — they influence move ordering
// without overriding Stockfish's well-tuned heuristics.  Zero disables
// a theory.  Theories are designed for PRECISION: narrow conditions with
// low false-positive rates, so each bonus is meaningful.

// Quiet theories — move creates or exploits a tactical pattern
int TactObs_CreatePin        = 8000;   // Q/R/B creates new pin vs enemy king (Ch 2)
int TactObs_PinPressure      = 4000;   // attack pinned piece from pawn-safe sq (Ch 2)
int TactObs_DiscoveredCheck  = 7000;   // discovered check to pawn-safe sq (Ch 3)
int TactObs_ForkWithKing     = 10000;  // knight fork including king, safe sq (Ch 5)
int TactObs_Fork             = 6000;   // knight fork 2+ high-value, safe sq (Ch 5)
int TactObs_BackRankThreat   = 5000;   // R/Q to back rank, king has 0 flights (Ch 7)

// Capture theory
int TactObs_PinnedCapture    = 3000;   // capturing a pinned piece (Ch 2)

TUNE(SetRange(0, 20000), TactObs_CreatePin);
TUNE(SetRange(0, 20000), TactObs_PinPressure);
TUNE(SetRange(0, 20000), TactObs_DiscoveredCheck);
TUNE(SetRange(0, 20000), TactObs_ForkWithKing);
TUNE(SetRange(0, 20000), TactObs_Fork);
TUNE(SetRange(0, 20000), TactObs_BackRankThreat);
TUNE(SetRange(0, 20000), TactObs_PinnedCapture);

namespace {

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

            // ── Tactical theory scoring v4 ──
            // Each theory detects a precise tactical pattern from "Chess
            // Tactics from Scratch."  Bonuses add to history (no override
            // tier) so Stockfish's heuristics still participate in ordering.

            int theoryBonus = 0;

            // Pawn-safe check: is the destination not attacked by enemy pawns?
            // Reuses threatByLesser[KNIGHT] which equals pawn attacks.
            bool safeFromPawns = !(threatByLesser[KNIGHT] & to);

            // ── CreatePin (Ch 2): Q/R/B creates a pin alignment ──
            // "Two pieces on same line = precondition for pin — just add
            // the attacker."  We check: after the move, exactly one enemy
            // piece sits between our slider and the enemy king on the same
            // line, with no friendly pieces blocking.
            if (TactObs_CreatePin
                && (pt == QUEEN || pt == ROOK || pt == BISHOP))
            {
                // Squares strictly between destination and king
                Bitboard btwn = between_bb(to, enemyKingSq)
                              & ~square_bb(enemyKingSq);
                if (btwn)
                {
                    // Verify piece type can attack along this direction
                    bool fileOrRank = (file_of(to) == file_of(enemyKingSq)
                                    || rank_of(to) == rank_of(enemyKingSq));
                    if (pt == QUEEN
                        || (pt == ROOK   &&  fileOrRank)
                        || (pt == BISHOP && !fileOrRank))
                    {
                        // Occupancy after our piece moves from→to
                        Bitboard occAfter = (occ ^ square_bb(from))
                                          | square_bb(to);
                        Bitboard blockers = btwn & occAfter;
                        // Exactly one blocker, and it's an enemy piece
                        if (blockers && !more_than_one(blockers)
                            && (blockers & pos.pieces(~us)))
                            theoryBonus += TactObs_CreatePin;
                    }
                }
            }

            // ── PinPressure (Ch 2): attack pinned piece from safe sq ──
            // "Function matters more than material — a pinned piece cannot
            // defend itself."  Only fires from pawn-safe squares to avoid
            // promoting moves that just hang the attacker.
            if (TactObs_PinPressure && pinnedEnemy && safeFromPawns)
            {
                Bitboard moveAtks = attacks_bb(pc, to, occ);
                if (moveAtks & pinnedEnemy)
                    theoryBonus += TactObs_PinPressure;
            }

            // ── DiscoveredCheck (Ch 3): discoverer moves off king ray ──
            // "Discovered checks are the most devastating form."  Our piece
            // blocks a slider aimed at the enemy king; moving OFF the ray
            // reveals check.  Only boost if landing on pawn-safe square.
            if (TactObs_DiscoveredCheck
                && (discoverers & from) && safeFromPawns)
            {
                Bitboard ray = line_bb(from, enemyKingSq);
                if (!(ray & to))   // moving OFF the ray → discovered check
                    theoryBonus += TactObs_DiscoveredCheck;
            }

            // ── Knight Fork (Ch 5): attack 2+ high-value targets ──
            // Only from pawn-safe squares — a fork on a pawn-attacked
            // square is worthless.  Fork with king forces king to move,
            // guaranteeing material gain.
            if (pt == KNIGHT && safeFromPawns)
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

            // ── BackRankThreat (Ch 7): R/Q to back rank, king trapped ──
            // "Back rank weakness requires 0 flight squares."  Check that
            // all king escape squares off the back rank are blocked by
            // the enemy's own pieces (classic pawn-shield trap).
            if (TactObs_BackRankThreat && kingOnBackRank
                && (pt == ROOK || pt == QUEEN) && rank_of(to) == backRank)
            {
                Rank secondRank = (us == WHITE) ? RANK_7 : RANK_2;
                Bitboard frontFlights = attacks_bb<KING>(enemyKingSq)
                                      & rank_bb(secondRank);
                // All escape squares blocked by own pieces (or none exist)
                if (frontFlights && !(frontFlights & ~pos.pieces(~us)))
                    theoryBonus += TactObs_BackRankThreat;
            }

            // Add theory bonus directly to history score
            m.value += theoryBonus;
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
