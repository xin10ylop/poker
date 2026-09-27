"""Render a decision into the full-information dashboard the LLM reads.

Sections (each can be toggled per prompt variant):
  table     stakes, seats, stacks, positions, hero cards, board
  history   every action this hand, in bb and pot fractions
  hand      engine-verified hand reading (never let the LLM classify its own hand)
  quant     pot odds, MDF, SPR, geometric sizing, equity vs estimated range, EV table
  dossier   per-villain HUD stats (with sample sizes), archetype, tilt, sizing tells,
            recent showdowns, notes, net vs hero, engine range estimate
  reads     Jev's calibrated intuition (P(bluff), P(fold), tilt level, archetype)
  image     how the table perceives hero
  bankroll  stakes, bankroll, risk mode, session P/L
  menu      the legal action menu (ids) the answer must choose from
"""
from __future__ import annotations

from typing import Optional

from ..opponents import OpponentDB
from ..quant import QuantReport
from ..texture import pretty
from ..view import GameView

ALL_SECTIONS = ("table", "history", "hand", "quant", "dossier", "reads", "image", "bankroll", "menu")


def _bb(x: float) -> str:
    v = round(x, 1)
    return f"{v:g}bb"


def render_history(view: GameView) -> str:
    bb = view.bb
    lines = []
    names = {p.seat: f"{p.name} ({p.position})" for p in view.players}
    hero = view.hero_seat
    for street, nb in (("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)):
        acts = [a for a in view.actions if a.street == street]
        if not acts and street != "preflop" and len(view.board) < nb:
            continue
        if street == "preflop":
            head = f"PREFLOP (blinds {view.sb / bb:g}/{1:g}bb):"
        else:
            pot_start = acts[0].pot_before if acts else view.pot
            new = pretty(view.board[nb - 1:nb] if nb > 3 else view.board[:3])
            head = f"{street.upper()} {new}  [board {pretty(view.board[:nb])}] (pot {_bb(pot_start / bb)}):"
        parts = []
        level = 0
        street_in: dict[int, int] = {}
        for a in acts:
            who = "HERO" if a.seat == hero else names[a.seat]
            if a.kind == "post_sb":
                level = max(level, a.to)
                street_in[a.seat] = a.to
                continue
            if a.kind == "post_bb":
                level = max(level, a.to)
                street_in[a.seat] = a.to
                continue
            if a.kind == "post_ante":
                continue
            if a.kind == "fold":
                parts.append(f"{who} folds")
            elif a.kind == "check":
                parts.append(f"{who} checks")
            elif a.kind == "call":
                parts.append(f"{who} calls {_bb(a.added / bb)}" + (" (ALL-IN)" if a.all_in else ""))
            elif a.kind == "bet":
                frac = a.to / max(1, a.pot_before)
                parts.append(f"{who} bets {_bb(a.to / bb)} ({frac:.0%} pot)" + (" (ALL-IN)" if a.all_in else ""))
            elif a.kind == "raise":
                if street == "preflop":
                    parts.append(f"{who} raises to {_bb(a.to / bb)}" + (" (ALL-IN)" if a.all_in else ""))
                else:
                    mult = a.to / max(1, level)
                    parts.append(f"{who} raises to {_bb(a.to / bb)} ({mult:.1f}x)" + (" (ALL-IN)" if a.all_in else ""))
            if a.kind in ("bet", "raise"):
                level = a.to
            street_in[a.seat] = a.to
        lines.append(head + " " + ("; ".join(parts) if parts else "(no action yet)"))
    return "\n".join(lines)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_dossier(db: OpponentDB, vi, view: GameView, reads: Optional[dict]) -> str:
    prof = db.get(vi.name)
    s = prof.summary()
    c = lambda k: prof.samples(k)
    p = vi.params
    arch_probs = prof.heuristic_archetype()
    top_arch = sorted(((k, v) for k, v in arch_probs.items() if k != "unknown"), key=lambda kv: -kv[1])[:2]
    lines = [f"### {vi.name} — {vi.position}, stack {vi.stack_bb:.1f}bb, {prof.hands} hands observed this session"]
    lines.append("Type (stats model): " + ", ".join(f"{k} {v:.0%}" for k, v in top_arch) +
                 f" (unknown {arch_probs.get('unknown', 0):.0%})")
    lines.append(f"Preflop: VPIP {s['vpip']:.0%} (n={c('vpip')}) | PFR {s['pfr']:.0%} | 3-bet {s['3bet']:.1%} "
                 f"(n={c('threebet')}) | fold to 3-bet {s['fold_to_3bet']:.0%} (n={c('fold_to_3bet')}) | "
                 f"steal {s['steal']:.0%} | fold to steal {s['fold_to_steal']:.0%}")
    lines.append(f"Postflop: c-bet {s['cbet']:.0%} (n={c('cbet')}) | fold to c-bet {s['fold_to_cbet']:.0%} "
                 f"(n={c('fold_to_cbet')}) | bets when checked to {prof.stat('bet_checked_to'):.0%} | "
                 f"barrel {prof.stat('barrel'):.0%} | donk {prof.stat('donk'):.0%} | check-raise {s['check_raise']:.0%} | "
                 f"AF {s['af']:.1f} | AFq {s['afq']:.0%}")
    lines.append(f"Folding: vs flop bet {prof.stat('fold_vs_bet_flop'):.0%} | vs turn bet "
                 f"{prof.stat('fold_vs_bet_turn'):.0%} | vs river bet {prof.stat('fold_vs_bet_river'):.0%} "
                 f"(n={c('fold_vs_bet_river')}) | vs raise {s['fold_vs_raise']:.0%} (n={c('fold_vs_raise')})")
    lines.append(f"Showdown: WTSD {s['wtsd']:.0%} (n={c('wtsd')}) | W$SD {s['wsd']:.0%} | avg bet size "
                 f"{s['avg_bet_size_pot']:.0%} pot")
    t = s["sizing_tells"]
    rb_k, rb_n = prof.counts.get("river_bluff", [0, 0])
    lines.append(f"Bluff evidence (showdowns): river bets that were bluffs {rb_k}/{rb_n}; big turn/river bets "
                 f"(>=75% pot) bluff rate {t['bigbet_bluff']:.0%} (n={t['bigbet_n']}), small bets "
                 f"{t['smallbet_bluff']:.0%} (n={t['smallbet_n']}) [population: rivers are under-bluffed]")
    tl = s["tilt"]
    if tl.get("recent_vpip") is not None:
        since = tl.get("hands_since_big_loss")
        lines.append(f"Tilt evidence: score {tl['score']:.2f}; last 12 hands VPIP {tl['recent_vpip']:.0%} / PFR "
                     f"{tl['recent_pfr']:.0%} (career {s['vpip']:.0%}/{s['pfr']:.0%}); "
                     f"{'big loss ' + str(since) + ' hands ago' if since is not None else 'no recent big loss'}; "
                     f"recent net {tl['recent_net_bb']:+.0f}bb")
    sds = prof.showdowns[-4:]
    if sds:
        lines.append("Recent showdowns:")
        for sd in sds:
            ln = "; ".join(f"{st} {k}{(' ' + format(sz, '.0%')) if k in ('bet', 'raise') else ''}"
                           for st, k, sz, hs in sd["lines"]) or "no postflop action"
            lines.append(f"  - showed {pretty(sd['hole'])} on {pretty(sd['board'])} ({sd['final_made'].replace('_', ' ')})"
                         f" after: {ln}; result {sd['net_bb']:+.0f}bb")
    if prof.notes:
        lines.append("Notes:")
        for n in prof.notes[-6:]:
            lines.append(f"  - {n['text']}")
    lines.append(f"His result in pots against you this session: {prof.net_vs_hero_bb:+.0f}bb "
                 f"({'he is losing to you' if prof.net_vs_hero_bb < 0 else 'he is winning vs you'})")
    comp = ", ".join(f"{k} {v:.0%}" for k, v in vi.composition.items() if isinstance(v, (int, float))
                     and k != "combos")
    lines.append(f"Engine estimate of his range THIS hand ({vi.combos:.0f} weighted combos)"
                 + (f": {comp}" if comp else " (preflop: see hand classes below)"))
    lines.append(f"  top hand classes: {vi.range_text[:600]}")
    lines.append(f"  hero equity vs this range: {vi.equity_vs:.1%}")
    return "\n".join(lines)


def render_reads(reads: dict) -> str:
    lines = ["Intuition model (Jev, calibrated probabilities from the same data):"]
    for name, r in reads.items():
        parts = []
        if "villain_bluffing" in r:
            parts.append(f"P(current bet is a bluff) = {r['villain_bluffing']:.2f}")
        if "villain_folds_to_bet" in r:
            parts.append(f"P(folds to a bet/raise now) = {r['villain_folds_to_bet']:.2f}")
        if "tilt_label" in r:
            parts.append(f"tilt: {r['tilt_label']} ({r.get('tilt_score', 0):.1f}/3)")
        if "archetype" in r:
            parts.append("type: " + ", ".join(f"{k} {v:.0%}" for k, v in r["archetype"].items() if v >= 0.05))
        lines.append(f"  {name}: " + "; ".join(parts))
    return "\n".join(lines)


def render_quant(rep: QuantReport, view: GameView, show_pick: bool = True) -> str:
    lines = [f"Pot {rep.pot_bb:.1f}bb | to call {rep.to_call_bb:.1f}bb" +
             (f" -> pot odds: you need {rep.pot_odds:.1%} equity; MDF {rep.mdf:.0%}" if rep.pot_odds else "") +
             f" | SPR {rep.spr:.1f} | effective stack {rep.eff_stack_bb:.1f}bb | you are "
             f"{'IN' if rep.in_position else 'OUT OF'} position"]
    if rep.geometric_bet:
        lines.append(f"Geometric bet (all-in by the river with equal pot fractions): {rep.geometric_bet:.0%} pot per street")
    lines.append(f"Your equity vs the engine's estimated range(s): {rep.equity:.1%}")
    lines.append("Engine EV per action (bb, one-street look-ahead with the villain model above; "
                 "fold = 0 reference):")
    for o in rep.options:
        extra = []
        if o.fold_prob is not None:
            extra.append(f"villain folds {o.fold_prob:.0%} (break-even {o.breakeven_fold:.0%})"
                         if o.breakeven_fold is not None else f"villain folds {o.fold_prob:.0%}")
        if o.eq_called is not None:
            extra.append(f"your equity when called {o.eq_called:.0%}")
        if o.raise_prob:
            extra.append(f"villain raises {o.raise_prob:.0%}")
        if o.detail:
            extra.append(o.detail)
        risk = f", risk-adj {o.risk_adj_bb:+.2f}" if abs(o.risk_adj_bb - o.ev_bb) > 0.05 else ""
        lines.append(f"  {o.id} {o.label}: EV {o.ev_bb:+.2f}{risk}" + (f"  ({'; '.join(extra)})" if extra else ""))
    if show_pick:
        lines.append(f"Engine's pick: {rep.best.id} {rep.best.label}")
    return "\n".join(lines)


def render_dashboard(view: GameView, rep: QuantReport, db: OpponentDB, bankroll_ctx: Optional[dict] = None,
                     reads: Optional[dict] = None, session: Optional[dict] = None,
                     sections: tuple = ALL_SECTIONS) -> str:
    bb = view.bb
    out = []
    if "bankroll" in sections and bankroll_ctx:
        b = bankroll_ctx
        out.append(f"## Money\nStakes {b['stakes']} (1bb = {b['bb_value']}), buy-in {b['buy_in']}. Bankroll "
                   f"{b['bankroll']} = {b['bankroll_buyins']} buy-ins, risk mode: {b['risk_mode']}. Session: "
                   f"{b['session_pnl']} over {b['session_hands']} hands (status {b['session_status']}). Rake "
                   f"{b['rake']}.")
    if "table" in sections:
        seats = []
        for p in view.players:
            tag = "HERO" if p.seat == view.hero_seat else p.name
            state = "folded" if not p.in_hand else ("all-in" if p.all_in else "in hand")
            seats.append(f"{p.position} {tag} {(p.stack + p.bet) / bb:.1f}bb ({state})")
        sess = f" | hand #{session.get('hand_number', '?')} of the session" if session else ""
        out.append(f"## Table\nNo-limit hold'em, {len(view.players)} players{sess}\nSeats: " + " | ".join(seats))
        out.append(f"## Your hand\nYou are {view.hero.position} holding {pretty(view.hole)} "
                   f"({view.hole[0]} {view.hole[1]}). Board: {pretty(view.board) if view.board else '(preflop)'}. "
                   f"Street: {view.street.upper()}.")
    if "hand" in sections:
        out.append(f"Engine hand reading (trust this, do not re-derive): {rep.hand_desc}. "
                   f"Board texture: {rep.texture}. Nut status: {rep.nut_info}.")
    if "history" in sections:
        out.append("## Action so far\n" + render_history(view))
    if "dossier" in sections and rep.villains:
        out.append("## Opponent dossier (what you have learned this session)")
        for vi in rep.villains:
            out.append(render_dossier(db, vi, view, reads))
    if "reads" in sections and reads:
        out.append("## Reads\n" + render_reads(reads))
    if "image" in sections:
        hi = db.hero_image_summary()
        out.append(f"## Your table image\nOpponents have seen you play VPIP {hi['vpip_seen']:.0%} / PFR "
                   f"{hi['pfr_seen']:.0%} over {hi['hands']} hands; caught bluffing at showdown {hi['caught_bluffing']}x, "
                   f"showed value {hi['showed_value']}x.")
    if "quant" in sections or "quant_nopick" in sections:
        out.append("## Numbers\n" + render_quant(rep, view, show_pick="quant" in sections))
    if "menu" in sections:
        menu = "\n".join(f"  {o.id}: {o.label}" for o in rep.options)
        out.append("## Legal actions (answer with one of these ids)\n" + menu)
    return "\n\n".join(out)


def jev_state(view: GameView, rep: QuantReport, db: OpponentDB, bankroll_ctx: Optional[dict] = None,
              include_quant: bool = True) -> dict:
    """Structured state for Jev (System One models like JSON with named fields)."""
    bb = view.bb
    villains = []
    for vi in rep.villains:
        prof = db.get(vi.name)
        s = prof.summary()
        villains.append({
            "name": vi.name, "position": vi.position, "stack_bb": round(vi.stack_bb, 1), "hands_observed": prof.hands,
            "stats": {k: s[k] for k in ("vpip", "pfr", "3bet", "fold_to_3bet", "cbet", "fold_to_cbet",
                                        "fold_vs_river_bet", "fold_vs_raise", "af", "wtsd", "wsd")},
            "bluffs_shown_on_river": prof.counts.get("river_bluff", [0, 0]),
            "sizing_tells": s["sizing_tells"], "tilt_evidence": s["tilt"],
            "recent_showdowns": [f"{pretty(sd['hole'])} on {pretty(sd['board'])} ({sd['final_made']}) after "
                                 + ", ".join(f"{st} {k}" for st, k, sz, hs in sd["lines"]) for sd in prof.showdowns[-3:]],
            "notes": [n["text"] for n in prof.notes[-5:]],
            "engine_range_estimate": vi.composition,
        })
    state = {
        "hand": {"street": view.street, "hero_cards": pretty(view.hole), "board": pretty(view.board),
                 "hero_hand": rep.hand_desc, "board_texture": rep.texture, "pot_bb": round(rep.pot_bb, 1),
                 "to_call_bb": round(rep.to_call_bb, 1), "hero_position": view.hero.position,
                 "hero_in_position": rep.in_position, "spr": round(rep.spr, 1)},
        "action_history": render_history(view),
        "villains": villains,
    }
    if include_quant:
        state["engine"] = {"hero_equity_vs_range": round(rep.equity, 3),
                           "pot_odds_needed": round(rep.pot_odds, 3) if rep.pot_odds else None,
                           "actions": [o.brief() for o in rep.options]}
    if bankroll_ctx:
        state["money"] = {k: bankroll_ctx[k] for k in ("stakes", "bankroll_buyins", "risk_mode")}
    return state
