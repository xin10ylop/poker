"""Card primitives on top of eval7 (a fast C hand evaluator).

Cards are plain strings like "Ah", "Td", "2c" everywhere in the public API.
A *combo* is a canonical 2-tuple of cards, higher card first.
A *hand class* is one of the 169 preflop classes: "AA", "AKs", "AKo", ...
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}  # '2' -> 0 ... 'A' -> 12
ALL_CARDS: list[str] = [r + s for r in RANKS for s in SUITS]
_E7 = {c: eval7.Card(c) for c in ALL_CARDS}

Combo = tuple[str, str]


def normalize_card(card: str) -> str:
    """Accept 'ah', 'AH', '10h', 'Th' and return canonical 'Ah'/'Th'."""
    c = card.strip()
    if c.startswith("10"):
        c = "T" + c[2:]
    if len(c) != 2:
        raise ValueError(f"bad card {card!r}")
    r, s = c[0].upper(), c[1].lower()
    if r not in RANK_VALUE or s not in SUITS:
        raise ValueError(f"bad card {card!r}")
    return r + s


def parse_cards(text: str | list[str] | tuple) -> list[str]:
    """Parse 'AhKd', 'Ah Kd', 'Ah,Kd' or a list into canonical cards."""
    if isinstance(text, (list, tuple)):
        return [normalize_card(c) for c in text]
    t = text.replace(",", " ").replace("10", "T").split()
    out: list[str] = []
    for tok in t:
        if len(tok) % 2:
            raise ValueError(f"bad card string {text!r}")
        out.extend(normalize_card(tok[i:i + 2]) for i in range(0, len(tok), 2))
    return out


def rank_of(card: str) -> int:
    return RANK_VALUE[card[0]]


def to_eval7(cards) -> list:
    return [_E7[c] for c in cards]


def evaluate(cards) -> int:
    """Hand value (5-7 cards); higher is better."""
    return eval7.evaluate([_E7[c] for c in cards])


def hand_type(value: int) -> str:
    """'High Card', 'Pair', 'Two Pair', 'Trips', 'Straight', 'Flush', 'Full House', 'Quads', 'Straight Flush'."""
    return eval7.handtype(value)


def card_key(card: str) -> tuple[int, int]:
    return (RANK_VALUE[card[0]], SUITS.index(card[1]))


def make_combo(a: str, b: str) -> Combo:
    return (a, b) if card_key(a) > card_key(b) else (b, a)


ALL_COMBOS: list[Combo] = [make_combo(a, b) for a, b in itertools.combinations(ALL_CARDS, 2)]


def hand_class(combo) -> str:
    a, b = make_combo(*combo)
    ra, rb = a[0], b[0]
    if ra == rb:
        return ra + rb
    return ra + rb + ("s" if a[1] == b[1] else "o")


def _build_classes() -> tuple[list[str], dict[str, list[Combo]]]:
    classes: list[str] = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            hi, lo = RANKS[max(i, j)], RANKS[min(i, j)]
            if i == j:
                cls = hi + lo
            elif i > j:
                cls = hi + lo + "s"
            else:
                cls = hi + lo + "o"
            if cls not in classes:
                classes.append(cls)
    members: dict[str, list[Combo]] = {c: [] for c in classes}
    for combo in ALL_COMBOS:
        members[hand_class(combo)].append(combo)
    return classes, members


ALL_CLASSES, CLASS_COMBOS = _build_classes()  # 169 classes, grid order


@lru_cache(maxsize=None)
def combos_of_class(cls: str) -> tuple[Combo, ...]:
    return tuple(CLASS_COMBOS[cls])


def class_size(cls: str) -> int:
    return 6 if len(cls) == 2 else (4 if cls.endswith("s") else 12)


def fmt_cards(cards) -> str:
    return " ".join(cards) if cards else "-"


def remaining_deck(dead) -> list[str]:
    d = set(dead)
    return [c for c in ALL_CARDS if c not in d]


def stable_hash(*parts) -> int:
    """Deterministic 32-bit hash (Python's hash() is salted per process)."""
    import zlib
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF
