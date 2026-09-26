import random
import string as _string
from typing import Any


def pick_k_distinct(rng: random.Random, items: list[Any], k: int) -> list[Any]:
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    return rng.sample(items, k)


def bart_default_config():
    return {
        # probability a *record* (Concept) is targeted at all
        "p_record": 0.25,
        # probability a targeted record will corrupt each eligible field
        "p_field": 0.50,
        # character-level ops (applied independently; low rates recommended)
        "p_char_delete": 0.03,
        "p_char_insert": 0.03,
        "p_char_subst": 0.05,
        "p_char_transp": 0.01,
        # token-level
        "p_token_split": 0.01,
        "p_token_merge": 0.01,
        # surface tweaks
        "p_case": 0.01,
        "p_whitespace": 0.01,
        # confusion models
        "p_ocr": 0.02,
        "p_phonetic": 0.01,
        # occasionally (rare) swap labels between two random nodes (value confusion)
        "p_label_swap": 0.01,
        # fields to consider inside ground_set
        "fields": [
            "labels",
            "alt_labels",
            "related_synonyms",
            "exact_synonyms",
            "definitions",
        ],
        # also (rarely) perturb Concept.name (for strictly label-driven setups)
        "p_mutate_name": 0.03,
    }


class BARTNoiser:
    """
    A lightweight BART-like error generator for string fields.
    Models character-level (delete/insert/substitute/transpose), OCR/keyboard confusions,
    token-level split/merge, casing/whitespace tweaks, and optional phonetic tweaks.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, *, seed: int | None = None):
        self.cfg = {**bart_default_config(), **(cfg or {})}
        self.rng = random.Random(seed)

        # crude keyboard adjacency for QWERTY; used for plausible typos
        self._adj = {
            "q": "w",
            "w": "qes",
            "e": "wsdr",
            "r": "edft",
            "t": "rfgy",
            "y": "tghu",
            "u": "yhj i",
            "i": "ujko",
            "o": "iklp",
            "p": "ol",
            "a": "sqwz",
            "s": "awedxz",
            "d": "serfcx",
            "f": "drtgcv",
            "g": "ftyhbv",
            "h": "gyujnb",
            "j": "huikm n",
            "k": "jiol, m",
            "l": "kop; .",
            "z": "asx",
            "x": "zsdc",
            "c": "xdfv",
            "v": "cfgb",
            "b": "vghn",
            "n": "bhjm",
            "m": "njk,",
        }
        # OCR confusions
        self._ocr_pairs = [
            ("0", "O"),
            ("1", "l"),
            ("1", "I"),
            ("2", "Z"),
            ("5", "S"),
            ("8", "B"),
            ("rn", "m"),
            ("cl", "d"),
        ]
        # phonetic-ish replacements
        self._ph_map = [
            ("ph", "f"),
            ("ght", "t"),
            ("ck", "k"),
            ("c", "k"),
            ("q", "k"),
            ("x", "ks"),
            ("z", "s"),
        ]

    def _maybe(self, p: float) -> bool:
        return self.rng.random() < p

    def _subst_char(self, ch: str) -> str:
        # pick plausible neighbor or random ascii letter/digit
        lower = ch.lower()
        cand = self._adj.get(lower, "") or ""
        if cand and self._maybe(0.7):
            rep = self.rng.choice(list(cand.replace(" ", "")))
            return rep.upper() if ch.isupper() else rep
        # fallback: random ascii
        alphabet = _string.ascii_letters + _string.digits
        rep = self.rng.choice(alphabet)
        return rep.upper() if ch.isupper() else rep

    def _insert_char(self, around: str | None = None) -> str:
        # insert a plausible neighbor if available
        if around:
            lower = around.lower()
            cand = self._adj.get(lower, "") or ""
            if cand and self._maybe(0.6):
                ch = self.rng.choice(list(cand.replace(" ", "")))
                return ch
        return self.rng.choice(_string.ascii_letters + _string.digits)

    def _apply_ocr(self, s: str) -> str:
        # randomly apply one OCR confusion pair
        pair = self.rng.choice(self._ocr_pairs)
        a, b = pair
        if a in s and self._maybe(0.5):
            return s.replace(a, b)
        if b in s:
            return s.replace(b, a)
        return s

    def _apply_phonetic(self, s: str) -> str:
        # apply one phonetic-ish mapping
        frm, to = self.rng.choice(self._ph_map)
        if frm in s and self._maybe(0.7):
            return s.replace(frm, to)
        return s

    def _case_whitespace(self, s: str) -> str:
        out = []
        for ch in s:
            if ch.isspace() and self._maybe(self.cfg["p_whitespace"]):
                # drop or duplicate space
                if self._maybe(0.5):
                    continue
                else:
                    out.append(ch)
                    out.append(ch)
            elif ch.isalpha() and self._maybe(self.cfg["p_case"]):
                out.append(ch.swapcase())
            else:
                out.append(ch)
        return "".join(out)

    def _char_ops(self, s: str) -> str:
        if not s:
            return s
        chars = list(s)
        i = 0
        out = []
        while i < len(chars):
            ch = chars[i]
            # delete
            if self._maybe(self.cfg["p_char_delete"]):
                i += 1
                continue
            # transpose with next
            if i + 1 < len(chars) and self._maybe(self.cfg["p_char_transp"]):
                out.append(chars[i + 1])
                out.append(ch)
                i += 2
                continue
            # substitute
            if self._maybe(self.cfg["p_char_subst"]):
                out.append(self._subst_char(ch))
            else:
                out.append(ch)
            # insert after
            if self._maybe(self.cfg["p_char_insert"]):
                out.append(self._insert_char(around=ch))
            i += 1
        return "".join(out)

    def _token_ops(self, s: str) -> str:
        toks = s.split()
        if not toks:
            return s
        # split: choose one token and split at a plausible interior point
        if len(toks) >= 1 and self._maybe(self.cfg["p_token_split"]):
            j = self.rng.randrange(len(toks))
            tk = toks[j]
            if len(tk) > 2:
                cut = self.rng.randrange(1, len(tk))
                toks = toks[:j] + [tk[:cut], tk[cut:]] + toks[j + 1 :]
        # merge: choose an adjacent pair
        if len(toks) >= 2 and self._maybe(self.cfg["p_token_merge"]):
            j = self.rng.randrange(len(toks) - 1)
            toks = toks[:j] + [toks[j] + toks[j + 1]] + toks[j + 2 :]
        return " ".join(toks)

    def noise_string(self, s: str) -> str:
        if not s or s.strip() == "":
            return s
        t = s
        # 1) OCR/phonetic (occasionally)
        if self._maybe(self.cfg["p_ocr"]):
            t = self._apply_ocr(t)
        if self._maybe(self.cfg["p_phonetic"]):
            t = self._apply_phonetic(t)
        # 2) char ops
        t = self._char_ops(t)
        # 3) token ops
        t = self._token_ops(t)
        # 4) case/whitespace tweaks
        t = self._case_whitespace(t)
        return t

    def noise_list(self, values: list[str]) -> list[str]:
        out = []
        for v in values:
            if self._maybe(self.cfg["p_field"]):
                out.append(self.noise_string(v))
            else:
                out.append(v)
        return out
