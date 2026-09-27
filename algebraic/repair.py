"""
algebraic.repair — deterministic syntax repair and lexical state for free text.

Pure functions, no dependencies.  Used by the algebra MCP (`json_repair`,
`balance_repair`, `lexer_state`) and as the recovering fallback of the typed JSON
generator (algebraic.jev_json).

Contract of `json_repair`:
  * if the input already parses, it is returned byte-for-byte;
  * otherwise the LONGEST PREFIX that ends exactly after a complete JSON value is kept
    and closed -- a whole incomplete token (`,` / `"key":` / `tru` / `"abc`) is dropped,
    not shaved off one character at a time;
  * string-aware: `'` inside `"..."`, escapes, trailing commas, lone `{`;
  * the `valid` flag is the RESULT OF json.loads on the returned string, never a guess;
  * on failure the ORIGINAL text is returned unchanged (never a mutated non-JSON string).
"""
from __future__ import annotations
import json
import re

_WS = " \t\n\r"
_OPEN = "([{"
_CLOSE = ")]}"
_PAIRS = {")": "(", "]": "[", "}": "{"}
_ESC = {"(": ")", "[": "]", "{": "}"}

_SCALAR = re.compile(r'(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)')


# --------------------------------------------------------------------------- lexer
def lexer_state(text: str) -> str:
    """Final lexical state of free text: normal / dq / sq / comment / esc.

    `esc` remembers WHICH quote it is escaping, so it always returns to that quote
    (a bug in the first version sent an escaped `'` back into double-quote mode).
    """
    st, q = "normal", ""
    for c in text:
        if st == "normal":
            if c == '"':
                st, q = "dq", '"'
            elif c == "'":
                st, q = "sq", "'"
            elif c == "#":
                st = "comment"
        elif st == "dq" or st == "sq":
            if c == "\\":
                st = "esc"
            elif c == q:
                st = "normal"
        elif st == "comment":
            if c == "\n":
                st = "normal"
        elif st == "esc":
            st = "dq" if q == '"' else "sq"
    return st


# --------------------------------------------------------------------------- balance
def _balance(s: str) -> str:
    """Append the closers needed to balance (), [], {} and quotes (string-aware)."""
    stack, in_str, esc, q = [], False, False, ""
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_str = False
        else:
            if ch in "\"'":
                in_str, q = True, ch
            elif ch in _OPEN:
                stack.append(ch)
            elif ch in _CLOSE:
                if stack and stack[-1] == _PAIRS[ch]:
                    stack.pop()
    out = s
    if in_str:
        out += q
    for o in reversed(stack):
        out += _ESC[o]
    return out


def balance_repair(text: str):
    """(balanced_text, ok) -- ok is False when a closer appears without its opener."""
    r = _balance(text)
    d, ok = 0, True
    in_str, esc, q = False, False, ""
    for ch in r:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_str = False
            continue
        if ch in "\"'":
            in_str, q = True, ch
        elif ch in _OPEN:
            d += 1
        elif ch in _CLOSE:
            d -= 1
            if d < 0:
                ok = False
    if d > 0:
        ok = False
    return r, ok


# ------------------------------------------------------------- longest valid prefix
def _string_end(s: str, i: int) -> int:
    """Index just past the closing quote, or -1 if the string is unterminated."""
    n, esc = len(s), False
    j = i + 1
    while j < n:
        c = s[j]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            return j + 1
        j += 1
    return -1


def _closers(kinds) -> str:
    return "".join(_ESC[k] for k in reversed(kinds))


def _longest_complete_prefix(s: str):
    """(cut, kinds) for the longest prefix ending after a complete value, else None.

    A cut is always a position at which the open containers can simply be closed:
    either right after a complete value, or right after an opening bracket.
    """
    n, i = len(s), 0
    stack = []          # [open_char, state];  '{': k=key ':' 'v','c'   '[': 'v','c'
    best = None

    def kinds():
        return tuple(c for c, _ in stack)

    def note(pos):
        nonlocal best
        if best is None or pos >= best[0]:
            best = (pos, kinds())

    while i < n:
        ch = s[i]
        if ch in _WS:
            i += 1
            continue
        if not stack:                                   # root: exactly one value
            if ch == "{":
                stack.append(["{", "k"]); i += 1; note(i); continue
            if ch == "[":
                stack.append(["[", "v"]); i += 1; note(i); continue
            if ch == '"':
                j = _string_end(s, i)
                note(j if j >= 0 else n)                # close the unterminated string
                break
            m = _SCALAR.match(s, i)
            if m:
                note(m.end())
            break
        k, st = stack[-1]
        if k == "[":
            if st == "v":
                if ch == "]":
                    stack.pop(); i += 1; note(i)
                    if stack:
                        stack[-1][1] = "c"
                    else:
                        break
                elif ch == "{":
                    stack.append(["{", "k"]); i += 1; note(i)
                elif ch == "[":
                    stack.append(["[", "v"]); i += 1; note(i)
                elif ch == '"':
                    j = _string_end(s, i)
                    i = j if j >= 0 else n
                    note(i); stack[-1][1] = "c"
                else:
                    m = _SCALAR.match(s, i)
                    if not m:
                        break
                    i = m.end(); note(i); stack[-1][1] = "c"
            else:                                       # 'c': comma or ]
                if ch == ",":
                    stack[-1][1] = "v"; i += 1
                elif ch == "]":
                    stack.pop(); i += 1; note(i)
                    if stack:
                        stack[-1][1] = "c"
                    else:
                        break
                else:
                    break
        else:                                           # object
            if st == "k":                               # key or }
                if ch == "}":
                    stack.pop(); i += 1; note(i)
                    if stack:
                        stack[-1][1] = "c"
                    else:
                        break
                elif ch == '"':
                    j = _string_end(s, i)
                    if j < 0:
                        break                               # unterminated KEY -> drop it
                    i = j; stack[-1][1] = ":"
                else:
                    break
            elif st == ":":
                if ch != ":":
                    break
                stack[-1][1] = "v"; i += 1
            elif st == "v":
                if ch == "{":
                    stack.append(["{", "k"]); i += 1; note(i)
                elif ch == "[":
                    stack.append(["[", "v"]); i += 1; note(i)
                elif ch == '"':
                    j = _string_end(s, i)
                    i = j if j >= 0 else n
                    note(i); stack[-1][1] = "c"
                else:
                    m = _SCALAR.match(s, i)
                    if not m:
                        break
                    i = m.end(); note(i); stack[-1][1] = "c"
            else:                                       # 'c': comma or }
                if ch == ",":
                    stack[-1][1] = "k"; i += 1
                elif ch == "}":
                    stack.pop(); i += 1; note(i)
                    if stack:
                        stack[-1][1] = "c"
                    else:
                        break
                else:
                    break
    return best


def _strip_trailing_commas(s: str) -> str:
    """String-aware removal of `,` directly before `]` or `}`."""
    out, in_str, esc, q = [], False, False, ""
    n = len(s)
    for idx, ch in enumerate(s):
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                in_str = False
            continue
        if ch in "\"'":
            in_str, q = True, ch
        elif ch == ",":
            j = idx + 1
            while j < n and s[j] in _WS:
                j += 1
            if j < n and s[j] in "]}":
                continue
        out.append(ch)
    return "".join(out)


def _parses(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _drop_dangling_escape(s: str) -> str:
    """Drop a trailing backslash that escapes nothing (`..."x\\` at end of input)."""
    j = len(s)
    while j > 0 and s[j - 1] == "\\":
        j -= 1
    return s[:-1] if (len(s) - j) % 2 == 1 else s


_FENCE = re.compile(r"^```[A-Za-z0-9_+-]*\s*\n?(.*?)\n?```\s*$", re.S)


# --------------------------------------------------------------------------- repair
def _candidates(text: str, depth: int = 0):
    """Repair candidates, most faithful first; every candidate is verified by json.loads."""
    if depth > 4:
        return []
    s = text.strip()
    if not s:
        return []
    out = []

    m = _FENCE.match(s)                               # ```json ... ``` fenced block
    if m:
        out.extend(_candidates(m.group(1), depth + 1))

    for i, ch in enumerate(s):                        # leading prose before the JSON
        if ch in "[{":
            if i:
                out.extend(_candidates(s[i:], depth + 1))
            break

    if '"' not in s and s.count("'") >= 2:            # single-quoted JSON
        out.extend(_candidates(s.replace("'", '"'), depth + 1))

    for base in (s, _drop_dangling_escape(s)):        # longest complete prefix, closed
        p = _longest_complete_prefix(base)
        if p:
            out.append(base[:p[0]] + _closers(p[1]))

    st = _strip_trailing_commas(s)                    # trailing commas removed
    if st != s:
        out.append(st)
        p = _longest_complete_prefix(st)
        if p:
            out.append(st[:p[0]] + _closers(p[1]))

    out.append(_balance(_drop_dangling_escape(s)))    # balance unbalanced quotes/brackets
    out.append(_balance(s))
    return out


def json_repair(text: str):
    """(repaired_json, valid). See the module docstring for the exact contract."""
    if _parses(text):
        return text, True
    for c in _candidates(text):
        if c and _parses(c):
            return c, True
    return text, False                                # honest: the ORIGINAL text
