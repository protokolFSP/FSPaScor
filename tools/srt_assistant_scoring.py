"""
SRT-based assistant scoring (German FSP dialogues).

Outputs compatible with your CSV schema (schema_version=7):
filename,assistant_overall_score,anamnesis_end_s,presentation_start_s,feedback_start_s,
p1_overall,p1_duration_s,p1_grammar_per_100w,p1_language_quality,p1_clarity,p1_fluency,
p1_filler_per_100w,p1_long_pauses_per_min,
p2_overall,p2_duration_s,p2_grammar_per_100w,p2_language_quality,p2_clarity,p2_fluency,
p2_filler_per_100w,p2_long_pauses_per_min,p2_excl_short_q_count,p2_excl_short_q_s,schema_version

Key idea:
- Phase cuts from SRT timestamps (pattern + stats).
- Assistant vs OTHER with 2-state Viterbi (no diarization).
- Long pauses counted only for "silence before assistant speaks" to avoid counting patient speech as pause.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import language_tool_python


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PhaseCuts:
    anamnesis_end_s: float
    presentation_start_s: float
    feedback_start_s: float


_FILLERS = {"äh", "aeh", "ähm", "aehm", "hm", "mm", "naja", "halt", "quasi", "sozusagen"}

_PRESENTATION_PATTERNS = [
    r"\bich habe (hier )?(eine[nr]? )?(neue[nr]? )?(patient(in|en)?|fall)\b",
    r"\bdarf ich (ihnen )?(den )?(fall|patient(in|en)?) (kurz )?vorstellen\b",
    r"\bhaben sie (kurz )?zeit\b",
    r"\bich würde (gern|gerne) (den )?(fall|die patientin|den patienten) vorstellen\b",
    r"\bich (stelle|präsentiere) (ihnen )?(jetzt )?(den )?(fall|die patientin|den patienten)\b",
]

_FEEDBACK_PATTERNS = [
    r"\brückmeldung\b",
    r"\bfeedback\b",
    r"\bwas kann ich (besser|verbessern)\b",
    r"\bwie war (das|meine vorstellung)\b",
]

_QUESTION_START = re.compile(
    r"^\s*(haben|hätten|hatten|können|könnte|würden|würde|ist|sind|war|waren|wie|was|wann|warum|wieso|wo|welche|wieviel|gibt es)\b",
    re.IGNORECASE,
)

_PRESENTATION_VOCAB = {
    "anamnese", "vorgeschichte", "vorerkrankungen", "medikation", "allergien", "befund",
    "diagnose", "differentialdiagnose", "therapie", "verlauf", "labor", "bildgebung",
    "ekg", "rr", "spo2", "status", "aufnahme",
}


def parse_srt(path: Path) -> List[Segment]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", raw.strip(), flags=re.MULTILINE)
    out: List[Segment] = []

    for b in blocks:
        lines = [ln.strip("\ufeff").strip() for ln in b.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        ts_line = None
        for ln in lines[:2]:
            if "-->" in ln:
                ts_line = ln
                break
        if not ts_line:
            continue

        m = re.search(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", ts_line)
        if not m:
            continue

        start = srt_time_to_s(m.group(1))
        end = srt_time_to_s(m.group(2))

        # text lines = everything after the timestamp line
        idx = lines.index(ts_line)
        text = " ".join(lines[idx + 1 :])
        text = normalize_text(text)
        if text:
            out.append(Segment(start=start, end=end, text=text))

    out.sort(key=lambda s: (s.start, s.end))
    return out


def srt_time_to_s(s: str) -> float:
    hh, mm, rest = s.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def merge_segments(segs: List[Segment], gap_s: float = 0.8) -> List[Utterance]:
    if not segs:
        return []
    utts: List[Utterance] = []

    cur_start = segs[0].start
    cur_end = segs[0].end
    cur_text = segs[0].text

    for s in segs[1:]:
        gap = s.start - cur_end
        if gap <= gap_s:
            cur_end = max(cur_end, s.end)
            cur_text = (cur_text + " " + s.text).strip()
        else:
            utts.append(Utterance(start=cur_start, end=cur_end, text=cur_text))
            cur_start, cur_end, cur_text = s.start, s.end, s.text

    utts.append(Utterance(start=cur_start, end=cur_end, text=cur_text))
    return utts


def tokenise(text: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß\-]+", (text or "").lower())


def is_question(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if "?" in t:
        return True
    return bool(_QUESTION_START.search(t))


def detect_phases(
    utts: List[Utterance],
    audio_duration_s: float,
    anamnesis_end_s: float = 1200.0,
    window_s: float = 120.0,
    step_s: float = 30.0,
) -> PhaseCuts:
    anam_end = min(float(anamnesis_end_s), float(audio_duration_s))

    feedback_start = float(audio_duration_s)
    for u in utts:
        if u.start < anam_end:
            continue
        for pat in _FEEDBACK_PATTERNS:
            if re.search(pat, u.text, re.IGNORECASE):
                feedback_start = u.start
                break
        if feedback_start < audio_duration_s:
            break

    pres_start: Optional[float] = None
    for u in utts:
        if u.start < anam_end:
            continue
        for pat in _PRESENTATION_PATTERNS:
            if re.search(pat, u.text, re.IGNORECASE):
                pres_start = max(anam_end, u.start - 2.0)
                break
        if pres_start is not None:
            break

    if pres_start is None:
        pres_start = detect_presentation_by_stats(
            utts=utts,
            start_s=anam_end,
            end_s=min(feedback_start, float(audio_duration_s)),
            window_s=float(window_s),
            step_s=float(step_s),
        )

    pres_start = float(pres_start)
    if pres_start > feedback_start:
        feedback_start = float(audio_duration_s)

    return PhaseCuts(anamnesis_end_s=anam_end, presentation_start_s=pres_start, feedback_start_s=float(feedback_start))


def detect_presentation_by_stats(utts: List[Utterance], start_s: float, end_s: float, window_s: float, step_s: float) -> float:
    if end_s - start_s < window_s:
        return start_s

    best_score = -1.0
    best_t = start_s
    t = start_s

    while t + window_s <= end_s:
        win = [u for u in utts if u.start >= t and u.start < t + window_s]
        if len(win) < 6:
            t += step_s
            continue

        q = sum(1 for u in win if is_question(u.text))
        question_ratio = q / max(1, len(win))

        words = sum(len(tokenise(u.text)) for u in win)
        avg_words = words / max(1, len(win))

        score = (1.0 - question_ratio) * 0.7 + clip01(avg_words / 18.0) * 0.3
        if score > best_score:
            best_score = score
            best_t = t

        t += step_s

    return best_t


def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def emission_logprobs(u: Utterance, mode: str, force_assistant: bool) -> Tuple[float, float]:
    if force_assistant:
        return 0.0, -1e18

    toks = tokenise(u.text)
    wc = len(toks)
    q = 1.0 if is_question(u.text) else 0.0

    second = sum(1 for t in toks if t in {"sie", "ihnen", "ihr", "ihre", "ihren"})
    first = sum(1 for t in toks if t in {"ich", "mir", "mich", "mein", "meine", "meinen"})
    pres = sum(1 for t in toks if t in _PRESENTATION_VOCAB)

    second_r = second / max(1, wc)
    first_r = first / max(1, wc)
    pres_r = pres / max(1, wc)

    if mode == "p1":
        # assistant (anamnesis): questions + 2nd person, less "ich"
        logit = 2.2 * q + 2.0 * second_r - 1.5 * first_r + 0.02 * wc
    else:
        # assistant (presentation): longer + presentation vocab, fewer questions
        logit = 2.2 * pres_r + 0.06 * wc - 2.2 * q - 0.3 * second_r

    p0 = sigmoid(logit)
    p0 = min(0.999999, max(0.000001, p0))
    return math.log(p0), math.log(1.0 - p0)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def viterbi_labels(utts: List[Utterance], mode: str, anchor_assistant: Optional[List[bool]] = None) -> List[int]:
    # 0=ASSISTANT, 1=OTHER
    n = len(utts)
    if n == 0:
        return []
    anchor_assistant = anchor_assistant or [False] * n

    log_stay = math.log(0.88)
    log_switch = math.log(0.12)

    dp0 = [-1e18] * n
    dp1 = [-1e18] * n
    bp0 = [0] * n
    bp1 = [0] * n

    e0, e1 = emission_logprobs(utts[0], mode=mode, force_assistant=anchor_assistant[0])
    dp0[0], dp1[0] = e0, e1

    for i in range(1, n):
        e0, e1 = emission_logprobs(utts[i], mode=mode, force_assistant=anchor_assistant[i])

        a = dp0[i - 1] + log_stay
        b = dp1[i - 1] + log_switch
        if a >= b:
            dp0[i] = a + e0
            bp0[i] = 0
        else:
            dp0[i] = b + e0
            bp0[i] = 1

        a = dp1[i - 1] + log_stay
        b = dp0[i - 1] + log_switch
        if a >= b:
            dp1[i] = a + e1
            bp1[i] = 1
        else:
            dp1[i] = b + e1
            bp1[i] = 0

    states = [0] * n
    states[-1] = 0 if dp0[-1] >= dp1[-1] else 1
    for i in range(n - 1, 0, -1):
        states[i - 1] = bp0[i] if states[i] == 0 else bp1[i]
    return states


def p2_reliability_weight(p2_duration_s: float, min_ok_s: float = 90.0, full_ok_s: float = 180.0) -> float:
    d = float(p2_duration_s)
    if d <= min_ok_s:
        return 0.0
    if d >= full_ok_s:
        return 1.0
    return (d - min_ok_s) / (full_ok_s - min_ok_s)


def grammar_errors_per_100w(text: str, tool: language_tool_python.LanguageTool) -> float:
    words = tokenise(text)
    if not words:
        return 0.0
    matches = tool.check(text)
    return (len(matches) / len(words)) * 100.0


def grammar_score(err_per_100w: float) -> float:
    s = 100.0 - 5.0 * float(err_per_100w)
    return max(0.0, min(100.0, s))


def lexical_richness_score(text: str) -> float:
    toks = tokenise(text)
    if len(toks) < 30:
        return 50.0
    types = len(set(toks))
    cttr = types / math.sqrt(2.0 * len(toks))
    s = (cttr - 0.15) / (0.35 - 0.15) * 100.0
    return max(0.0, min(100.0, s))


def language_quality_score(grammar_s: float, lex_s: float) -> float:
    return 0.75 * grammar_s + 0.25 * lex_s


def filler_per_100w(text: str) -> float:
    toks = tokenise(text)
    if not toks:
        return 0.0
    f = sum(1 for t in toks if t in _FILLERS)
    return (f / len(toks)) * 100.0


def clarity_score(text: str) -> float:
    toks = tokenise(text)
    if len(toks) < 10:
        return 70.0
    bigrams = list(zip(toks, toks[1:]))
    rep = sum(1 for i in range(1, len(bigrams)) if bigrams[i] == bigrams[i - 1]) / max(1, len(bigrams))
    weird = weird_token_rate(text)
    s = 100.0 - 200.0 * rep - 120.0 * weird
    return max(0.0, min(100.0, s))


def weird_token_rate(text: str) -> float:
    toks = re.findall(r"\S+", (text or ""))
    if not toks:
        return 0.0
    weird = 0
    for t in toks:
        if re.search(r"[A-Za-zÄÖÜäöüß]", t) is None:
            weird += 1
        elif len(t) >= 18 and re.fullmatch(r"[A-Za-zÄÖÜäöüß\-]+", t) is None:
            weird += 1
    return weird / len(toks)


def fluency_score(filler_100w: float, pauses_per_min: float) -> float:
    s = 100.0 - 4.0 * filler_100w - 2.5 * pauses_per_min
    return max(0.0, min(100.0, s))


def phase_overall(lang_q: float, clarity: float, fluency: float) -> float:
    return 0.50 * lang_q + 0.25 * clarity + 0.25 * fluency


def assistant_long_pauses_per_min(
    phase_utts: List[Utterance],
    labels: List[int],
    pause_threshold_s: float = 0.5,
) -> float:
    """
    Count silence gaps BEFORE assistant utterances:
      gap = utt[i].start - utt[i-1].end
      if gap>=threshold and labels[i]==ASSISTANT => counts as assistant hesitation pause

    This avoids counting patient/Oberarzt speaking time as "pause".
    """
    if len(phase_utts) < 2:
        return 0.0

    ass_dur = sum(max(0.0, u.end - u.start) for u, lab in zip(phase_utts, labels) if lab == 0)
    if ass_dur <= 1e-6:
        return 0.0

    cnt = 0
    for i in range(1, len(phase_utts)):
        gap = phase_utts[i].start - phase_utts[i - 1].end
        if gap >= pause_threshold_s and labels[i] == 0:
            cnt += 1

    return cnt / (ass_dur / 60.0)


def score_from_srt(srt_path: Path, anamnesis_end_s: float = 1200.0) -> Dict[str, float]:
    segs = parse_srt(srt_path)
    utts = merge_segments(segs, gap_s=0.8)
    if not utts:
        raise ValueError(f"No usable utterances in {srt_path}")

    audio_duration_s = max(u.end for u in utts)
    cuts = detect_phases(utts, audio_duration_s=audio_duration_s, anamnesis_end_s=anamnesis_end_s)

    # IMPORTANT: p1 is strictly 0..anamnesis_end
    p1_utts = [u for u in utts if u.start < cuts.anamnesis_end_s]
    # p2 is presentation..feedback
    p2_utts = [u for u in utts if u.start >= cuts.presentation_start_s and u.start < cuts.feedback_start_s]

    # Anchors
    p1_anchor = [False] * len(p1_utts)
    for i, u in enumerate(p1_utts[: min(15, len(p1_utts))]):
        if re.search(r"\bmein name\b", u.text, re.IGNORECASE) and re.search(r"\bassisten", u.text, re.IGNORECASE):
            p1_anchor[i] = True
            break

    p2_anchor = [False] * len(p2_utts)
    for i, u in enumerate(p2_utts[: min(20, len(p2_utts))]):
        if any(re.search(p, u.text, re.IGNORECASE) for p in _PRESENTATION_PATTERNS):
            p2_anchor[i] = True
            break

    p1_labels = viterbi_labels(p1_utts, mode="p1", anchor_assistant=p1_anchor)
    p2_labels = viterbi_labels(p2_utts, mode="p2", anchor_assistant=p2_anchor)

    p1_ass_text = " ".join(u.text for u, lab in zip(p1_utts, p1_labels) if lab == 0).strip()
    p2_ass_text = " ".join(u.text for u, lab in zip(p2_utts, p2_labels) if lab == 0).strip()

    p1_ass_dur = sum(max(0.0, u.end - u.start) for u, lab in zip(p1_utts, p1_labels) if lab == 0)
    p2_ass_dur = sum(max(0.0, u.end - u.start) for u, lab in zip(p2_utts, p2_labels) if lab == 0)

    # Exclude short Oberarzt questions bookkeeping (p2)
    short_q_words = 6
    p2_excl_count = 0
    p2_excl_s = 0.0
    for u, lab in zip(p2_utts, p2_labels):
        if lab != 0 and is_question(u.text) and len(tokenise(u.text)) <= short_q_words:
            p2_excl_count += 1
            p2_excl_s += max(0.0, u.end - u.start)

    tool = language_tool_python.LanguageTool("de-DE")

    def metrics(text: str, phase_utts: List[Utterance], labels: List[int], ass_dur: float) -> Dict[str, float]:
        err100 = grammar_errors_per_100w(text, tool)
        gscore = grammar_score(err100)
        lex = lexical_richness_score(text)
        langq = language_quality_score(gscore, lex)

        fill100 = filler_per_100w(text)
        pauses_pm = assistant_long_pauses_per_min(phase_utts, labels, pause_threshold_s=0.5)
        flu = fluency_score(fill100, pauses_pm)

        cla = clarity_score(text)
        overall = phase_overall(langq, cla, flu)

        return {
            "duration_s": float(ass_dur),
            "grammar_per_100w": float(err100),
            "language_quality": float(langq),
            "clarity": float(cla),
            "fluency": float(flu),
            "filler_per_100w": float(fill100),
            "long_pauses_per_min": float(pauses_pm),
            "overall": float(overall),
        }

    p1 = metrics(p1_ass_text, p1_utts, p1_labels, p1_ass_dur)
    p2 = metrics(p2_ass_text, p2_utts, p2_labels, p2_ass_dur)

    rel = p2_reliability_weight(p2["duration_s"])
    denom = 0.65 + 0.35 * rel
    assistant_overall = (0.65 * p1["overall"] + 0.35 * rel * p2["overall"]) / denom if denom > 0 else p1["overall"]

    return {
        "assistant_overall_score": float(assistant_overall),
        "anamnesis_end_s": float(cuts.anamnesis_end_s),
        "presentation_start_s": float(cuts.presentation_start_s),
        "feedback_start_s": float(cuts.feedback_start_s),

        "p1_overall": p1["overall"],
        "p1_duration_s": p1["duration_s"],
        "p1_grammar_per_100w": p1["grammar_per_100w"],
        "p1_language_quality": p1["language_quality"],
        "p1_clarity": p1["clarity"],
        "p1_fluency": p1["fluency"],
        "p1_filler_per_100w": p1["filler_per_100w"],
        "p1_long_pauses_per_min": p1["long_pauses_per_min"],

        "p2_overall": p2["overall"],
        "p2_duration_s": p2["duration_s"],
        "p2_grammar_per_100w": p2["grammar_per_100w"],
        "p2_language_quality": p2["language_quality"],
        "p2_clarity": p2["clarity"],
        "p2_fluency": p2["fluency"],
        "p2_filler_per_100w": p2["filler_per_100w"],
        "p2_long_pauses_per_min": p2["long_pauses_per_min"],
        "p2_excl_short_q_count": float(p2_excl_count),
        "p2_excl_short_q_s": float(p2_excl_s),

        "schema_version": 7.0,
    }
