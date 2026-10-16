"""Compact character CTC models, grammar validation, decoding, and rejection."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from me2.contracts import finite, integer

LEGACY_CHARSET = "abcdefghijklmnopqrstuvwxyz "
CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789 "
TEXT_NORMALIZER_VERSION = "nfkd-ascii-alnum-space-v1"
GRAMMAR_SCHEMA_VERSION = 1
DECODER_VERSION = "ctc-prefix-logsumexp-v1"
DECODER_RUNTIME_VERSION = "bounded-prefix-runtime-v3-diagnostics"
BLANK_ID = len(CHARSET)
CHAR_TO_ID = {character: index for index, character in enumerate(CHARSET)}


class DecoderBudgetError(RuntimeError):
    """System resource failure, never a terminal command or ordinary OOS."""

    def __init__(self, reason: str, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.diagnostics = {**(diagnostics or {}), "budget_reason": reason}


class InferenceInputError(ValueError):
    """A categorized invalid runtime input; never an actionable command."""

    def __init__(self, category: str, message: str, diagnostics=None):
        super().__init__(message)
        self.category = category
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class DecoderLimits:
    max_frames: int = 2048
    max_expansions: int = 250_000
    max_grammar_characters: int = 100_000

    def __post_init__(self) -> None:
        integer("max_frames", self.max_frames, 1, 4096)
        integer("max_expansions", self.max_expansions, 1, 2_000_000)
        integer("max_grammar_characters", self.max_grammar_characters, 1, 100_000)


@dataclass(frozen=True)
class GrammarEntry:
    """One unambiguous legal spoken form and its product semantics."""

    text: str
    intent: str
    variation: str = ""
    slots: Mapping[str, str] = field(default_factory=dict)
    provenance: str = "approved_train_alias"

    def normalized(self) -> GrammarEntry:
        return GrammarEntry(
            text=normalize_transcript(self.text),
            intent=self.intent.strip().casefold(),
            variation=self.variation.strip(),
            slots=dict(sorted((str(key), str(value)) for key, value in self.slots.items())),
            provenance=self.provenance.strip(),
        )

    def semantic_key(self) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        return self.intent, self.variation, tuple(sorted(self.slots.items()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["slots"] = dict(self.slots)
        return value


@dataclass(frozen=True)
class AcceptancePolicy:
    """Validation-frozen thresholds; decoder scores are not probabilities."""

    min_normalized_log_score: float = -1.0e30
    min_margin: float = -1.0e30
    min_candidate_posterior: float = 0.0
    version: str = "uncalibrated-v1"

    def __post_init__(self) -> None:
        finite("min_normalized_log_score", self.min_normalized_log_score, -1e30, 1e30)
        finite("min_margin", self.min_margin, -1e30, 1e30)
        finite("min_candidate_posterior", self.min_candidate_posterior, 0.0, 1.0)
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Acceptance policy requires a non-empty version")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> AcceptancePolicy:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("Acceptance policy must be an object")
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown acceptance policy field")
        return cls(
            min_normalized_log_score=(
                value.get("min_normalized_log_score", -1.0e30)
            ),
            min_margin=value.get("min_margin", -1.0e30),
            min_candidate_posterior=value.get("min_candidate_posterior", 0.0),
            version=value.get("version", "uncalibrated-v1"),
        )


@dataclass(frozen=True)
class DecodeCandidate:
    entry: GrammarEntry
    raw_log_score: float
    normalized_log_score: float
    candidate_posterior: float


@dataclass(frozen=True)
class CommandResult:
    decoded_text: str
    intent: str | None
    slots: Mapping[str, str]
    variation: str
    raw_log_score: float
    normalized_log_score: float
    candidate_posterior: float
    margin: float
    accepted: bool
    rejection_reason: str | None
    model_version: str = ""
    grammar_version: str = ""
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    system_error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def actionable_intent(self) -> str | None:
        return self.intent if self.accepted and self.system_error is None else None


class CharacterAudioCNN(nn.Module):
    """Legacy 3-convolution, 2-layer BiGRU model retained for migration."""

    def __init__(
        self,
        n_mels: int = 40,
        hidden_size: int = 256,
        output_symbols: int = len(LEGACY_CHARSET) + 1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        encoded_mels = math.ceil(math.ceil(n_mels / 2) / 2)
        self.recurrent = nn.GRU(
            128 * encoded_mels,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, output_symbols)

    @staticmethod
    def output_lengths(feature_lengths: Tensor) -> Tensor:
        first = torch.div(feature_lengths + 1, 2, rounding_mode="floor")
        return torch.div(first + 1, 2, rounding_mode="floor")

    def forward(self, features: Tensor, feature_lengths: Tensor | None = None) -> Tensor:
        encoded = self.encoder(features).permute(0, 3, 1, 2)
        encoded = encoded.reshape(encoded.size(0), encoded.size(1), -1)
        if feature_lengths is not None:
            lengths = self.output_lengths(feature_lengths).clamp(max=encoded.size(1))
            packed = pack_padded_sequence(
                encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed, _ = self.recurrent(packed)
            encoded, _ = pad_packed_sequence(
                packed, batch_first=True, total_length=encoded.size(1)
            )
        else:
            encoded, _ = self.recurrent(encoded)
        return self.classifier(encoded)


class CompactCTCModel(nn.Module):
    """Depthwise-separable temporal encoder with one small bidirectional GRU."""

    def __init__(
        self,
        n_mels: int = 40,
        width: int = 48,
        projection_size: int = 128,
        hidden_size: int = 96,
        output_symbols: int = len(CHARSET) + 1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.width = width
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, width // 2, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU6(inplace=True),
        )
        self.encoder = nn.Sequential(
            _SeparableCTCBlock(width // 2, width, stride=(2, 2)),
            _SeparableCTCBlock(width, width, stride=(1, 1)),
        )
        encoded_mels = math.ceil(math.ceil(n_mels / 2) / 2)
        self.projection = nn.Sequential(
            nn.Linear(width * encoded_mels, projection_size),
            nn.ReLU6(inplace=True),
        )
        self.recurrent = nn.GRU(
            projection_size,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, output_symbols)

    @staticmethod
    def output_lengths(feature_lengths: Tensor) -> Tensor:
        return torch.div(feature_lengths + 1, 2, rounding_mode="floor")

    def forward(self, features: Tensor, feature_lengths: Tensor | None = None) -> Tensor:
        encoded = self.encoder(self.stem(features)).permute(0, 3, 1, 2)
        encoded = encoded.reshape(encoded.size(0), encoded.size(1), -1)
        encoded = self.projection(encoded)
        if feature_lengths is not None:
            lengths = self.output_lengths(feature_lengths).clamp(min=1, max=encoded.size(1))
            packed = pack_padded_sequence(
                encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed, _ = self.recurrent(packed)
            encoded, _ = pad_packed_sequence(
                packed, batch_first=True, total_length=encoded.size(1)
            )
        else:
            encoded, _ = self.recurrent(encoded)
        return self.classifier(encoded)


class _SeparableCTCBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int]
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(inputs))))


def normalize_transcript(text: str) -> str:
    """Normalize spoken text without deleting digits or inventing expansions."""

    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.replace("’", "'").replace("`", "'")
    normalized = re.sub(r"(?<=[a-z])'(?=[a-z])", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def transcript_target(text: str, charset: str = CHARSET) -> list[int]:
    normalized = normalize_transcript(text)
    if not normalized:
        raise ValueError("CTC transcript cannot be empty")
    character_to_id = {character: index for index, character in enumerate(charset)}
    unsupported = sorted(set(normalized) - set(character_to_id))
    if unsupported:
        raise ValueError(f"Unsupported CTC transcript characters: {unsupported}")
    return [character_to_id[character] for character in normalized]


def required_ctc_frames(target: Sequence[int] | str) -> int:
    """Minimum frames required by CTC, including repeated-adjacent separators."""

    values: Sequence[Any] = target
    return len(values) + sum(left == right for left, right in pairwise(values))


def coerce_grammar_entries(
    entries: Iterable[GrammarEntry | Mapping[str, Any] | tuple[str, str]],
) -> tuple[GrammarEntry, ...]:
    normalized: list[GrammarEntry] = []
    by_text: dict[str, GrammarEntry] = {}
    for raw_entry in entries:
        if isinstance(raw_entry, GrammarEntry):
            entry = raw_entry.normalized()
        elif isinstance(raw_entry, Mapping):
            entry = GrammarEntry(
                text=str(raw_entry.get("text", "")),
                intent=str(raw_entry.get("intent", "")),
                variation=str(raw_entry.get("variation", "")),
                slots={
                    str(key): str(value)
                    for key, value in dict(raw_entry.get("slots", {})).items()
                },
                provenance=str(raw_entry.get("provenance", "approved_train_alias")),
            ).normalized()
        else:
            text, intent = raw_entry
            entry = GrammarEntry(text=str(text), intent=str(intent)).normalized()
        if not entry.text or not entry.intent:
            raise ValueError("Grammar entries require non-empty normalized text and intent")
        if entry.intent == "out_of_scope":
            continue
        existing = by_text.get(entry.text)
        if existing is not None and existing.semantic_key() != entry.semantic_key():
            raise ValueError(
                "Conflicting grammar semantics for normalized text "
                f"{entry.text!r}: {existing.semantic_key()} != {entry.semantic_key()}"
            )
        if existing is None:
            by_text[entry.text] = entry
            normalized.append(entry)
    if not normalized:
        raise ValueError("Grammar must contain at least one actionable terminal")
    return tuple(sorted(normalized, key=lambda item: (item.text, item.semantic_key())))


@lru_cache(maxsize=8)
def _compile_grammar(
    serialized_entries: tuple[
        tuple[str, str, str, tuple[tuple[str, str], ...], str], ...
    ],
) -> tuple[dict[str, dict[str, str]], dict[str, GrammarEntry]]:
    children: dict[str, dict[str, str]] = {"": {}}
    terminals: dict[str, GrammarEntry] = {}
    for text, intent, variation, slots, provenance in serialized_entries:
        node = ""
        for character in text:
            next_node = node + character
            children.setdefault(node, {})[character] = next_node
            children.setdefault(next_node, {})
            node = next_node
        terminals[text] = GrammarEntry(
            text=text,
            intent=intent,
            variation=variation,
            slots=dict(slots),
            provenance=provenance,
        )
    return children, terminals


def _serialized_grammar(
    entries: Iterable[GrammarEntry | Mapping[str, Any] | tuple[str, str]],
) -> tuple[tuple[str, str, str, tuple[tuple[str, str], ...], str], ...]:
    return tuple(
        (
            entry.text,
            entry.intent,
            entry.variation,
            tuple(sorted(entry.slots.items())),
            entry.provenance,
        )
        for entry in coerce_grammar_entries(entries)
    )


def _log_add(left: float, right: float) -> float:
    return float(np.logaddexp(left, right))


def _log_softmax(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("CTC logits contain non-finite values")
    maximum = np.max(values)
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def greedy_ctc_decode(
    logits: np.ndarray | Tensor,
    *,
    charset: str = CHARSET,
    blank_id: int | None = None,
    input_length: int | None = None,
) -> str:
    """Collapse the most likely frame path without applying a grammar."""

    blank_id = len(charset) if blank_id is None else int(blank_id)
    if blank_id != len(charset):
        raise ValueError("CTC blank must immediately follow the character vocabulary")
    frame_logits = (
        logits.detach().cpu().numpy() if isinstance(logits, Tensor) else np.asarray(logits)
    )
    expected_symbols = len(charset) + 1
    if frame_logits.ndim != 2 or frame_logits.shape[1] != expected_symbols:
        raise ValueError(
            f"Expected frame logits shaped [time, {expected_symbols}], got "
            f"{tuple(frame_logits.shape)}"
        )
    if not np.isfinite(frame_logits).all():
        raise ValueError("CTC logits contain non-finite values")
    usable_length = frame_logits.shape[0] if input_length is None else int(input_length)
    if usable_length < 1 or usable_length > frame_logits.shape[0]:
        raise ValueError("input_length must select at least one available CTC frame")

    collapsed: list[str] = []
    previous_id: int | None = None
    for symbol_id in np.argmax(frame_logits[:usable_length], axis=1).tolist():
        if symbol_id != previous_id and symbol_id != blank_id:
            collapsed.append(charset[symbol_id])
        previous_id = symbol_id
    return "".join(collapsed)


def prefix_beam_search(
    logits: np.ndarray | Tensor,
    entries: Iterable[GrammarEntry | Mapping[str, Any] | tuple[str, str]],
    *,
    charset: str = CHARSET,
    blank_id: int | None = None,
    beam_width: int = 16,
    input_length: int | None = None,
    top_k: int = 8,
    limits: DecoderLimits | None = None,
    diagnostics: dict | None = None,
) -> list[DecodeCandidate]:
    """Run full-utterance CTC prefix beam search constrained to a trie."""

    limits = limits or DecoderLimits()
    diagnostics = diagnostics if diagnostics is not None else {}
    integer("beam_width", beam_width, 1, 256)
    integer("top_k", top_k, 1, 256)
    if not charset or len(set(charset)) != len(charset):
        raise ValueError("CTC charset must contain unique characters")
    blank_id = len(charset) if blank_id is None else blank_id
    if blank_id != len(charset):
        raise ValueError("CTC blank must immediately follow the character vocabulary")
    character_to_id = {character: index for index, character in enumerate(charset)}
    serialized = _serialized_grammar(entries)
    if sum(len(entry[0]) for entry in serialized) > limits.max_grammar_characters:
        raise DecoderBudgetError("decoder_grammar_budget_exhausted")
    children, terminals = _compile_grammar(serialized)
    diagnostics.update(
        beam_width=beam_width, grammar_terminal_count=len(terminals),
        minimum_terminal_frames=min((required_ctc_frames(t) for t in terminals), default=0),
        pruned_prefix_count=0, beam_pruning_observed=False,
    )
    unsupported = sorted(
        {
            character
            for text in terminals
            for character in text
            if character not in character_to_id
        }
    )
    if unsupported:
        raise ValueError(f"Grammar contains characters outside the vocabulary: {unsupported}")

    frame_logits = (
        logits.detach().cpu().numpy() if isinstance(logits, Tensor) else np.asarray(logits)
    )
    expected_symbols = len(charset) + 1
    if frame_logits.ndim != 2 or frame_logits.shape[1] != expected_symbols:
        raise InferenceInputError("invalid_logits_shape",
            f"Expected frame logits shaped [time, {expected_symbols}], got "
            f"{tuple(frame_logits.shape)}"
        )
    if not frame_logits.shape[0]:
        raise InferenceInputError("insufficient_valid_frames", "No CTC output frames")
    usable_length = frame_logits.shape[0] if input_length is None else input_length
    integer("input_length", usable_length, 1, frame_logits.shape[0])
    diagnostics.update(output_frames=frame_logits.shape[0], valid_output_frames=usable_length)
    if usable_length > limits.max_frames:
        raise DecoderBudgetError("decoder_frame_budget_exhausted", diagnostics)
    frame_logits = frame_logits[:usable_length]
    if not np.isfinite(frame_logits).all():
        raise InferenceInputError("invalid_nonfinite_logits", "CTC logits contain non-finite values", diagnostics)

    negative_infinity = float("-inf")
    beams: dict[str, tuple[float, float]] = {"": (0.0, negative_infinity)}
    expansions = 0
    for raw_frame in frame_logits:
        frame = _log_softmax(raw_frame.astype(np.float64, copy=False))
        next_beams: dict[str, list[float]] = {}

        def add(
            prefix: str,
            *,
            blank_score: float | None = None,
            nonblank_score: float | None = None,
            target_beams: dict[str, list[float]] = next_beams,
        ) -> None:
            scores = target_beams.setdefault(
                prefix, [negative_infinity, negative_infinity]
            )
            if blank_score is not None:
                scores[0] = _log_add(scores[0], blank_score)
            if nonblank_score is not None:
                scores[1] = _log_add(scores[1], nonblank_score)

        for prefix, (blank_score, nonblank_score) in beams.items():
            expansions += 1 + bool(prefix) + len(children[prefix])
            if expansions > limits.max_expansions:
                raise DecoderBudgetError("decoder_work_budget_exhausted", {
                    **diagnostics, "expansions": expansions,
                })
            total_score = _log_add(blank_score, nonblank_score)
            add(prefix, blank_score=total_score + frame[blank_id])

            if prefix:
                repeated_id = character_to_id[prefix[-1]]
                add(prefix, nonblank_score=nonblank_score + frame[repeated_id])

            for character, child in children[prefix].items():
                character_id = character_to_id[character]
                extension_score = (
                    blank_score if prefix and character == prefix[-1] else total_score
                )
                add(child, nonblank_score=extension_score + frame[character_id])

        ranked = sorted(
            next_beams.items(),
            key=lambda item: (-_log_add(item[1][0], item[1][1]), item[0]),
        )
        pruned = sum(
            math.isfinite(_log_add(scores[0], scores[1]))
            for _, scores in ranked[beam_width:]
        )
        diagnostics["pruned_prefix_count"] += pruned
        diagnostics["beam_pruning_observed"] |= pruned > 0
        ranked = ranked[:beam_width]
        beams = {
            prefix: (scores[0], scores[1]) for prefix, scores in ranked
        }

    terminal_scores: list[tuple[float, str, GrammarEntry]] = []
    for prefix, (blank_score, nonblank_score) in beams.items():
        terminal = terminals.get(prefix)
        if terminal is not None and math.isfinite(_log_add(blank_score, nonblank_score)):
            terminal_scores.append(
                (_log_add(blank_score, nonblank_score), prefix, terminal)
            )
    terminal_scores.sort(key=lambda item: (-item[0], item[1], item[2].semantic_key()))
    finite_prefixes = [p for p, scores in beams.items() if math.isfinite(_log_add(*scores))]
    diagnostics.update(
        surviving_prefix_count=len(finite_prefixes),
        nonempty_prefix_count=sum(bool(p) for p in finite_prefixes),
        longest_surviving_prefix=max(map(len, finite_prefixes), default=0),
        terminal_count=len(terminal_scores), expansions=expansions,
    )
    if not terminal_scores:
        return []

    intent_scores: dict[str, float] = {}
    for score, _, terminal in terminal_scores:
        intent_scores[terminal.intent] = _log_add(
            intent_scores.get(terminal.intent, negative_infinity), score
        )
    intent_log_normalizer = _logsumexp(list(intent_scores.values()))
    candidates = [
        DecodeCandidate(
            entry=terminal,
            raw_log_score=score,
            normalized_log_score=score / usable_length,
            candidate_posterior=math.exp(
                intent_scores[terminal.intent] - intent_log_normalizer
            ),
        )
        for score, _, terminal in terminal_scores
    ]
    return candidates[:top_k]


def decode_command(
    logits: np.ndarray | Tensor,
    entries: Iterable[GrammarEntry | Mapping[str, Any] | tuple[str, str]],
    *,
    charset: str = CHARSET,
    blank_id: int | None = None,
    beam_width: int = 16,
    input_length: int | None = None,
    acceptance_policy: AcceptancePolicy | Mapping[str, Any] | None = None,
    model_version: str = "",
    grammar_version: str = "",
    limits: DecoderLimits | None = None,
) -> CommandResult:
    policy = (
        acceptance_policy
        if isinstance(acceptance_policy, AcceptancePolicy)
        else AcceptancePolicy.from_mapping(acceptance_policy)
    )
    diagnostics: dict = {}
    candidates = prefix_beam_search(
        logits,
        entries,
        charset=charset,
        blank_id=blank_id,
        beam_width=beam_width,
        input_length=input_length,
        top_k=max(beam_width, 2),
        limits=limits,
        diagnostics=diagnostics,
    )
    if not candidates:
        return CommandResult(
            decoded_text="",
            intent=None,
            slots={},
            variation="",
            raw_log_score=float("-inf"),
            normalized_log_score=float("-inf"),
            candidate_posterior=0.0,
            margin=float("-inf"),
            accepted=False,
            rejection_reason="no_terminal_candidate",
            model_version=model_version,
            grammar_version=grammar_version,
            diagnostics={
                **diagnostics,
                "diagnostic_category": (
                    "insufficient_valid_frames"
                    if diagnostics["valid_output_frames"] < diagnostics["minimum_terminal_frames"]
                    else "legal_prefix_no_terminal"
                    if diagnostics["nonempty_prefix_count"]
                    else "no_legal_trie_path"
                ),
                # Pruning observed is evidence, not proof that a correct path
                # was removed. Unsupported wording requires a known reference.
                "phrase_support": "unknown_without_reference",
                "next_action": "check_reference_grammar_capture_lengths_and_greedy_alignment",
            },
        )

    best = candidates[0]
    runner_up = next(
        (candidate for candidate in candidates[1:] if candidate.entry.intent != best.entry.intent),
        None,
    )
    margin = (
        best.normalized_log_score - runner_up.normalized_log_score
        if runner_up is not None
        else 0.0
    )
    rejection_reason = None
    if best.normalized_log_score < policy.min_normalized_log_score:
        rejection_reason = "acoustic_score_below_threshold"
    elif margin < policy.min_margin:
        rejection_reason = "semantic_margin_below_threshold"
    elif best.candidate_posterior < policy.min_candidate_posterior:
        rejection_reason = "candidate_posterior_below_threshold"
    accepted = rejection_reason is None
    return CommandResult(
        decoded_text=best.entry.text,
        intent=best.entry.intent,
        slots=dict(best.entry.slots),
        variation=best.entry.variation,
        raw_log_score=best.raw_log_score,
        normalized_log_score=best.normalized_log_score,
        candidate_posterior=best.candidate_posterior,
        margin=margin,
        accepted=accepted,
        rejection_reason=rejection_reason,
        model_version=model_version,
        grammar_version=grammar_version,
        candidate_count=len(candidates),
        diagnostics={
            **diagnostics,
            "diagnostic_category": "accepted" if accepted else "rejection_policy_failure",
        },
    )


def inspect_phrase_support(text: str, entries, *, output_frames: int | None = None) -> dict:
    """Diagnostic only: a supplied trial phrase never alters grammar or scoring."""
    normalized = normalize_transcript(text)
    legal = {entry.text for entry in coerce_grammar_entries(entries)}
    required = required_ctc_frames(normalized)
    return {
        "expected_phrase_supported": normalized in legal,
        "normalization_changed": normalized != text,
        "expected_required_frames": required,
        "expected_alignment_feasible": None if output_frames is None else output_frames >= required,
        "reference_diagnostic": (
            "unsupported_or_normalization_mismatched_phrase" if normalized not in legal
            else "insufficient_valid_frames" if output_frames is not None and output_frames < required
            else "supported_grammar_terminal"
        ),
    }


def decode_grammar(
    logits: np.ndarray | Tensor,
    transcripts: Iterable[GrammarEntry | Mapping[str, Any] | tuple[str, str]],
    labels: list[str],
    beam_width: int = 16,
    *,
    charset: str = LEGACY_CHARSET,
    input_length: int | None = None,
) -> tuple[str, float, list[float], str]:
    """Backward-compatible adapter for the historical intent-probability API."""

    result_candidates = prefix_beam_search(
        logits,
        transcripts,
        charset=charset,
        blank_id=len(charset),
        beam_width=beam_width,
        input_length=input_length,
        top_k=max(beam_width, len(labels)),
    )
    scores = {label: float("-inf") for label in labels}
    best_text_by_label: dict[str, tuple[float, str]] = {}
    for candidate in result_candidates:
        label = candidate.entry.intent
        if label not in scores:
            continue
        scores[label] = _log_add(scores[label], candidate.raw_log_score)
        current = best_text_by_label.get(label)
        if current is None or candidate.raw_log_score > current[0]:
            best_text_by_label[label] = (candidate.raw_log_score, candidate.entry.text)
    finite_scores = [score for score in scores.values() if math.isfinite(score)]
    if not finite_scores:
        fallback = labels.index("out_of_scope") if "out_of_scope" in labels else 0
        probabilities = [0.0] * len(labels)
        probabilities[fallback] = 1.0
        return labels[fallback], 0.0, probabilities, ""
    normalizer = _logsumexp(finite_scores)
    probabilities = [
        math.exp(scores[label] - normalizer) if math.isfinite(scores[label]) else 0.0
        for label in labels
    ]
    best_index = int(np.argmax(probabilities))
    best_label = labels[best_index]
    return (
        best_label,
        probabilities[best_index],
        probabilities,
        best_text_by_label.get(best_label, (0.0, ""))[1],
    )


def _logsumexp(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))
