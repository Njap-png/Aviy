"""Educational byte-level BPE tokenizer trained only on supplied corpus data."""

from __future__ import annotations

import heapq
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from aviy.data.pipeline import PreparedDataset


class TokenizerMetrics(TypedDict):
    samples: int
    words: int
    tokens: int
    characters: int
    fertility_tokens_per_word: float
    characters_per_token: float
    character_coverage_rate: float
    byte_coverage_rate: float
    unknown_characters: int
    round_trip_rate: float


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        seen: set[tuple[int, int]] = set()
        for index, pair in enumerate(self.merges):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or token < 0
                    or token >= 256 + index
                    for token in pair
                )
            ):
                raise ValueError(f"Invalid BPE merge at index {index}")
            if pair in seen:
                raise ValueError(f"Duplicate BPE merge at index {index}")
            seen.add(pair)
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.vocabulary_size = 256 + len(self.merges)

        pieces = [bytes([index]) for index in range(256)]
        for left, right in self.merges:
            pieces.append(pieces[left] + pieces[right])
        self._pieces = pieces

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string")
        return " ".join(unicodedata.normalize("NFC", text).split())

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocabulary_size: int = 16384,
        minimum_pair_frequency: int = 2,
    ) -> ByteBPETokenizer:
        """Train deterministic byte-level BPE using incremental pair counts."""
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size < 256
        ):
            raise ValueError("vocabulary_size must be an integer of at least 256")
        if (
            isinstance(minimum_pair_frequency, bool)
            or not isinstance(minimum_pair_frequency, int)
            or minimum_pair_frequency < 1
        ):
            raise ValueError("minimum_pair_frequency must be a positive integer")

        try:
            iterator = iter(texts)
        except TypeError as exc:
            raise TypeError("Tokenizer training texts must be iterable") from exc

        tokens: list[int] = []
        previous: list[int] = []
        following: list[int] = []
        alive: list[bool] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        heads: list[int] = []

        for sequence_id, text in enumerate(iterator):
            normalized = cls.normalize(text)
            if not normalized:
                continue

            encoded = normalized.encode("utf-8")
            sequence_nodes = []

            for position, token in enumerate(encoded):
                node = len(tokens)
                sequence_nodes.append(node)
                tokens.append(token)
                previous.append(-1)
                following.append(-1)
                alive.append(True)
                sequence_ids.append(sequence_id)
                positions.append(position)

            if sequence_nodes:
                heads.append(sequence_nodes[0])
                for index in range(len(sequence_nodes) - 1):
                    left = sequence_nodes[index]
                    right = sequence_nodes[index + 1]
                    following[left] = right
                    previous[right] = left

        if not tokens:
            raise ValueError("Tokenizer training requires non-empty approved text")

        # Pair occurrence heaps retain the original sequence and position order.
        pair_occurrences: dict[
            tuple[int, int],
            list[tuple[int, int, int, int]],
        ] = {}
        pair_counts: Counter[tuple[int, int]] = Counter()
        candidates: list[
            tuple[int, int, int, int, int, tuple[int, int]]
        ] = []

        def valid_occurrence(
            pair: tuple[int, int],
            occurrence: tuple[int, int, int, int],
        ) -> bool:
            _, _, left, right = occurrence
            return (
                alive[left]
                and alive[right]
                and following[left] == right
                and tokens[left] == pair[0]
                and tokens[right] == pair[1]
            )

        def earliest(pair: tuple[int, int]):
            occurrences = pair_occurrences.get(pair)
            if occurrences is None:
                return None

            while occurrences and not valid_occurrence(pair, occurrences[0]):
                heapq.heappop(occurrences)

            return occurrences[0] if occurrences else None

        def refresh(pair: tuple[int, int]) -> None:
            count = pair_counts[pair]
            if count <= 0:
                return

            first = earliest(pair)
            if first is None:
                return

            sequence_id, position, left, right = first
            heapq.heappush(
                candidates,
                (-count, sequence_id, position, left, right, pair),
            )

        def add_edge(left: int, right: int) -> None:
            pair = (tokens[left], tokens[right])
            occurrences = pair_occurrences.setdefault(pair, [])
            occurrence = (
                sequence_ids[left],
                positions[left],
                left,
                right,
            )
            heapq.heappush(occurrences, occurrence)
            pair_counts[pair] += 1
            refresh(pair)

        def remove_edge(left: int, right: int) -> None:
            pair = (tokens[left], tokens[right])
            pair_counts[pair] -= 1
            refresh(pair)

        for node in range(len(tokens)):
            right = following[node]
            if right != -1:
                add_edge(node, right)

        merges: list[tuple[int, int]] = []
        next_id = 256

        while next_id < vocabulary_size and candidates:
            selected = None

            while candidates:
                negative_count, sequence_id, position, left, right, pair = (
                    heapq.heappop(candidates)
                )

                count = pair_counts[pair]
                first = earliest(pair)

                if count <= 0 or first is None:
                    continue

                if (
                    -negative_count != count
                    or first != (sequence_id, position, left, right)
                ):
                    continue

                selected = pair
                break

            if selected is None:
                break

            if pair_counts[selected] < minimum_pair_frequency:
                break

            merged_id = next_id
            next_id += 1
            merges.append(selected)

            occurrences = pair_occurrences[selected]

            # Merge every currently valid occurrence in left-to-right order.
            while True:
                first = earliest(selected)
                if first is None:
                    break

                heapq.heappop(occurrences)
                _, _, left, right = first

                if not valid_occurrence(selected, first):
                    continue

                before = previous[left]
                after = following[right]

                if before != -1:
                    remove_edge(before, left)

                remove_edge(left, right)

                if after != -1:
                    remove_edge(right, after)

                tokens[left] = merged_id
                following[left] = after

                if after != -1:
                    previous[after] = left

                alive[right] = False
                previous[right] = -1
                following[right] = -1

                if before != -1:
                    add_edge(before, left)

                if after != -1:
                    add_edge(left, after)

        return cls(merges)

    @classmethod
    def train_from_prepared_dataset(
        cls,
        dataset: PreparedDataset,
        *,
        vocabulary_size: int = 16384,
        minimum_pair_frequency: int = 2,
        authorization: object | None = None,
    ) -> ByteBPETokenizer:
        """Train from a passing dataset only with matching explicit authorization."""
        from aviy.training.authorization import (
            fingerprint_prepared_dataset,
            require_training_authorization,
        )

        if not dataset.audit.passed:
            raise ValueError("Tokenizer training requires a passing dataset audit")
        training_texts = [
            record.text for record in dataset.records if record.split == "train"
        ]
        if not training_texts:
            raise ValueError("The audited dataset has no training-split records")
        require_training_authorization(
            authorization,
            expected_stage="tokenizer_training",
            expected_fingerprint=fingerprint_prepared_dataset(dataset),
        )
        return cls.train(
            training_texts,
            vocabulary_size=vocabulary_size,
            minimum_pair_frequency=minimum_pair_frequency,
        )

    def encode(self, text: str) -> list[int]:
        """Encode text using a priority queue while preserving BPE merge order."""
        encoded = self.normalize(text).encode("utf-8")
        if not encoded:
            return []

        tokens = list(encoded)
        previous = list(range(-1, len(tokens) - 1))
        following = list(range(1, len(tokens) + 1))
        following[-1] = -1
        alive = [True] * len(tokens)

        candidates: list[tuple[int, int, int, int]] = []

        def add_candidate(left: int, right: int) -> None:
            if left == -1 or right == -1:
                return

            pair = (tokens[left], tokens[right])
            rank = self.ranks.get(pair)
            if rank is not None:
                heapq.heappush(
                    candidates,
                    (rank, left, left, right),
                )

        for left in range(len(tokens) - 1):
            add_candidate(left, left + 1)

        while candidates:
            rank, _, left, right = heapq.heappop(candidates)

            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or self.ranks.get((tokens[left], tokens[right])) != rank
            ):
                continue

            before = previous[left]
            after = following[right]

            tokens[left] = 256 + rank
            following[left] = after

            if after != -1:
                previous[after] = left

            alive[right] = False
            previous[right] = -1
            following[right] = -1

            add_candidate(before, left)
            add_candidate(left, after)

        result = []
        node = 0
        while node != -1:
            if alive[node]:
                result.append(tokens[node])
            node = following[node]

        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode strictly validated integer token IDs into normalized text."""
        pieces = self._pieces

        try:
            iterator = iter(token_ids)
        except TypeError as exc:
            raise TypeError("Token IDs must be an iterable of integers") from exc

        output = []
        for token in iterator:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError("Token IDs must be integers, not booleans or other types")
            if token < 0 or token >= len(pieces):
                raise ValueError(f"Unknown tokenizer token ID: {token}")
            output.append(pieces[token])

        try:
            return b"".join(output).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Token IDs do not decode to valid UTF-8 text") from exc

    def decode_generated(self, token_ids: Iterable[int]) -> str:
        """Decode model output, replacing incomplete or invalid UTF-8 sequences."""
        pieces = self._pieces

        try:
            iterator = iter(token_ids)
        except TypeError as exc:
            raise TypeError("Token IDs must be an iterable of integers") from exc

        output = []
        for token in iterator:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError("Token IDs must be integers, not booleans or other types")
            if token < 0 or token >= len(pieces):
                raise ValueError(f"Unknown tokenizer token ID: {token}")
            output.append(pieces[token])

        return b"".join(output).decode("utf-8", errors="replace")

    def measure(self, texts: Iterable[str]) -> TokenizerMetrics:
        words = 0
        tokens = 0
        characters = 0
        round_trip = 0
        covered_characters = 0
        total_characters = 0
        covered_bytes = 0
        total_bytes = 0
        samples = 0
        for text in texts:
            normalized = self.normalize(text)
            if not normalized:
                continue
            ids = self.encode(normalized)
            decoded = self.decode(ids)
            expected_bytes = normalized.encode("utf-8")
            decoded_bytes = decoded.encode("utf-8")
            total_characters += len(normalized)
            total_bytes += len(expected_bytes)
            if decoded == normalized:
                covered_characters += len(normalized)
                covered_bytes += len(expected_bytes)
                round_trip += 1
            else:
                # Byte-level tokenization has no unknown-token ID. Count only
                # characters that survive the actual encode/decode round trip.
                matched = 0
                for expected, actual in zip(normalized, decoded):
                    if expected != actual:
                        break
                    matched += 1
                covered_characters += matched
                covered_bytes += len(decoded_bytes)
            words += len(normalized.split())
            tokens += len(ids)
            characters += len(normalized)
            samples += 1
        return {
            "samples": samples,
            "words": words,
            "tokens": tokens,
            "characters": characters,
            "fertility_tokens_per_word": tokens / max(words, 1),
            "characters_per_token": characters / max(tokens, 1),
            "character_coverage_rate": (
                covered_characters / total_characters if total_characters else 0.0
            ),
            "byte_coverage_rate": (
                min(covered_bytes, total_bytes) / total_bytes if total_bytes else 0.0
            ),
            "unknown_characters": total_characters - covered_characters,
            "round_trip_rate": round_trip / max(samples, 1),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"format": "aviy-byte-bpe-v1", "merges": self.merges},
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Tokenizer file is unreadable or invalid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("Tokenizer payload must be a JSON object")
        if payload.get("format") != "aviy-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise TypeError("Tokenizer merges must be a list")
        merges: list[tuple[int, int]] = []
        for index, pair in enumerate(raw_merges):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    isinstance(token, bool) or not isinstance(token, int)
                    for token in pair
                )
            ):
                raise ValueError(f"Invalid serialized BPE merge at index {index}")
            merges.append((pair[0], pair[1]))
        return cls(merges)
