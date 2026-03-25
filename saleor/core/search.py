import re
from typing import TYPE_CHECKING

from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F, Value

from .utils.text import strip_accents

if TYPE_CHECKING:
    from django.db.models import QuerySet


def _sanitize_word(word: str) -> str:
    """Remove PostgreSQL tsquery metacharacters from a word.

    Replaces special characters that have meaning in tsquery syntax with spaces,
    so that e.g. "22:20" becomes "22 20" (two separate tokens) rather than "2220".

    Preserved: alphanumeric, underscore, hyphen, @, period
    Replaced with space: parentheses, &, |, !, :, <, >, ', *
    """
    return re.sub(r"[()&|!:<>\'*]", " ", word).strip()


def _tokenize(value: str) -> list[dict]:
    """Tokenize a search string into structured tokens.

    Recognizes:
    - Quoted phrases: "green tea"
    - OR operator: coffee OR tea
    - Negation: -decaf
    - Regular words: coffee
    """
    tokens: list[dict] = []
    i = 0

    while i < len(value):
        # Skip whitespace
        if value[i].isspace():
            i += 1
            continue

        # Check for negation prefix
        negated = False
        if value[i] == "-" and i + 1 < len(value) and not value[i + 1].isspace():
            negated = True
            i += 1

        # Quoted phrase
        if i < len(value) and value[i] == '"':
            end = value.find('"', i + 1)
            if end == -1:
                phrase = value[i + 1 :]
                i = len(value)
            else:
                phrase = value[i + 1 : end]
                i = end + 1
            words = []
            for w in phrase.split():
                sanitized = _sanitize_word(w)
                words.extend(sanitized.split())
            if words:
                tokens.append({"type": "phrase", "words": words, "negated": negated})
            continue

        # Read the next word
        end = i
        while end < len(value) and not value[end].isspace():
            end += 1
        raw_word = value[i:end]
        i = end

        # OR operator (must not be negated)
        if raw_word == "OR" and not negated:
            tokens.append({"type": "or"})
            continue

        # AND operator – explicit AND is a no-op (AND is implicit between terms)
        if raw_word == "AND" and not negated:
            continue

        sanitized = _sanitize_word(raw_word)
        for word in sanitized.split():
            tokens.append({"type": "word", "word": word, "negated": negated})

    return tokens


def _format_token(token: dict) -> str:
    neg = "!" if token["negated"] else ""
    if token["type"] == "word":
        return f"{neg}{token['word']}:*"
    if token["type"] == "phrase":
        words = token["words"]
        if len(words) == 1:
            return f"{neg}{words[0]}"
        phrase_tsquery = " <-> ".join(words)
        return f"!({phrase_tsquery})" if neg else f"({phrase_tsquery})"
    return ""


def parse_search_query(value: str) -> str | None:
    """Parse a search string into a raw PostgreSQL tsquery with prefix matching.

    Supports websearch-compatible syntax:
    - Multiple words (implicit AND): "coffee shop" -> coffee:* & shop:*
    - OR operator: "coffee OR tea" -> coffee:* | tea:*
    - Negation: "-decaf" -> !decaf:*
    - Quoted phrases (exact match): '"green tea"' -> (green <-> tea)

    Returns:
        A raw tsquery string, or None if the input yields no searchable terms.

    """
    value = value.strip()
    if not value:
        return None

    tokens = _tokenize(value)
    if not tokens:
        return None

    # Group OR-connected terms into segments so we can parenthesize them.
    # OR connects only the immediately adjacent terms, so "A OR B C" means
    # (A | B) & C, not A | (B & C).
    #
    # Algorithm: walk tokens, track which terms are preceded by OR, then
    # merge OR-connected terms into the same segment.
    terms: list[tuple[dict, bool]] = []  # (token, preceded_by_or)
    preceded_by_or = False
    for token in tokens:
        if token["type"] == "or":
            preceded_by_or = True
            continue
        terms.append((token, preceded_by_or))
        preceded_by_or = False

    # Build segments: consecutive OR-connected terms share a segment
    segments: list[list[dict]] = []
    for token, is_or_connected in terms:
        if is_or_connected and segments:
            segments[-1].append(token)
        else:
            segments.append([token])

    # Format each segment; parenthesize multi-term OR segments only when
    # there are also AND-connected segments (to avoid unnecessary parens)
    multiple_segments = len(segments) > 1
    formatted_segments: list[str] = []
    for segment in segments:
        parts = [_format_token(t) for t in segment]
        if len(parts) == 1:
            formatted_segments.append(parts[0])
        else:
            or_expr = " | ".join(parts)
            formatted_segments.append(f"({or_expr})" if multiple_segments else or_expr)

    result = " & ".join(formatted_segments)
    return result if result else None


def prefix_search(qs: "QuerySet", value: str) -> "QuerySet":
    """Apply prefix-based search to a queryset with perfect match prioritization.

    Supports websearch-compatible syntax (AND, OR, negation, quoted phrases)
    while adding prefix matching so partial words produce results.

    Scoring: exact (websearch) matches get 2x weight, prefix matches get 1x.

    The queryset must have a ``search_vector`` SearchVectorField.
    """
    if not value:
        # return a original queryset annotated with search_rank=0
        # to allow default RANK sorting
        return qs.annotate(search_rank=Value(0))

    value = strip_accents(value)

    parsed_query = parse_search_query(value)
    if not parsed_query:
        # return empty queryset as the provided value is not searchable
        # annotated with search_rank=0 to allow default RANK sorting
        return qs.annotate(search_rank=Value(0)).none()

    # Prefix query – broadens matching via :*
    prefix_query = SearchQuery(parsed_query, search_type="raw", config="simple")

    # Exact (websearch) query – used only for ranking, not filtering
    exact_query = SearchQuery(value, search_type="websearch", config="simple")

    qs = qs.filter(search_vector=prefix_query).annotate(
        prefix_rank=SearchRank(F("search_vector"), prefix_query),
        exact_rank=SearchRank(F("search_vector"), exact_query),
        search_rank=F("exact_rank") * 2 + F("prefix_rank"),
    )

    return qs
