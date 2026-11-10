"""Text helpers: comparison-friendly normalisation, Ukrainian lemmas and full-text literals.

Postgres has no Ukrainian dictionary, so lemmas come from pymorphy3 here and reach the database as
ready tsvector and tsquery literals. That keeps search independent of the database's locale and
text parser, which splits words at apostrophes ("зобов'язання" becomes two words).
"""

import re
import unicodedata

_APOSTROPHES = "'’ʼ`´‘"
_QUOTES = '«»„“”"'
_DASHES = "‐‑‒–—―−"
_SPACES = "    "

# Words that carry no meaning for search, compared after lemmatisation.
_STOPWORD_TEXT = """
а або адже але аж б би бо бути в вам вас весь ви від він вона вони воно все всі всього вже
де для до е є ж же з за зі із й і її їх його к коли котрий лише ми між на над наш не ні о
об однак от по під після при про саме свій та так також там те тим то тобто тут у усі хоча
це цей ця ці через чи що щоб як який якщо я хто скільки чому навіщо куди могти мати самий
сам такий той мій ваш їхній
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_WORD = re.compile(r"[a-z0-9а-яіїєґ]+(?:['-][a-z0-9а-яіїєґ]+)*")
_NOT_TERM = re.compile(r"[^a-z0-9а-яіїєґ-]")
_NUMBER = re.compile(r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
# Postgres keeps at most this many positions per tsvector; later words still count, unpositioned.
_MAX_POSITION = 16383
# Readings kept per ambiguous word; more only adds noise.
_MAX_READINGS = 3


def normalize(text: str) -> str:
    """Lower-cases text and gives it one kind of apostrophe, quote, dash and space.

    For comparing evidence and answers, not for display.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    text = text.translate(
        {ord(c): "'" for c in _APOSTROPHES}
        | {ord(c): '"' for c in _QUOTES}
        | {ord(c): "-" for c in _DASHES}
        | {ord(c): " " for c in _SPACES}
    )
    return re.sub(r"\s+", " ", text).strip()


def numbers_in(text: str) -> set[str]:
    """The numbers written in text, as "28000" or "1.5", whatever the separators."""
    found = set()
    for match in _NUMBER.finditer(normalize(text)):
        found.add(match.group().replace(" ", "").replace(",", "."))
    return found


class Lemmatizer:
    """Turns Ukrainian text into lemmas for full-text search.

    pymorphy3's Ukrainian dictionary has no word frequencies, so an ambiguous form has several
    equally likely readings: "пені" is "пеня" or "пен", "гроші" is "гроші" or "гріш". Every reading
    is kept, on both the document and the question side, so the right one always meets itself.
    """

    def __init__(self) -> None:
        import pymorphy3  # loads its dictionaries, about 0.1 s

        self._morph = pymorphy3.MorphAnalyzer(lang="uk")
        self._cache: dict[str, list[str]] = {}

    def lemmas(self, word: str) -> list[str]:
        """The distinct lemmas of one lower-case word, most usual reading first."""
        cached = self._cache.get(word)
        if cached is None:
            if word.isascii():  # Latin words and digits stay as they are
                cached = [word]
            else:
                readings = [parse.normal_form for parse in self._morph.parse(word)]
                cached = list(dict.fromkeys(readings))[:_MAX_READINGS]
            if len(self._cache) < 200_000:
                self._cache[word] = cached
        return cached

    def words(self, text: str) -> list[list[str]]:
        """For each word of text that is not a stop word, its search terms.

        A hyphenated word gives its parts and the joined form, so "ризик-менеджер" matches both
        "ризикменеджер" and "менеджер". Apostrophes are dropped, so "зобов'язання" is one term.
        """
        found = []
        for word in _WORD.findall(normalize(text)):
            readings = self.lemmas(word)
            # "має" is first of all a form of "мати", a stop word; its rarer readings go with it.
            if word in STOPWORDS or readings[0] in STOPWORDS:
                continue
            terms: list[str] = []
            for lemma in readings:
                # Only these characters may reach a tsvector or tsquery literal unquoted.
                lemma = _NOT_TERM.sub("", lemma.casefold())
                if lemma in STOPWORDS:
                    continue
                parts = [part for part in lemma.split("-") if part]
                if len(parts) > 1:
                    terms.append("".join(parts))
                    terms.extend(part for part in parts if part not in STOPWORDS)
                elif parts:
                    terms.append(parts[0])
            if terms:
                found.append(list(dict.fromkeys(terms)))
        return found

    def lexemes(self, text: str) -> list[str]:
        """Every search term of text, in order."""
        return [term for terms in self.words(text) for term in terms]


def tsvector_literal(heading_words: list[list[str]], body_words: list[list[str]]) -> str:
    """A tsvector literal; each word's terms share its position. Heading words are marked A,
    body words D, so the query can weigh them differently."""
    positions: dict[str, list[str]] = {}
    for index, terms in enumerate([*heading_words, *body_words], start=1):
        weight = "A" if index <= len(heading_words) else ""
        for term in terms:
            places = positions.setdefault(term, [])
            if index <= _MAX_POSITION:
                places.append(f"{index}{weight}")
    parts = []
    for term, places in positions.items():
        parts.append(f"'{term}':{','.join(places)}" if places else f"'{term}'")
    return " ".join(parts)


def tsquery_literal(terms: list[str]) -> str | None:
    """A tsquery matching any of the terms, or None when there are none."""
    unique = list(dict.fromkeys(terms))
    if not unique:
        return None
    return " | ".join(f"'{term}'" for term in unique)
