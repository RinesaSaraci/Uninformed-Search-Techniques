"""
Ndihmës të ripërdorshëm për validimin e hyrjeve.

Të gjitha funksionet kthejnë një listë mesazhesh gabimi në shqip;
lista bosh do të thotë që hyrja është e vlefshme.

Këtu ndodhen vetëm kontrollet e përgjithshme (numra, kufij, forma matricash,
kufijtë e kërkimit, përsëritjet në një varg). Validimi kompleks që varet nga
problemi (p.sh. kufizimet e Sudoku-së, të orarit të punëtorëve) mbetet në
skedarin e detyrës përkatëse.
"""

from __future__ import annotations

MAX_GOLFERS = 64


# ---------------------------------------------------------------------------
# Numrat
# ---------------------------------------------------------------------------

def is_int(value) -> bool:
    """True për numra të plotë të vërtetë (bool përjashtohet)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_int(value, label: str, minimum: int | None = None,
                 maximum: int | None = None) -> list[str]:
    """Numër i plotë, opsionalisht brenda [minimum, maximum]."""
    if not is_int(value):
        return [f"{label} duhet të jetë numër i plotë."]
    if minimum is not None and maximum is not None and not minimum <= value <= maximum:
        return [f"{label} duhet të jetë ndërmjet {minimum} dhe {maximum}."]
    if minimum is not None and value < minimum:
        return [f"{label} duhet të jetë të paktën {minimum}."]
    if maximum is not None and value > maximum:
        return [f"{label} nuk mund të jetë më i madh se {maximum}."]
    return []


def validate_positive_int(value, label: str) -> list[str]:
    if not is_int(value) or value < 1:
        return [f"{label} duhet të jetë numër i plotë pozitiv."]
    return []


def validate_non_negative_int(value, label: str) -> list[str]:
    if not is_int(value) or value < 0:
        return [f"{label} duhet të jetë numër i plotë jo-negativ."]
    return []


def validate_optional_positive_int(value, label: str) -> list[str]:
    """None lejohet (= pa kufi); përndryshe numër i plotë pozitiv."""
    return [] if value is None else validate_positive_int(value, label)


# ---------------------------------------------------------------------------
# Kufijtë e kërkimit
# ---------------------------------------------------------------------------

def validate_search_limits(max_nodes, timeout_seconds) -> list[str]:
    """Kufiri i nyjeve (None ose > 0) dhe timeout (None ose > 0 sekonda)."""
    errors: list[str] = []
    if max_nodes is not None and (not is_int(max_nodes) or max_nodes < 1):
        errors.append("Kufiri i nyjeve duhet të jetë numër i plotë pozitiv.")
    if timeout_seconds is not None:
        if not is_number(timeout_seconds):
            errors.append("Timeout duhet të jetë numër.")
        elif timeout_seconds <= 0:
            errors.append("Timeout duhet të jetë më i madh se 0 sekonda.")
    return errors


# ---------------------------------------------------------------------------
# Matricat dhe vargjet
# ---------------------------------------------------------------------------

def validate_matrix_shape(matrix, rows: int, cols: int, label: str = "Matrica") -> list[str]:
    """Kontrollon që `matrix` është listë me `rows` rreshta me nga `cols` elemente."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) != rows:
        return [f"{label} duhet të ketë saktësisht {rows} rreshta."]
    errors = []
    for i, row in enumerate(matrix, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            errors.append(f"Rreshti {i} duhet të ketë saktësisht {cols} qeliza.")
    return errors


def validate_matrix_values(matrix, minimum: int, maximum: int,
                           empty_value: int | None = None) -> list[str]:
    """
    Çdo qelizë duhet të jetë numër i plotë në [minimum, maximum].
    Supozon që forma është kontrolluar më parë.
    """
    errors = []
    for i, row in enumerate(matrix, start=1):
        for j, value in enumerate(row, start=1):
            if not is_int(value):
                errors.append(f"Qeliza ({i}, {j}) duhet të jetë numër i plotë.")
            elif not minimum <= value <= maximum:
                hint = f" ({empty_value} = bosh)" if empty_value is not None else ""
                errors.append(
                    f"Qeliza ({i}, {j}) ka vlerë {value}; lejohen {minimum}–{maximum}{hint}."
                )
    return errors


def find_duplicates(values, ignore=None) -> list:
    """Vlerat që shfaqen më shumë se një herë (përveç `ignore`), të renditura."""
    seen, duplicates = set(), set()
    for value in values:
        if value == ignore:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def is_permutation(values, expected) -> bool:
    """A përmban `values` secilën vlerë nga `expected` saktësisht një herë?"""
    values = list(values)
    return len(values) == len(expected) and set(values) == set(expected)


def matrix_columns(matrix) -> list[list]:
    return [list(column) for column in zip(*matrix)] if matrix else []


# ---------------------------------------------------------------------------
# Social Golfers – konfigurimi
# ---------------------------------------------------------------------------

def validate_golfer_config(num_golfers, num_groups, group_size) -> list[str]:
    """Validon parametrat n, g, p të Social Golfers Problem."""
    errors: list[str] = []
    for value, label in (
        (num_golfers, "Numri i golferëve (n)"),
        (num_groups, "Numri i grupeve (g)"),
        (group_size, "Madhësia e grupit (p)"),
    ):
        if not is_int(value):
            errors.append(f"{label} duhet të jetë numër i plotë.")
    if errors:
        return errors

    if num_groups < 1:
        errors.append("Numri i grupeve (g) duhet të jetë të paktën 1.")
    if group_size < 2:
        errors.append("Madhësia e grupit (p) duhet të jetë të paktën 2 (përndryshe nuk ka çifte).")
    if num_golfers < 2:
        errors.append("Numri i golferëve (n) duhet të jetë të paktën 2.")
    if num_golfers > MAX_GOLFERS:
        errors.append(f"Numri i golferëve (n) nuk mund të jetë më i madh se {MAX_GOLFERS}.")
    if num_golfers != num_groups * group_size:
        errors.append(
            f"Duhet të vlejë n = g × p, por {num_golfers} ≠ {num_groups} × {group_size} "
            f"= {num_groups * group_size}. Java nuk mund të ndahet plotësisht në grupe."
        )
    return errors
