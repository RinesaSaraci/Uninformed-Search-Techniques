"""
Validimi i hyrjeve të përdoruesit dhe i konfigurimeve të problemeve.

Funksionet kthejnë një listë mesazhesh gabimi në shqip.
Lista bosh do të thotë që konfigurimi është i vlefshëm.
"""

from __future__ import annotations

MAX_GOLFERS = 64


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_search_limits(max_nodes, timeout_seconds) -> list[str]:
    """Validon kufijtë e përbashkët të kërkimit."""
    errors: list[str] = []
    if max_nodes is not None and (not _is_int(max_nodes) or max_nodes < 1):
        errors.append("Kufiri i nyjeve duhet të jetë numër i plotë pozitiv.")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            errors.append("Timeout duhet të jetë numër.")
        elif timeout_seconds <= 0:
            errors.append("Timeout duhet të jetë më i madh se 0 sekonda.")
    return errors


def validate_golfer_config(num_golfers, num_groups, group_size) -> list[str]:
    """Validon parametrat n, g, p të Social Golfers Problem."""
    errors: list[str] = []
    for value, label in (
        (num_golfers, "Numri i golferëve (n)"),
        (num_groups, "Numri i grupeve (g)"),
        (group_size, "Madhësia e grupit (p)"),
    ):
        if not _is_int(value):
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
