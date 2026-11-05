"""
Detyra 1 – Social Golfers Problem me DFS dhe Backtracking.

Përshkrimi
----------
n = g × p golferë luajnë çdo javë të ndarë në g grupe me nga p golferë.
Synimi është të ndërtohen sa më shumë javë, ashtu që asnjë çift golferësh
të mos luajë bashkë më shumë se një herë.

Modelimi
--------
Variables:   grupet e çdo jave, G[w][j] (java w, grupi j).
Domain:      çdo G[w][j] merr vlerë nga kombinimet C(n, p) të golferëve.
Constraints: (1) çdo golfer shfaqet saktësisht një herë në javë;
             (2) çdo grup ka saktësisht p golferë;
             (3) asnjë çift nuk përsëritet nëpër javë.
Objective:   maksimizimi i numrit të javëve të vlefshme w.

Kërkimi (DFS me Backtracking)
-----------------------------
Një nyje e pemës së kërkimit është një orar i pjesshëm. Një hap (degë)
shton një grup të ri në javën aktuale. Kur java mbushet, fillon java e radhës.
Nëse asnjë grup nuk mund të shtohet, algoritmi zhbën grupin e fundit
(Backtracking) dhe provon alternativën e radhës – edhe nëpër javë të mëparshme.

Kandidatët (grupet) gjenerohen në mënyrë "lazy" vetëm për nyjen aktuale;
nuk gjenerohen paraprakisht as C(n, p) grupet dhe as ndarjet javore.

Thyerja e simetrisë (Symmetry Breaking) – nuk humbet asnjë zgjidhje:
  * Java 1 fiksohet në formën kanonike {1..p}, {p+1..2p}, ...
    (çdo orar mund të riemërtohet që java e parë të duket kështu).
  * Renditja e grupeve brenda javës nuk ka rëndësi: grupi i radhës gjithmonë
    ndërtohet rreth një golferi "pivot" ende të pacaktuar, kështu çdo ndarje
    javore arrihet vetëm një herë (jo g! herë).
  * Golferët brenda grupit merren me indeks rritës (një grup = një bashkësi).

Renditja e variablave (opsionale):
  "mrv"    – pivot = golferi me më pak partnerë të mundshëm (fail-first);
  "lowest" – pivot = golferi me numrin më të vogël.
Renditja e vlerave: deterministe (seed=None) ose e rastësishme me seed.
Asnjëra nuk e ndryshon algoritmin: mbetet DFS me Backtracking i plotë.

Krasitja (Pruning) – Forward Checking:
  pas shtimit të një grupi, çdo golfer i pacaktuar i javës duhet të ketë
  ende të paktën p − 1 partnerë të pacaktuar me të cilët nuk ka luajtur.

Numëruesit:
  generated – golferë kandidatë të provuar për një vend në grup;
  rejected  – kandidatë që krijojnë çift të përsëritur, ose grupe të plota
              që dështojnë forward checking;
  explored  – grupe të pranuara dhe të vendosura (nyje të pemës);
  valid     – javë të plota të ndërtuara gjatë kërkimit;
  backtracks– grupe të zhbëra.

Strukturat e të dhënave: bitmask-a (int) për golferët e përdorur në javë
dhe për partnerët e takuar nga secili golfer.

Brenda kodit golferët indeksohen 0..n-1; në rezultat paraqiten 1..n.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import comb, factorial

from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import (
    validate_golfer_config,
    validate_non_negative_int,
    validate_positive_int,
    validate_search_limits,
)

DEFAULT_NUM_GOLFERS = 32
DEFAULT_NUM_GROUPS = 8
DEFAULT_GROUP_SIZE = 4
DEFAULT_MAX_NODES = 200_000
DEFAULT_TIMEOUT_SECONDS = 20.0
PIVOT_RULES = ("mrv", "lowest")

Week = list[tuple[int, ...]]
Schedule = list[Week]


class InvalidConfigurationError(ValueError):
    """Konfigurim i pavlefshëm; `errors` përmban mesazhet në shqip."""

    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Landscape / hapësira e kërkimit
# ---------------------------------------------------------------------------

def compute_landscape(num_golfers: int, num_groups: int, group_size: int) -> dict:
    """
    Llogarit madhësitë teorike të problemit.

    pair_bound   = floor(C(n,2) / (g · C(p,2)))  – kufiri sipas numrit të çifteve
    golfer_bound = floor((n−1) / (p−1))          – çdo golfer takon p−1 të rinj në javë
    upper_bound  = min(pair_bound, golfer_bound) – kufiri i sipërm teorik

    Kufiri i sipërm NUK garanton që ai numër javësh është i arritshëm.
    """
    errors = validate_golfer_config(num_golfers, num_groups, group_size)
    if errors:
        raise InvalidConfigurationError(errors)

    possible_groups = comb(num_golfers, group_size)
    pairs_per_group = comb(group_size, 2)
    pairs_per_week = num_groups * pairs_per_group
    total_pairs = comb(num_golfers, 2)
    pair_bound = total_pairs // pairs_per_week
    golfer_bound = (num_golfers - 1) // (group_size - 1)
    # Numri i ndarjeve të ndryshme të n golferëve në g grupe të pa-renditura
    weekly_partitions = factorial(num_golfers) // (
        factorial(group_size) ** num_groups * factorial(num_groups)
    )
    return {
        "num_golfers": num_golfers,
        "num_groups": num_groups,
        "group_size": group_size,
        "possible_groups": possible_groups,
        "pairs_per_group": pairs_per_group,
        "pairs_per_week": pairs_per_week,
        "total_pairs": total_pairs,
        "pair_bound": pair_bound,
        "golfer_bound": golfer_bound,
        "upper_bound": min(pair_bound, golfer_bound),
        "weekly_partitions": weekly_partitions,
    }


# ---------------------------------------------------------------------------
# Rezultati
# ---------------------------------------------------------------------------

@dataclass
class SocialGolfersResult:
    num_golfers: int
    num_groups: int
    group_size: int
    requested_weeks: int
    schedule: Schedule                 # golferët 1..n
    landscape: dict
    stats: SearchStats
    target_reached: bool
    optimality_proven: bool
    optimality_message: str
    is_valid: bool
    validation_errors: list[str]
    best_found_at_explored: int = 0    # nyjet e zgjeruara kur u gjet orari më i mirë
    best_found_at_seconds: float = 0.0
    seed: int | None = None
    pivot_rule: str = "mrv"
    trace: list[dict] = field(default_factory=list)

    @property
    def weeks_constructed(self) -> int:
        return len(self.schedule)

    @property
    def upper_bound(self) -> int:
        return self.landscape["upper_bound"]


# ---------------------------------------------------------------------------
# DFS me Backtracking
# ---------------------------------------------------------------------------

def _bits(mask: int) -> list[int]:
    """Kthen indekset e biteve aktive të një bitmask-e, në rend rritës."""
    result = []
    while mask:
        low = mask & -mask
        result.append(low.bit_length() - 1)
        mask ^= low
    return result


def _group_mask(group) -> int:
    mask = 0
    for golfer in group:
        mask |= 1 << golfer
    return mask


def solve_social_golfers(
    num_golfers: int = DEFAULT_NUM_GOLFERS,
    num_groups: int = DEFAULT_NUM_GROUPS,
    group_size: int = DEFAULT_GROUP_SIZE,
    requested_weeks: int | None = None,
    max_nodes: int | None = DEFAULT_MAX_NODES,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    seed: int | None = None,
    trace_limit: int = 0,
    pivot_rule: str = "mrv",
) -> SocialGolfersResult:
    """
    Ndërton orarin me DFS dhe Backtracking.

    requested_weeks – numri i javëve që kërkohen (parazgjedhje: kufiri i sipërm).
    max_nodes       – kufiri i nyjeve të zgjeruara (None = pa kufi).
    timeout_seconds – kufiri i kohës (None = pa kufi).
    seed            – None: renditje deterministe e kandidatëve;
                      numër: renditje e rastësishme e riprodhueshme.
    trace_limit     – sa ngjarje të para të kërkimit të ruhen për vizualizim.

    Nëse kërkimi ndërpritet, kthehet orari më i mirë i vlefshëm i gjetur.
    """
    errors = validate_golfer_config(num_golfers, num_groups, group_size)
    errors += validate_search_limits(max_nodes, timeout_seconds)
    if errors:
        raise InvalidConfigurationError(errors)

    landscape = compute_landscape(num_golfers, num_groups, group_size)
    upper_bound = landscape["upper_bound"]
    if requested_weeks is None:
        requested_weeks = upper_bound
    week_errors = validate_positive_int(requested_weeks, "Numri i javëve")
    if week_errors:
        raise InvalidConfigurationError(week_errors)
    if requested_weeks > upper_bound:
        raise InvalidConfigurationError([
            f"Numri i kërkuar i javëve ({requested_weeks}) e tejkalon kufirin e sipërm "
            f"teorik ({upper_bound}); një orar i tillë nuk mund të ekzistojë."
        ])
    trace_errors = validate_non_negative_int(trace_limit, "Kufiri i gjurmës")
    if trace_errors:
        raise InvalidConfigurationError(trace_errors)
    if pivot_rule not in PIVOT_RULES:
        raise InvalidConfigurationError([
            f"Rregull i panjohur i pivot-it: {pivot_rule}. Lejohen: {', '.join(PIVOT_RULES)}."
        ])

    n, g, p = num_golfers, num_groups, group_size
    full_mask = (1 << n) - 1
    rng = random.Random(seed) if seed is not None else None
    stats = SearchStats(max_nodes=max_nodes, timeout_seconds=timeout_seconds)
    trace: list[dict] = []

    # met[i] – bitmask e golferëve me të cilët i ka luajtur tashmë
    met = [0] * n
    completed: Schedule = []        # javët e plota (0..n-1)
    current_week: Week = []         # grupet e javës në ndërtim
    used = 0                        # golferët e caktuar në javën aktuale
    best: Schedule = []
    # kur u gjet orari më i mirë (numëruesit realë në atë moment)
    best_found_at = {"explored": 0, "seconds": 0.0}

    def log(event: str, group=None) -> None:
        if len(trace) < trace_limit:
            trace.append({
                "event": event,
                "week": len(completed) + 1,
                "group_index": len(current_week) + 1,
                "group": tuple(x + 1 for x in group) if group is not None else None,
                # thellësia = numri i grupeve të vendosura nga kërkimi (pa javën 1 fikse)
                "depth": len(completed) * g + len(current_week) - g,
            })

    def add_pairs(group) -> None:
        for a, b in combinations(group, 2):
            met[a] |= 1 << b
            met[b] |= 1 << a

    def remove_pairs(group) -> None:
        # Çiftet nuk përsëriten kurrë, prandaj fshirja e bitit është e saktë.
        for a, b in combinations(group, 2):
            met[a] &= ~(1 << b)
            met[b] &= ~(1 << a)

    def place(group) -> None:
        nonlocal used, current_week, best
        add_pairs(group)
        current_week.append(group)
        used |= _group_mask(group)
        if used == full_mask:                     # java u plotësua
            completed.append(current_week)
            current_week = []
            used = 0
            stats.valid += 1
            if len(completed) > len(best):
                best = [list(week) for week in completed]
                best_found_at["explored"] = stats.explored
                best_found_at["seconds"] = stats.elapsed_now()

    def unplace() -> None:
        nonlocal used, current_week
        if not current_week:                      # zhbëjmë mbylljen e javës
            current_week = completed.pop()
            used = full_mask
        group = current_week.pop()
        used &= ~_group_mask(group)
        remove_pairs(group)
        stats.backtracks += 1
        log("backtrack", group)

    def forward_check_ok(new_used: int) -> bool:
        """
        Forward checking: çdo golfer ende i pacaktuar në javë duhet të ketë
        të paktën p − 1 partnerë të pacaktuar me të cilët nuk ka luajtur.
        (Çiftet e grupit të ri nuk prekin golferët e pacaktuar.)
        """
        remaining = full_mask & ~new_used
        for x in _bits(remaining):
            available = remaining & ~met[x] & ~(1 << x)
            if available.bit_count() < p - 1:
                return False
        return True

    def candidate_groups():
        """
        Gjeneron (lazy) grupet e vlefshme për hapin e radhës.

        Zgjidhet një golfer "pivot" i pacaktuar dhe ndërtohen të gjitha grupet
        që e përmbajnë atë. Meqë pivot-i duhet patjetër të jetë në njërin grup
        të javës, çdo ndarje javore arrihet saktësisht një herë (nuk humbet
        asnjë zgjidhje dhe nuk përsëritet asnjë permutacion i grupeve).
          - "mrv":    pivot = golferi me më pak partnerë të mundshëm (fail-first)
          - "lowest": pivot = golferi me numrin më të vogël; anëtarët e tjerë > pivot
        """
        unassigned = full_mask & ~used
        if pivot_rule == "mrv":
            # Fail-first: golferi me më pak partnerë të mundshëm në javën aktuale
            leader = min(
                _bits(unassigned),
                key=lambda x: ((unassigned & ~met[x]).bit_count(), x),
            )
            candidates = unassigned & ~met[leader] & ~(1 << leader)
        else:
            leader = (unassigned & -unassigned).bit_length() - 1
            candidates = unassigned & ~met[leader] & ~((1 << (leader + 1)) - 1)
        pool = _bits(candidates)
        if rng is not None:
            rng.shuffle(pool)
        yield from extend_group([leader], 1 << leader, pool, 0)

    def extend_group(members: list[int], members_mask: int, pool: list[int], start: int):
        """
        Ndërton grupin golfer pas golferi (çdo bashkësi gjenerohet një herë,
        sepse anëtarët merren nga `pool` vetëm me indeks rritës).

        Çdo golfer i provuar numërohet si gjendje e gjeneruar. Refuzohet nëse
        ka luajtur tashmë me ndonjë anëtar (kufizimi 3). Grupi i plotë refuzohet
        nëse dështon forward checking.
        """
        if len(members) == p:
            if not forward_check_ok(used | members_mask):
                stats.rejected += 1
                log("prune", tuple(sorted(members)))
                return
            yield tuple(sorted(members))
            return
        needed = p - len(members)
        for i in range(start, len(pool) - needed + 1):
            x = pool[i]
            stats.generated += 1
            if met[x] & members_mask:
                stats.rejected += 1
                continue
            members.append(x)
            yield from extend_group(members, members_mask | (1 << x), pool, i + 1)
            members.pop()

    stats.start()

    # Thyerja e simetrisë: java 1 kanonike
    first_week = [tuple(range(j * p, (j + 1) * p)) for j in range(g)]
    for group in first_week:
        place(group)

    reason = TERMINATION_EXHAUSTED
    if len(completed) >= requested_weeks:
        reason = TERMINATION_GOAL_REACHED
    else:
        # Steka e DFS: çdo element është iteratori i kandidatëve të një nyjeje
        stack = [candidate_groups()]
        try:
            while stack:
                group = next(stack[-1], None)
                if group is None:
                    # Asnjë alternativë tjetër: Backtracking
                    stack.pop()
                    if stack:
                        unplace()
                    continue
                log("place", group)
                place(group)
                stats.record_explored(depth=len(stack))
                if len(completed) >= requested_weeks:
                    reason = TERMINATION_GOAL_REACHED
                    break
                stack.append(candidate_groups())
        except SearchLimitReached as limit:
            reason = limit.reason

    stats.stop(reason)

    # Për paraqitje: grupet e çdo jave renditen sipas golferit më të vogël
    schedule = [sorted(tuple(x + 1 for x in group) for group in week) for week in best]
    is_valid, validation_errors = verify_schedule(schedule, n, g, p)
    weeks_found = len(schedule)
    target_reached = weeks_found >= requested_weeks

    if is_valid and weeks_found == upper_bound:
        optimality_proven = True
        optimality_message = (
            f"Optimaliteti është vërtetuar: orari i validuar ka {weeks_found} javë, "
            f"aq sa kufiri i sipërm teorik."
        )
    elif is_valid and reason == TERMINATION_EXHAUSTED:
        optimality_proven = True
        optimality_message = (
            f"Optimaliteti është vërtetuar me kërkim të plotë: maksimumi është "
            f"{weeks_found} javë (kufiri teorik {upper_bound} nuk është i arritshëm)."
        )
    elif reason == TERMINATION_GOAL_REACHED:
        optimality_proven = False
        optimality_message = (
            f"U arritën {weeks_found} javët e kërkuara. Kërkimi u ndal këtu, "
            f"prandaj nuk është provuar nëse ekzistojnë më shumë (deri në {upper_bound})."
        )
    else:
        optimality_proven = False
        optimality_message = (
            f"{stats.termination_message} Orari më i mirë i gjetur ka "
            f"{weeks_found} javë. Optimaliteti NUK është vërtetuar; kjo nuk provon "
            f"që më shumë javë janë të pamundura."
        )

    return SocialGolfersResult(
        num_golfers=n,
        num_groups=g,
        group_size=p,
        requested_weeks=requested_weeks,
        schedule=schedule,
        landscape=landscape,
        stats=stats,
        target_reached=target_reached,
        optimality_proven=optimality_proven,
        optimality_message=optimality_message,
        is_valid=is_valid,
        validation_errors=validation_errors,
        best_found_at_explored=best_found_at["explored"],
        best_found_at_seconds=best_found_at["seconds"],
        seed=seed,
        pivot_rule=pivot_rule,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Validimi i pavarur
# ---------------------------------------------------------------------------

def verify_schedule(
    schedule: Schedule, num_golfers: int, num_groups: int, group_size: int
) -> tuple[bool, list[str]]:
    """
    Verifikon një orar në mënyrë të pavarur nga algoritmi i kërkimit.

    Golferët pritet të jenë 1..n. Kontrollohen:
      - numri i grupeve në çdo javë;
      - madhësia e çdo grupi;
      - golferë të vlefshëm dhe të pa-përsëritur brenda javës;
      - çdo golfer saktësisht një herë në javë;
      - asnjë çift që përsëritet nëpër javë.
    Kthen (is_valid, lista_e_gabimeve).
    """
    errors: list[str] = []
    all_golfers = set(range(1, num_golfers + 1))
    pair_counts: Counter = Counter()

    for w, week in enumerate(schedule, start=1):
        if len(week) != num_groups:
            errors.append(f"Java {w}: ka {len(week)} grupe në vend të {num_groups}.")
        seen: Counter = Counter()
        for j, group in enumerate(week, start=1):
            if len(group) != group_size:
                errors.append(
                    f"Java {w}, grupi {j}: ka {len(group)} golferë në vend të {group_size}."
                )
            for golfer in group:
                if golfer not in all_golfers:
                    errors.append(f"Java {w}, grupi {j}: golfer i pavlefshëm {golfer}.")
                seen[golfer] += 1
            for a, b in combinations(sorted(group), 2):
                pair_counts[(a, b)] += 1
        duplicates = sorted(x for x, c in seen.items() if c > 1)
        missing = sorted(all_golfers - set(seen))
        if duplicates:
            errors.append(f"Java {w}: golferë që shfaqen më shumë se një herë: {duplicates}.")
        if missing:
            errors.append(f"Java {w}: golferë që mungojnë: {missing}.")

    repeated = sorted(pair for pair, c in pair_counts.items() if c > 1)
    for a, b in repeated:
        errors.append(
            f"Çifti ({a}, {b}) luan bashkë {pair_counts[(a, b)]} herë."
        )
    return (not errors, errors)


# ---------------------------------------------------------------------------
# Përgatitja e rezultateve për ndërfaqen
# ---------------------------------------------------------------------------

def schedule_to_rows(schedule: Schedule) -> list[dict]:
    """Tabelë e plotë: një rresht për çdo grup të çdo jave."""
    return [
        {
            "Java": w,
            "Grupi": j,
            "Golferët": ", ".join(str(x) for x in group),
        }
        for w, week in enumerate(schedule, start=1)
        for j, group in enumerate(week, start=1)
    ]


def pair_week_matrix(schedule: Schedule, num_golfers: int) -> list[list[int]]:
    """
    Matricë n×n për vizualizim: m[a][b] = java (1..w) kur golferët a+1 dhe b+1
    luajtën bashkë; 0 = asnjëherë; −1 = luajtën bashkë më shumë se një herë.
    """
    matrix = [[0] * num_golfers for _ in range(num_golfers)]
    for w, week in enumerate(schedule, start=1):
        for group in week:
            for a, b in combinations(group, 2):
                for x, y in ((a - 1, b - 1), (b - 1, a - 1)):
                    matrix[x][y] = w if matrix[x][y] == 0 else -1
    return matrix


def schedule_to_matrix(schedule: Schedule) -> list[list[str]]:
    """Matricë javë × grupe; çdo qelizë përmban golferët e grupit."""
    return [["-".join(str(x) for x in group) for group in week] for week in schedule]


# ---------------------------------------------------------------------------
# Ekzekutim i pavarur:  python -m tasks.social_golfers
# ---------------------------------------------------------------------------

def _main() -> None:
    land = compute_landscape(DEFAULT_NUM_GOLFERS, DEFAULT_NUM_GROUPS, DEFAULT_GROUP_SIZE)
    print("Social Golfers Problem – DFS me Backtracking")
    print(f"n={land['num_golfers']}, g={land['num_groups']}, p={land['group_size']}")
    print(f"C(n,2) = {land['total_pairs']}, çifte në javë = {land['pairs_per_week']}, "
          f"kufiri i sipërm = {land['upper_bound']} javë")

    result = solve_social_golfers(seed=1)
    s = result.stats
    print(f"\nJavë të ndërtuara: {result.weeks_constructed} / {result.requested_weeks}")
    print(f"Përfundimi: {s.termination_message}")
    print(f"Generated={s.generated}, Explored={s.explored}, Rejected={s.rejected}, "
          f"Backtracks={s.backtracks}, Koha={s.elapsed_seconds:.2f}s")
    print(f"Validimi i pavarur: {'I VLEFSHËM' if result.is_valid else 'I PAVLEFSHËM'}")
    print(result.optimality_message)
    for w, week in enumerate(result.schedule, start=1):
        print(f"Java {w:2d}: " + "  ".join("-".join(map(str, grp)) for grp in week))


if __name__ == "__main__":
    _main()
