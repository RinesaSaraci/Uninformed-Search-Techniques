"""
Detyra 4c – Rotating Workforce Scheduling me DFS dhe Backtracking.

Model i thjeshtuar edukativ (jo një sistem real planifikimi).

Problemi
--------
W punëtorë, D ditë, S turne në ditë (Mëngjes, Pasdite, Natë), R punëtorë
për çdo turn. Orari i plotë ka D × S × R caktime (parazgjedhje 7×3×2 = 42).

Përkufizimet
------------
Dita e punës: dita kalendarike në të cilën FILLON turni.
Oraret e turneve: Mëngjes 06–14, Pasdite 14–22, Natë 22–06 (të nesërmen).
Pushimi ndërmjet turnit t1 në ditën d dhe turnit t2 në ditën d+1:
    rest(t1, t2) = fillimi(t2) + 24 − mbarimi(t1)   [orë]
    p.sh. Natë → Mëngjes: 6 + 24 − 30 = 0 orë; Pasdite → Mëngjes: 8 orë.
Ditë të njëpasnjëshme pune: ditë kalendarike radhazi me ndonjë turn.

Variables:     caktimi X[d][s][k] = punëtori në vendin k të turnit s, ditën d.
Domain:        punëtorët {P1, ..., PW}.
Initial State: orar bosh.
Constraints:
    1. çdo turn ka saktësisht R punëtorë;
    2. një punëtor nuk punon dy turne në të njëjtën ditë;
    3. turnet e njëpasnjëshme të papajtueshme ndalohen: rest(t1, t2) < pushimi minimal;
    4. Natë → Mëngjes (dita tjetër) ndalohet gjithmonë;
    5. nuk tejkalohet numri maksimal i ditëve të njëpasnjëshme të punës;
    6. ngarkesa shpërndahet sa më drejtë (kufizim i butë – objektivi).
Goal Test:     orari i plotë që plotëson kufizimet 1–5.
Objective:     min imbalance = max(turnet e punëtorit) − min(turnet e punëtorit).
               Kufiri i poshtëm: 0 nëse D·S·R pjesëtohet me W, përndryshe 1.

DFS me Backtracking (Branch and Bound)
--------------------------------------
Vendet plotësohen në rend: ditë → turn → vend. Për çdo vend provohen
punëtorët e lirë (ata me më pak turne së pari). Caktimi që shkel një
kufizim refuzohet; nëse dega dështon, caktimi zhbëhet dhe provohet punëtori
tjetër. Pas zgjidhjes së parë kërkimi VAZHDON për orar më të drejtë:
një degë krasitet kur kufiri i poshtëm i imbalance-it të saj ≥ më i miri i gjetur.
Optimaliteti shpallet vetëm kur (a) imbalance = kufiri teorik i poshtëm,
ose (b) kërkimi eksploron gjithë hapësirën.

Thyerja e simetrisë: brenda një turni punëtorët ruhen në rend rritës
(një turn = bashkësi, jo renditje), gjë që nuk humb asnjë orar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.search_tree import SearchTreeRecorder
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import validate_int, validate_non_negative_int, validate_search_limits

# (çelësi, emri, ora e fillimit, ora e mbarimit) – mbarimi > 24 do të thotë dita tjetër
SHIFTS = (
    ("morning", "Mëngjes", 6, 14),
    ("afternoon", "Pasdite", 14, 22),
    ("night", "Natë", 22, 30),
)
SHIFT_ABBR = {"morning": "M", "afternoon": "P", "night": "N"}
NIGHT, MORNING = "night", "morning"

DEFAULT_MAX_NODES = 200_000
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_TREE_NODES = 60

PRESETS = {
    "default": {
        "label": "Parazgjedhja: 6 punëtorë, 7 ditë, 3 turne × 2",
        "config": dict(num_workers=6, num_days=7, num_shifts=3, workers_per_shift=2,
                       max_consecutive_days=7, min_rest_hours=8),
    },
    "rest_days": {
        "label": "Me ditë pushimi: 8 punëtorë, maks. 5 ditë radhazi",
        "config": dict(num_workers=8, num_days=7, num_shifts=3, workers_per_shift=2,
                       max_consecutive_days=5, min_rest_hours=8),
    },
    "backtracking": {
        "label": "Me Backtracking: 7 punëtorë, maks. 6 ditë radhazi",
        "config": dict(num_workers=7, num_days=7, num_shifts=3, workers_per_shift=2,
                       max_consecutive_days=6, min_rest_hours=8),
    },
    "infeasible": {
        "label": "Pa zgjidhje (provohet me kërkim): 7 punëtorë, maks. 5 ditë radhazi",
        "config": dict(num_workers=7, num_days=7, num_shifts=3, workers_per_shift=2,
                       max_consecutive_days=5, min_rest_hours=8),
    },
}


class WorkforceConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class WorkforceConfig:
    num_workers: int = 6
    num_days: int = 7
    num_shifts: int = 3
    workers_per_shift: int = 2
    max_consecutive_days: int = 7
    min_rest_hours: int = 8

    @property
    def shifts(self):
        return SHIFTS[: self.num_shifts]

    @property
    def total_assignments(self) -> int:
        return self.num_days * self.num_shifts * self.workers_per_shift

    @property
    def imbalance_lower_bound(self) -> int:
        return 0 if self.total_assignments % self.num_workers == 0 else 1

    def max_working_days_per_worker(self) -> int:
        """Ditë maksimale pune në D ditë kur lejohen maks. k ditë radhazi."""
        k = self.max_consecutive_days
        return self.num_days - self.num_days // (k + 1)

    def rest_hours(self, prev_shift: int, next_shift: int) -> int:
        """Orët e pushimit nga turni prev (dita d) te turni next (dita d+1)."""
        return self.shifts[next_shift][2] + 24 - self.shifts[prev_shift][3]

    def incompatible(self, prev_shift: int, next_shift: int) -> bool:
        """Kufizimet 3 dhe 4."""
        if self.shifts[prev_shift][0] == NIGHT and self.shifts[next_shift][0] == MORNING:
            return True
        return self.rest_hours(prev_shift, next_shift) < self.min_rest_hours


def validate_config(cfg: WorkforceConfig) -> list[str]:
    """Validimi i vlerave dhe kontrolli i pamundësisë para kërkimit."""
    errors: list[str] = []
    ranges = (
        (cfg.num_workers, "Numri i punëtorëve", 1, 20),
        (cfg.num_days, "Numri i ditëve", 1, 14),
        (cfg.num_shifts, "Numri i turneve", 1, 3),
        (cfg.workers_per_shift, "Punëtorë për turn", 1, 5),
        (cfg.max_consecutive_days, "Ditë maksimale radhazi", 1, 14),
        (cfg.min_rest_hours, "Pushimi minimal (orë)", 0, 24),
    )
    for value, label, low, high in ranges:
        errors += validate_int(value, label, low, high)
    if errors:
        return errors

    daily_need = cfg.num_shifts * cfg.workers_per_shift
    if cfg.num_workers < daily_need:
        errors.append(
            f"E pamundur: çdo ditë nevojiten {daily_need} punëtorë të ndryshëm "
            f"({cfg.num_shifts} turne × {cfg.workers_per_shift}), por janë vetëm {cfg.num_workers}."
        )
    capacity = cfg.num_workers * cfg.max_working_days_per_worker()
    if capacity < cfg.total_assignments:
        errors.append(
            f"E pamundur: me maks. {cfg.max_consecutive_days} ditë radhazi, çdo punëtor punon "
            f"të shumtën {cfg.max_working_days_per_worker()} nga {cfg.num_days} ditë; "
            f"kapaciteti {cfg.num_workers} × {cfg.max_working_days_per_worker()} = {capacity} "
            f"< {cfg.total_assignments} caktime të nevojshme."
        )
    return errors


def landscape(cfg: WorkforceConfig) -> dict:
    """Hapësira teorike: çdo vend merr një nga W punëtorët → W^(D·S·R)."""
    return {
        "variables": cfg.total_assignments,
        "domain_size": cfg.num_workers,
        "unrestricted_assignments": cfg.num_workers ** cfg.total_assignments,
        "imbalance_lower_bound": cfg.imbalance_lower_bound,
        "average_shifts_per_worker": cfg.total_assignments / cfg.num_workers,
    }


# ---------------------------------------------------------------------------
# Gjendja dhe kalimet
# ---------------------------------------------------------------------------

def initial_state(cfg: WorkforceConfig) -> list[list[list[int]]]:
    """Orari bosh: schedule[dita][turni] = lista e punëtorëve (0-indeks)."""
    return [[[] for _ in range(cfg.num_shifts)] for _ in range(cfg.num_days)]


def next_slot(cfg: WorkforceConfig, schedule) -> tuple[int, int] | None:
    """Turni i parë i paplotësuar (dita, turni), në rend ditë → turn."""
    for d in range(cfg.num_days):
        for s in range(cfg.num_shifts):
            if len(schedule[d][s]) < cfg.workers_per_shift:
                return d, s
    return None


def _worked_shift(cfg, schedule, worker: int, day: int) -> int | None:
    for s in range(cfg.num_shifts):
        if worker in schedule[day][s]:
            return s
    return None


def check_assignment(cfg: WorkforceConfig, schedule, day: int, shift: int, worker: int) -> str | None:
    """
    Kontrollon nëse punëtori mund të caktohet. Kthen None nëse lejohet,
    përndryshe arsyen (shqip). Supozon që ditët para `day` janë plotësuar.
    """
    if worker in schedule[day][shift]:
        return "është tashmë në këtë turn"
    if _worked_shift(cfg, schedule, worker, day) is not None:
        return "punon tashmë një turn tjetër këtë ditë (kufizimi 2)"
    if day > 0:
        prev = _worked_shift(cfg, schedule, worker, day - 1)
        if prev is not None and cfg.incompatible(prev, shift):
            if cfg.shifts[prev][0] == NIGHT and cfg.shifts[shift][0] == MORNING:
                return "Natë → Mëngjes (kufizimi 4)"
            return f"pushim {cfg.rest_hours(prev, shift)} orë < {cfg.min_rest_hours} (kufizimi 3)"
    run = 1
    d = day - 1
    while d >= 0 and _worked_shift(cfg, schedule, worker, d) is not None:
        run += 1
        d -= 1
    if run > cfg.max_consecutive_days:
        return f"{run} ditë radhazi > {cfg.max_consecutive_days} (kufizimi 5)"
    return None


def successors(cfg: WorkforceConfig, schedule) -> list[int]:
    """Punëtorët që mund të caktohen në vendin e radhës (pa thyerje simetrie)."""
    slot = next_slot(cfg, schedule)
    if slot is None:
        return []
    day, shift = slot
    return [w for w in range(cfg.num_workers)
            if check_assignment(cfg, schedule, day, shift, w) is None]


def is_goal(cfg: WorkforceConfig, schedule) -> bool:
    return verify_schedule(to_display(schedule), cfg)["valid"]


def to_display(schedule) -> list[list[list[int]]]:
    """0-indeks → 1-indeks (P1..PW) për paraqitje dhe validim."""
    return [[sorted(w + 1 for w in workers) for workers in day] for day in schedule]


def workloads(schedule_display, num_workers: int) -> list[int]:
    loads = [0] * num_workers
    for day in schedule_display:
        for workers in day:
            for w in workers:
                if 1 <= w <= num_workers:
                    loads[w - 1] += 1
    return loads


def imbalance(loads: list[int]) -> int:
    return max(loads) - min(loads) if loads else 0


# ---------------------------------------------------------------------------
# Validimi i pavarur
# ---------------------------------------------------------------------------

def verify_schedule(schedule, cfg: WorkforceConfig) -> dict:
    """
    Verifikon një orar (punëtorët 1..W) pa përdorur strukturat e kërkimit.
    Kthen {"valid": bool, "checks": [...], "loads": [...], "imbalance": int}.
    """
    W, D, S, R = cfg.num_workers, cfg.num_days, cfg.num_shifts, cfg.workers_per_shift
    problems = {k: [] for k in range(1, 6)}

    if len(schedule) != D or any(len(day) != S for day in schedule):
        problems[1].append(f"Orari duhet të ketë {D} ditë me nga {S} turne.")
    else:
        for d, day in enumerate(schedule):
            for s, workers in enumerate(day):
                name = f"Dita {d + 1}, {cfg.shifts[s][1]}"
                if len(workers) != R:
                    problems[1].append(f"{name}: {len(workers)} punëtorë në vend të {R}.")
                if len(set(workers)) != len(workers):
                    problems[1].append(f"{name}: punëtor i përsëritur.")
                if any(not 1 <= w <= W for w in workers):
                    problems[1].append(f"{name}: punëtor i pavlefshëm.")
        # harta punëtor → turni i secilës ditë
        for w in range(1, W + 1):
            day_shift = []
            for d, day in enumerate(schedule):
                worked = [s for s, workers in enumerate(day) if w in workers]
                if len(worked) > 1:
                    problems[2].append(f"P{w}, dita {d + 1}: {len(worked)} turne në të njëjtën ditë.")
                day_shift.append(worked[0] if worked else None)
            for d in range(1, D):
                prev, cur = day_shift[d - 1], day_shift[d]
                if prev is None or cur is None:
                    continue
                if cfg.shifts[prev][0] == NIGHT and cfg.shifts[cur][0] == MORNING:
                    problems[4].append(f"P{w}: Natë (dita {d}) → Mëngjes (dita {d + 1}).")
                rest = cfg.shifts[cur][2] + 24 - cfg.shifts[prev][3]
                if rest < cfg.min_rest_hours:
                    problems[3].append(f"P{w}, ditët {d}–{d + 1}: pushim {rest} orë < "
                                       f"{cfg.min_rest_hours}.")
            run = 0
            for d in range(D):
                run = run + 1 if day_shift[d] is not None else 0
                if run == cfg.max_consecutive_days + 1:
                    problems[5].append(f"P{w}: më shumë se {cfg.max_consecutive_days} ditë radhazi "
                                       f"(deri në ditën {d + 1}).")

    loads = workloads(schedule, W)
    imb = imbalance(loads)
    names = {
        1: "1. Numri i saktë i punëtorëve për turn",
        2: "2. Asnjë punëtor me dy turne në ditë",
        3: f"3. Pushim minimal {cfg.min_rest_hours} orë ndërmjet turneve",
        4: "4. Ndalohet Natë → Mëngjes",
        5: f"5. Maksimumi {cfg.max_consecutive_days} ditë pune radhazi",
    }
    checks = [{"constraint": names[k], "ok": not problems[k], "details": problems[k],
               "hard": True} for k in range(1, 6)]
    checks.append({
        "constraint": "6. Shpërndarja e drejtë e ngarkesës (kufizim i butë)",
        "ok": imb <= cfg.imbalance_lower_bound,
        "details": [f"Imbalance = {imb} (kufiri i poshtëm teorik = {cfg.imbalance_lower_bound})."],
        "hard": False,
    })
    return {
        "valid": all(c["ok"] for c in checks if c["hard"]),
        "checks": checks,
        "loads": loads,
        "imbalance": imb,
    }


# ---------------------------------------------------------------------------
# DFS me Backtracking (Branch and Bound)
# ---------------------------------------------------------------------------

def _max_more_days(current_run: int, remaining_days: int, max_consecutive: int) -> int:
    """
    Sa ditë të tjera mund të punojë një punëtor në `remaining_days` ditët e
    mbetura, nëse ka punuar `current_run` ditë radhazi deri tani.
    Strategjia "puno sa herë lejohet" jep maksimumin (relaksim i vlefshëm).
    """
    days, run = 0, current_run
    for _ in range(remaining_days):
        if run < max_consecutive:
            days += 1
            run += 1
        else:
            run = 0
    return days

@dataclass
class WorkforceResult:
    config: WorkforceConfig
    schedule: list[list[list[int]]] | None        # punëtorët 1..W
    found: bool
    optimal_proven: bool
    imbalance: int | None
    loads: list[int]
    solutions_found: int
    first_solution_explored: int | None
    message: str
    stats: SearchStats
    verification: dict | None
    tree: SearchTreeRecorder | None = None
    landscape: dict = field(default_factory=dict)


def solve_workforce_schedule(
    num_workers: int = 6,
    num_days: int = 7,
    num_shifts: int = 3,
    workers_per_shift: int = 2,
    max_consecutive_days: int = 7,
    min_rest_hours: int = 8,
    max_nodes: int | None = DEFAULT_MAX_NODES,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    optimize: bool = True,
    tree_nodes: int = DEFAULT_TREE_NODES,
) -> WorkforceResult:
    """
    optimize=True:  vazhdon pas zgjidhjes së parë për imbalance më të vogël.
    optimize=False: ndalet te zgjidhja e parë e vlefshme (pa pretenduar optimalitet).
    """
    cfg = WorkforceConfig(num_workers, num_days, num_shifts, workers_per_shift,
                          max_consecutive_days, min_rest_hours)
    errors = validate_config(cfg) + validate_search_limits(max_nodes, timeout_seconds)
    errors += validate_non_negative_int(tree_nodes, "Numri i nyjeve të pemës")
    if errors:
        raise WorkforceConfigError(errors)

    W, D, S, R = num_workers, num_days, num_shifts, workers_per_shift
    total = cfg.total_assignments
    avg_floor, avg_ceil = total // W, -(-total // W)
    lower_bound = cfg.imbalance_lower_bound

    schedule = initial_state(cfg)
    day_shift = [[-1] * D for _ in range(W)]       # day_shift[w][d] = turni ose −1
    loads = [0] * W
    stats = SearchStats(max_nodes=max_nodes, timeout_seconds=timeout_seconds)
    tree = SearchTreeRecorder(tree_nodes)
    root_id = tree.add(None, "Orar bosh", "explored", 0)

    best = {"schedule": None, "imbalance": None, "path_ids": [], "first_at": None, "count": 0}
    path_ids: list[int | None] = []

    class _OptimumReached(Exception):
        pass

    def consecutive_run(w: int, d: int) -> int:
        run, k = 1, d - 1
        while k >= 0 and day_shift[w][k] != -1:
            run += 1
            k -= 1
        return run

    def bound_ok(d: int) -> bool:
        """Krasitja e Branch and Bound mbi imbalance-in e mundshëm."""
        if best["imbalance"] is None:
            return True
        remaining_days = D - 1 - d
        max_possible_min = min(
            avg_floor,
            min(loads[w] + remaining_days + (1 if day_shift[w][d] == -1 else 0) for w in range(W)),
        )
        lower = max(max(loads), avg_ceil) - max_possible_min
        return max(lower, lower_bound) < best["imbalance"]

    def dfs(index: int, parent_id: int | None) -> None:
        if index == total:                         # Goal: të gjitha vendet u plotësuan
            imb = max(loads) - min(loads)
            stats.valid += 1
            best["count"] += 1
            if best["first_at"] is None:
                best["first_at"] = stats.explored
            if best["imbalance"] is None or imb < best["imbalance"]:
                best["schedule"] = [[list(ws) for ws in day] for day in schedule]
                best["imbalance"] = imb
                best["path_ids"] = list(path_ids)
            if not optimize or imb <= lower_bound:
                raise _OptimumReached
            return

        d, rest = divmod(index, S * R)
        s, k = divmod(rest, R)
        shift_workers = schedule[d][s]
        lowest = shift_workers[-1] + 1 if k > 0 else 0       # thyerja e simetrisë

        # Kontrolli i kapacitetit në fillim të ditës: a mund të mbulohen ende
        # vendet e mbetura duke respektuar ditët maksimale radhazi?
        if rest == 0 and d > 0:
            capacity = sum(_max_more_days(consecutive_run(w, d) - 1, D - d,
                                          max_consecutive_days) for w in range(W))
            if capacity < total - index:
                stats.rejected += 1
                return

        # Forward checking: mjaftojnë punëtorët e lirë për vendet e mbetura të ditës?
        free_today = sum(1 for w in range(W) if day_shift[w][d] == -1)
        if free_today < S * R - rest:
            return

        candidates = sorted(range(lowest, W), key=lambda w: (loads[w], w))
        for w in candidates:
            stats.generated += 1
            reason = None
            if day_shift[w][d] != -1:
                reason = "dy turne në ditë"
            elif d > 0 and day_shift[w][d - 1] != -1 and cfg.incompatible(day_shift[w][d - 1], s):
                reason = "pushim i pamjaftueshëm"
            elif consecutive_run(w, d) > max_consecutive_days:
                reason = "shumë ditë radhazi"
            if reason is not None:
                stats.rejected += 1
                tree.add(parent_id, f"D{d + 1}{SHIFT_ABBR[cfg.shifts[s][0]]}:P{w + 1}",
                         "invalid", index + 1)
                continue

            # caktimi
            shift_workers.append(w)
            day_shift[w][d] = s
            loads[w] += 1
            if not bound_ok(d):
                stats.rejected += 1
                shift_workers.pop()
                day_shift[w][d] = -1
                loads[w] -= 1
                continue
            node_id = tree.add(parent_id, f"D{d + 1}{SHIFT_ABBR[cfg.shifts[s][0]]}:P{w + 1}",
                               "explored", index + 1)
            path_ids.append(node_id)
            stats.record_explored(index + 1)
            dfs(index + 1, node_id)
            # Backtracking: zhbëjmë caktimin
            path_ids.pop()
            shift_workers.pop()
            day_shift[w][d] = -1
            loads[w] -= 1
            stats.backtracks += 1
            # dega u braktis (rruga e zgjidhjes më të mirë rishënohet në fund)
            tree.set_status(node_id, "backtrack")

    stats.start()
    reason = TERMINATION_EXHAUSTED
    try:
        dfs(0, root_id)
    except _OptimumReached:
        reason = TERMINATION_GOAL_REACHED
    except SearchLimitReached as limit:
        reason = limit.reason
    stats.stop(reason)

    if best["schedule"] is None:
        if reason == TERMINATION_EXHAUSTED:
            message = ("Kërkimi i plotë nuk gjeti orar të vlefshëm: konfigurimi NUK ka "
                       "zgjidhje me këto kufizime.")
        else:
            message = (f"{stats.termination_message} Nuk u gjet orar i vlefshëm brenda kufijve; "
                       f"kjo NUK provon që nuk ekziston.")
        return WorkforceResult(cfg, None, False, False, None, [0] * W, 0, None, message,
                               stats, None, tree, landscape(cfg))

    display = to_display(best["schedule"])
    verification = verify_schedule(display, cfg)
    imb = best["imbalance"]
    tree.mark_path(best["path_ids"][:-1], best["path_ids"][-1] if best["path_ids"] else None)

    if imb <= lower_bound:
        optimal = True
        message = (f"U gjet orar i vlefshëm me imbalance {imb}, i barabartë me kufirin e poshtëm "
                   f"teorik ({lower_bound}): objektivi është optimizuar plotësisht.")
    elif reason == TERMINATION_EXHAUSTED and optimize:
        optimal = True
        message = (f"Kërkimi i plotë (Branch and Bound) përfundoi: imbalance minimal = {imb}. "
                   f"Optimaliteti është vërtetuar.")
    elif not optimize:
        optimal = False
        message = (f"U gjet zgjidhja e parë e vlefshme (imbalance {imb}). Optimizimi ishte i "
                   f"çaktivizuar, prandaj optimaliteti NUK është vërtetuar.")
    else:
        optimal = False
        message = (f"{stats.termination_message} Orari më i mirë i gjetur ka imbalance {imb}; "
                   f"optimaliteti NUK është vërtetuar (kufiri i poshtëm = {lower_bound}).")

    return WorkforceResult(
        config=cfg, schedule=display, found=verification["valid"], optimal_proven=optimal,
        imbalance=imb, loads=verification["loads"], solutions_found=best["count"],
        first_solution_explored=best["first_at"], message=message, stats=stats,
        verification=verification, tree=tree, landscape=landscape(cfg),
    )


# ---------------------------------------------------------------------------
# Përgatitja për ndërfaqen
# ---------------------------------------------------------------------------

def schedule_to_rows(schedule, cfg: WorkforceConfig) -> list[dict]:
    rows = []
    for d, day in enumerate(schedule):
        row = {"Dita": f"Dita {d + 1}"}
        for s, workers in enumerate(day):
            row[cfg.shifts[s][1]] = ", ".join(f"P{w}" for w in workers)
        rows.append(row)
    return rows


def workload_rows(schedule, cfg: WorkforceConfig) -> list[dict]:
    rows = []
    for w in range(1, cfg.num_workers + 1):
        row = {"Punëtori": f"P{w}"}
        total = 0
        for s, shift in enumerate(cfg.shifts):
            count = sum(1 for day in schedule if w in day[s])
            row[shift[1]] = count
            total += count
        row["Gjithsej"] = total
        rows.append(row)
    return rows
