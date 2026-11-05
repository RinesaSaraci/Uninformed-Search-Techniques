"""
Regjistrimi i pemës reale të kërkimit dhe konvertimi në graf NetworkX.

Algoritmet thërrasin `SearchTreeRecorder.add(...)` në momentin kur gjenerojnë
një gjendje, pra pema përmban vetëm gjendje që algoritmi i ka krijuar
realisht. Kur arrihet `max_nodes`, nyjet e reja nuk regjistrohen më
(`truncated = True`). Meqë një nyje regjistrohet vetëm nëse prindi i saj
është regjistruar, nën-pema e ruajtur mbetet gjithmonë e lidhur dhe
marrëdhëniet prind–fëmijë janë ato reale.

Statuset e nyjeve:
    generated – gjendje e gjeneruar (ende në radhë/stekë ose e pazgjeruar)
    explored  – gjendje e zgjeruar
    duplicate – gjendje e refuzuar sepse është vizituar më parë
    invalid   – gjendje e refuzuar sepse shkel kufizimet
    goal      – gjendja e qëllimit
    solution  – gjendje në rrugën e zgjidhjes
    backtrack – degë e braktisur (Backtracking)
"""

from __future__ import annotations

import networkx as nx

STATUS_LABELS = {
    "generated": "E gjeneruar",
    "explored": "E eksploruar",
    "duplicate": "E refuzuar (e vizituar)",
    "invalid": "E refuzuar (e pavlefshme)",
    "goal": "Qëllimi",
    "solution": "Rruga e zgjidhjes",
    "backtrack": "Degë e braktisur",
}


class SearchTreeRecorder:
    """Ruan një nën-bashkësi të kufizuar të pemës reale të kërkimit."""

    def __init__(self, max_nodes: int = 0):
        self.max_nodes = max(0, int(max_nodes))
        self.nodes: dict[int, dict] = {}
        self.truncated = False
        self._next_id = 0

    @property
    def enabled(self) -> bool:
        return self.max_nodes > 0

    def add(self, parent_id: int | None, label: str, status: str,
            depth: int, action: str | None = None) -> int | None:
        """
        Regjistron një nyje. Kthen id-në e saj, ose None nëse nuk u regjistrua
        (kufiri u arrit ose prindi nuk është në pemë).
        """
        if not self.enabled:
            return None
        if parent_id is not None and parent_id not in self.nodes:
            return None
        if len(self.nodes) >= self.max_nodes:
            self.truncated = True
            return None
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = {
            "id": node_id,
            "parent": parent_id,
            "label": label,
            "status": status,
            "depth": depth,
            "action": action,
        }
        return node_id

    def set_status(self, node_id: int | None, status: str) -> None:
        if node_id is not None and node_id in self.nodes:
            self.nodes[node_id]["status"] = status

    def mark_path(self, node_ids, goal_id: int | None = None) -> None:
        for node_id in node_ids:
            self.set_status(node_id, "solution")
        self.set_status(goal_id, "goal")

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node_id, data in self.nodes.items():
            graph.add_node(node_id, **{k: v for k, v in data.items() if k != "id"})
        for node_id, data in self.nodes.items():
            if data["parent"] is not None:
                graph.add_edge(data["parent"], node_id, action=data["action"])
        return graph

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for data in self.nodes.values():
            counts[data["status"]] = counts.get(data["status"], 0) + 1
        return {
            "recorded_nodes": len(self.nodes),
            "max_nodes": self.max_nodes,
            "truncated": self.truncated,
            "status_counts": counts,
        }


def tree_layout(graph: nx.DiGraph) -> dict:
    """
    Pozicione hierarkike pa varësi të jashtme (pa Graphviz):
    y = −thellësia, x = renditja e gjetheve sipas DFS.
    """
    positions: dict[int, tuple[float, float]] = {}
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    next_x = [0.0]

    def place(node: int, depth: int) -> float:
        children = sorted(graph.successors(node))
        if not children:
            x = next_x[0]
            next_x[0] += 1.0
        else:
            xs = [place(child, depth + 1) for child in children]
            x = (xs[0] + xs[-1]) / 2
        positions[node] = (x, -depth)
        return x

    for root in sorted(roots):
        place(root, 0)
    return positions
