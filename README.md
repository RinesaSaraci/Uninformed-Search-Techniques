# AI Search Algorithms – Python & Streamlit

An educational project that implements four university assignments on
**uninformed search techniques** (DFS, BFS, IDDFS and Backtracking), with an
interactive Streamlit interface (in Albanian), real execution statistics and
search-tree visualizations.

## Assignments

| # | Problem | Algorithm |
|---|---------|-----------|
| 1 | Social Golfers Problem (32 golfers, 8 groups of 4) | DFS with Backtracking |
| 2 | Sudoku Solver (Easy, Medium, Hard) | BFS with Backtracking |
| 3 | Latin Square (n×n, default n = 5) | IDDFS with Backtracking |
| 4a | 8-Puzzle | BFS |
| 4b | Missionaries and Cannibals | BFS |
| 4c | Rotating Workforce Scheduling | DFS with Backtracking |

For each assignment the app shows the problem description, model (variables,
domain, constraints), objective function, landscape calculation, execution
results, visualizations, statistics and an independent validation.

## Requirements

- Python 3.12
- Streamlit, Matplotlib, NetworkX, Pandas, Pytest (see `requirements.txt`)

## Installation (Windows)

Open the folder in Visual Studio Code, open a terminal and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In Command Prompt (cmd) use `.venv\Scripts\activate.bat` instead.

## Running the application

```powershell
streamlit run app.py
```

The app opens in the browser. Use the sidebar to switch between assignments.
Algorithms only run when you press a button.

## Running the tests

```powershell
python -m pytest -v
```

## Project structure

```
ai_search_project/
├── app.py                 # Streamlit entry point and navigation
├── tasks/                 # Algorithms (independent of Streamlit)
│   ├── social_golfers.py
│   ├── sudoku.py
│   ├── latin_square.py
│   └── search_trees/
│       ├── eight_puzzle.py
│       ├── missionaries_cannibals.py
│       └── workforce_scheduling.py
├── ui/                    # One Streamlit page per assignment + shared components
├── utils/                 # Validation, statistics, visualization, search-tree recorder
├── data/                  # Verified Sudoku puzzles with their solutions
├── tests/                 # Pytest test suite
├── pytest.ini
└── requirements.txt
```

Every algorithm can also be run from the terminal, for example:

```powershell
python -m tasks.sudoku
python -m tasks.latin_square
python -m tasks.social_golfers
```

## How the algorithms work (short)

- **Social Golfers – DFS + Backtracking:** builds the schedule group by group.
  A group that repeats a pair is rejected; when no group fits, the last group is
  undone. Symmetry breaking (fixed first week, groups built around a pivot golfer)
  and forward checking reduce the search. The theoretical upper bound
  ⌊C(n,2) / (g·C(p,2))⌋ = 10 weeks is shown separately from the schedule found.
- **Sudoku – BFS + Backtracking:** a FIFO queue (`deque`) of partial boards.
  Each step fills one empty cell (chosen with MRV); invalid values are rejected and
  dead-end branches are abandoned.
- **Latin Square – IDDFS + Backtracking:** depth-limited DFS with limits
  0, 1, 2, …, n², restarting from the empty matrix each time. Depth = number of
  assigned cells. The answer always comes from the search, never from a formula.
- **8-Puzzle – BFS:** FIFO queue, visited set and parent links; returns the
  shortest path. Unsolvable states are detected first with the inversion-parity rule.
- **Missionaries and Cannibals – BFS:** state `(M_left, C_left, boat)`;
  both riverbanks are checked after every crossing.
- **Workforce Scheduling – DFS + Backtracking:** fills shifts slot by slot,
  respecting staffing, one shift per day, minimum rest (Night → Morning is never
  allowed) and maximum consecutive working days. After the first valid schedule
  it keeps searching (Branch and Bound) to minimize workload imbalance.

All statistics (generated, explored, rejected states, time, etc.) are counted
during the real execution. If a search hits its node limit or timeout, the app
says so clearly and never presents it as proof that no solution exists.

## Known limitations

- **Social Golfers 32-8-4:** DFS reliably finds **7 valid weeks** (seed 1), but not
  the theoretical maximum of 10, even after millions of nodes. Optimality is
  therefore reported as *not proven*. Smaller cases (e.g. 9-3-3) reach the bound,
  and 12-4-3 is proven optimal (4 weeks) by exhaustive search.
- **Sudoku:** BFS cannot fill an (almost) empty board within the limits because
  the queue grows too fast. Without MRV the Hard puzzle is interrupted by the limits.
- **Latin Square:** n ≤ 6 is solved (n = 6 takes about 2–3 s); n = 7 normally
  reaches the limits. Without symmetry breaking, n ≥ 5 reaches the limits.
- **8-Puzzle:** the hardest states (31 moves) take about 1.5 s.
- **Workforce Scheduling:** a simplified educational model. With the default
  6 workers and 3 shifts × 2, every worker must work all 7 days, so a maximum of
  fewer than 7 consecutive days is impossible; the app detects this before searching.
- Large search trees are shown partially (the first N generated states, with their
  real parent–child links).
