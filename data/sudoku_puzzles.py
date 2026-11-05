"""
Shembuj Sudoku për Detyrën 2.

Çdo enigmë ruhet si varg me 81 shifra (rreshtat njëri pas tjetrit),
ku 0 = qelizë bosh. Për secilën ruhet edhe zgjidhja.

Verifikimi: çdo enigmë është kontrolluar me një zgjidhës të pavarur
(numërues zgjidhjesh) që ka saktësisht një zgjidhje, dhe testet në
`tests/test_sudoku.py` e rikontrollojnë këtë në çdo ekzekutim:
  - enigma është e vlefshme (pa përsëritje),
  - zgjidhja është e plotë dhe e vlefshme,
  - zgjidhja ruan të gjitha shifrat fillestare.
"""

PUZZLES = {
    "easy": {
        "label": "E lehtë",
        "description": "Shembulli klasik i Sudoku-së (30 shifra fillestare).",
        "puzzle": (
            "530070000"
            "600195000"
            "098000060"
            "800060003"
            "400803001"
            "700020006"
            "060000280"
            "000419005"
            "000080079"
        ),
        "solution": (
            "534678912"
            "672195348"
            "198342567"
            "859761423"
            "426853791"
            "713924856"
            "961537284"
            "287419635"
            "345286179"
        ),
    },
    "medium": {
        "label": "Mesatare",
        "description": "Enigmë me 30 shifra fillestare dhe më shumë degëzime.",
        "puzzle": (
            "200080300"
            "060070084"
            "030500209"
            "000105408"
            "000000000"
            "402706000"
            "301007040"
            "720040060"
            "004010003"
        ),
        "solution": (
            "245981376"
            "169273584"
            "837564219"
            "976125438"
            "513498627"
            "482736951"
            "391657842"
            "728349165"
            "654812793"
        ),
    },
    "hard": {
        "label": "E vështirë",
        "description": "Enigmë shumë e vështirë me vetëm 21 shifra fillestare.",
        "puzzle": (
            "800000000"
            "003600000"
            "070090200"
            "050007000"
            "000045700"
            "000100030"
            "001000068"
            "008500010"
            "090000400"
        ),
        "solution": (
            "812753649"
            "943682175"
            "675491283"
            "154237896"
            "369845721"
            "287169534"
            "521974368"
            "438526917"
            "796318452"
        ),
    },
}

DIFFICULTIES = tuple(PUZZLES.keys())
