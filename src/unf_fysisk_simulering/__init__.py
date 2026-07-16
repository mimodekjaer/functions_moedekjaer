"""
unf_fysisk_simulering  —  Fysik Camp 2026, emnet "Fysisk Simulering"
====================================================================

Et lille bibliotek med færdige simulerings-funktioner. Idéen er, at *du* kun
skal beskrive **systemet** (fx en differentialligning eller nogle kemiske
reaktioner) som en lille funktion, og så klarer biblioteket selve simuleringen
og tegner pæne grafer.

Brug det sådan her::

    from unf_fysisk_simulering import *

    t, x = simuler_differential(lambda t, x: -0.5 * x, start=1.0,
                                t_slut=10.0, dt=0.1)
    plot_baner(t, x)

Alle funktioner, du skal bruge, hedder noget med ``simuler_...`` eller
``plot_...`` / ``vis_...`` / ``animer_...``. Se ``help(simuler_differential)``
for den fulde forklaring på en enkelt funktion — hver eneste af dem har en
dansk docstring.

De hurtige kerner bruger *numba*, hvis den er installeret. Er den ikke det,
kører alt stadig — bare langsommere. Variablen ``HAR_NUMBA`` fortæller dig
hvilken af delene::

    pip install "unf-fysisk-simulering[hurtig]"     # med numba
"""

from .simulering import *                       # noqa: F401,F403
from .simulering import __all__ as _OFFENTLIGT_API

__version__ = "0.1.0"

__all__ = list(_OFFENTLIGT_API) + ["__version__"]
