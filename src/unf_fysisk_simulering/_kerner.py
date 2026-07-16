"""
_kerner.py
==========

Hurtige "kerner" til de tunge simuleringer (tilfældig gang, TASEP, Ising,
skovbrand). De er skrevet, så de kan oversættes (kompileres) af *numba* og
køre meget hurtigere end almindelig Python.

Du behøver **ikke** at læse eller forstå denne fil for at lave opgaverne!
Den ligger her, så `simulering.py` kan importere de hurtige funktioner.

Numba er en *valgfri* afhængighed. Installér den med::

    pip install "unf-fysisk-simulering[hurtig]"

To detaljer som er rare at kende:

1. Hvis numba ikke er installeret (eller hvis miljøvariablen
   ``SIMULERING_INGEN_NUMBA`` er sat), bruges en "tom" erstatning for
   ``@njit``, og kernerne kører som ren Python. Så virker alt stadig –
   bare langsommere.

2. Når numba er aktiv, gemmer hver kerne den oprindelige Python-version i
   ``kerne.py_func``. `simulering.py` bruger den til at give pæne fejl-
   meddelelser, hvis numba-versionen skulle fejle.
"""

import os
import numpy as np

# ------------------------------------------------------------------------
# Forsøg at hente numbas @njit. Falder tilbage til en "tom" dekorator,
# så koden også virker uden numba.
# ------------------------------------------------------------------------
if os.environ.get("SIMULERING_INGEN_NUMBA"):
    raise_import = True
else:
    raise_import = False

try:
    if raise_import:
        raise ImportError("SIMULERING_INGEN_NUMBA er sat")
    from numba import njit
    HAR_NUMBA = True
except Exception:
    HAR_NUMBA = False

    def njit(*args, **kwargs):
        """Tom erstatning for numbas @njit, når numba ikke er tilgængelig."""
        # Brugt som @njit (uden parentes)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        # Brugt som @njit(...) (med argumenter)
        def dekorator(funktion):
            return funktion
        return dekorator


# ========================================================================
# 1) Tilfældig gang på et gitter (1D, 2D eller 3D)
# ========================================================================
@njit(cache=False)
def kerne_tilfaeldig_gang(antal_skridt, antal_baner, dim, seed):
    np.random.seed(seed)
    baner = np.zeros((antal_baner, antal_skridt + 1, dim))
    for b in range(antal_baner):
        for s in range(1, antal_skridt + 1):
            # kopiér forrige position
            for d in range(dim):
                baner[b, s, d] = baner[b, s - 1, d]
            # gå ét skridt langs en tilfældig akse, i en tilfældig retning
            akse = np.random.randint(0, dim)
            if np.random.random() < 0.5:
                baner[b, s, akse] += 1.0
            else:
                baner[b, s, akse] -= 1.0
    return baner


# ========================================================================
# 2) TASEP (trafik på en ensrettet vej med ét spor)
#    p_ind, p_hop, p_ud er allerede ganget med dt (altså sandsynligheder).
#    Opdatering fra højre mod venstre, så ingen bil flyttes to gange.
# ========================================================================
@njit(cache=False)
def kerne_tasep(L, p_ind, p_hop, p_ud, antal_skridt, seed):
    np.random.seed(seed)
    vej = np.zeros(L, dtype=np.int8)
    historik = np.zeros((antal_skridt + 1, L), dtype=np.int8)
    for n in range(1, antal_skridt + 1):
        # udkørsel ved sidste felt
        if vej[L - 1] == 1:
            if np.random.random() < p_ud:
                vej[L - 1] = 0
        # fremryk: gennemløb bindingerne fra højre mod venstre
        for i in range(L - 2, -1, -1):
            if vej[i] == 1 and vej[i + 1] == 0:
                if np.random.random() < p_hop:
                    vej[i] = 0
                    vej[i + 1] = 1
        # indkørsel ved første felt
        if vej[0] == 0:
            if np.random.random() < p_ind:
                vej[0] = 1
        # gem hele vejen til rum-tid-diagrammet
        for i in range(L):
            historik[n, i] = vej[i]
    return historik


# ========================================================================
# 3) 2D Ising-model (magnetisme) med Metropolis-Monte-Carlo
# ========================================================================
@njit(cache=False)
def kerne_ising(L, T, antal_fejecyklusser, seed):
    np.random.seed(seed)
    # tilfældig start: hver spin er +1 eller -1
    gitter = np.zeros((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                gitter[i, j] = 1
            else:
                gitter[i, j] = -1

    magnetisering = np.zeros(antal_fejecyklusser)
    for cyklus in range(antal_fejecyklusser):
        # én fejecyklus = L*L forsøg på at vende en spin
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            s = gitter[i, j]
            nabosum = (gitter[(i + 1) % L, j] + gitter[(i - 1) % L, j]
                       + gitter[i, (j + 1) % L] + gitter[i, (j - 1) % L])
            dE = 2.0 * s * nabosum
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                gitter[i, j] = -s
        # gennemsnitlig magnetisering pr. spin
        sum_spin = 0.0
        for i in range(L):
            for j in range(L):
                sum_spin += gitter[i, j]
        magnetisering[cyklus] = sum_spin / (L * L)
    return gitter, magnetisering


# ========================================================================
# 4) Skovbrand-celleautomat (bonus)
#    0 = tom, 1 = træ, 2 = brændende
# ========================================================================
@njit(cache=False)
def kerne_skovbrand(L, p_vaekst, p_lyn, antal_skridt, seed):
    np.random.seed(seed)
    gitter = np.zeros((L, L), dtype=np.int8)
    # start med nogle træer
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                gitter[i, j] = 1

    historik = np.zeros((antal_skridt + 1, L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            historik[0, i, j] = gitter[i, j]

    for n in range(1, antal_skridt + 1):
        ny = np.zeros((L, L), dtype=np.int8)
        for i in range(L):
            for j in range(L):
                tilstand = gitter[i, j]
                if tilstand == 2:
                    ny[i, j] = 0  # brændende -> tom
                elif tilstand == 1:
                    # brænder en nabo?
                    nabo_braender = False
                    if gitter[(i + 1) % L, j] == 2:
                        nabo_braender = True
                    if gitter[(i - 1) % L, j] == 2:
                        nabo_braender = True
                    if gitter[i, (j + 1) % L] == 2:
                        nabo_braender = True
                    if gitter[i, (j - 1) % L] == 2:
                        nabo_braender = True
                    if nabo_braender:
                        ny[i, j] = 2
                    elif np.random.random() < p_lyn:
                        ny[i, j] = 2  # lynnedslag
                    else:
                        ny[i, j] = 1
                else:
                    # tom -> måske vokser et træ
                    if np.random.random() < p_vaekst:
                        ny[i, j] = 1
                    else:
                        ny[i, j] = 0
        gitter = ny
        for i in range(L):
            for j in range(L):
                historik[n, i, j] = gitter[i, j]
    return historik
