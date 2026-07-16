# functions_moedekjaer
It is a function collection written primarylly by Mikkel Møller Mødekjær

This repo ships **two importable packages** in one `modekjar` distribution:

| Package | What |
|---|---|
| `modekjar` | The original function collection — error propagation, χ² fitting, Fisher/ROC, plot helpers |
| `unf_fysisk_simulering` | Simulation library for **UNF Fysik Camp**, topic *"Fysisk Simulering"* (Danish) |

```bash
pip install modekjar            # both packages
pip install "modekjar[hurtig]"  # same, plus numba for the fast simulation kernels
```

```python
import modekjar
from unf_fysisk_simulering import *
```

The `unf_fysisk_simulering` code is a copy of the standalone
[unf-fysisk-simulering](https://github.com/UNF-Science-Camps/FysikCamp2026) package, kept
here alongside the original functions. Its `__version__` tracks the upstream release it was
copied from and is deliberately independent of `modekjar`'s own version.

---

# unf_fysisk_simulering

Simulerings-bibliotek til **UNF Fysik Camp**, emnet *"Fysisk Simulering"*.

Idéen er, at du kun skal beskrive **systemet** — en differentialligning, nogle kemiske
reaktioner, en hastighedsmodel — som en lille funktion. Så klarer biblioteket selve
simuleringen og tegner pæne grafer. Alt er på dansk, og hver eneste funktion har en
docstring, der forklarer både hvad den gør, og hvad fysikken bag er.

## Installation

```bash
pip install modekjar
```

I Colab (øverst i notebooken):

```python
!pip install -q modekjar
```

De tunge kerner (tilfældig gang, TASEP, Ising, skovbrand) kører meget hurtigere med
[numba](https://numba.pydata.org/). Er numba ikke installeret, virker **alt stadig** —
bare langsommere:

```bash
pip install "modekjar[hurtig]"     # med numba
```

Variablen `HAR_NUMBA` fortæller dig, hvilken af delene du kører med. Colab har numba med
i forvejen, så der behøver du ikke `[hurtig]`.

## Kom i gang

```python
import numpy as np
import matplotlib.pyplot as plt
from unf_fysisk_simulering import *

def logistisk(t, x):
    return 0.8 * x * (1 - x / 100)      # dx/dt = r·x·(1 - x/K)

t, x = simuler_differential(logistisk, start=1.0, t_slut=20.0, dt=0.1)
plot_baner(t, x, titel="Logistisk vækst")
plt.show()
```

## Hvad er der i?

### Simulering

| Funktion | Hvad |
|---|---|
| `simuler_differential(f, start, t_slut, dt, metode)` | Differentialligninger — Eulers metode, RK4, RK45 |
| `simuler_maruyama(drift, stoej, start, t_slut, dt, ...)` | Stokastiske ligninger (Euler-Maruyama) |
| `simuler_brownsk(D, start, t_slut, dt, ...)` | Brownsk bevægelse |
| `simuler_tilfaeldig_gang(antal_skridt, antal_baner, dim)` | Tilfældig gang i 1D/2D/3D |
| `simuler_monte_carlo(forsoeg, antal)` | Kør et tilfældigt forsøg mange gange |
| `simuler_poisson(rate, t_slut, dt, metode)` | Poisson-proces, tids- eller hændelses-drevet |
| `simuler_gillespie(start_tilstand, reaktioner, t_slut)` | Gillespie-algoritmen |
| `simuler_tasep(L, alpha, beta, t_slut, dt)` | TASEP — trafik, ribosomer |
| `simuler_trafik(hastighedsmodel, L, antal_biler, ...)` | Trafik på en ringvej med **din egen** hastighedsmodel |
| `simuler_ising(L, T, antal_fejecyklusser)` | 2D Ising-model (magnetisme) |
| `simuler_skovbrand(L, p_vaekst, p_lyn, antal_skridt)` | Skovbrand-celleautomat |

### Biologi

| Funktion | Hvad |
|---|---|
| `simuler_epidemi_gitter(L, p_smitte, p_rask, ...)` | Epidemi på et gitter (rumlig SIR) |
| `simuler_epidemi_agenter(antal_personer, fart, radius, ...)` | Epidemi blandt personer, der bevæger sig |
| `simuler_wright_fisher(antal_individer, start_frekvens, s, ...)` | Genetisk drift |
| `simuler_neuron(stroem, stoej, ...)` | Neuron, der fyrer (integrate-and-fire) |
| `simuler_reaktion_diffusion(F, k, ...)` | Turing-mønstre — pletter og striber |
| `simuler_flok(antal_fugle, stoej, ...)` | Fugleflok uden leder (Vicsek) |

### Grafer

| Funktion | Hvad |
|---|---|
| `plot_baner(t, x, ...)` | Kurver og baner |
| `plot_histogram(data, ...)` | En fordeling |
| `plot_rumtid(historik, ...)` | Rum-tid-diagram (trafikpropper) |
| `plot_faseplan(x, y, ...)` | To variable mod hinanden |
| `vis_gitter(gitter, ...)` | Øjebliksbillede af et gitter |
| `animer_gitter(historik, ...)` | Animation af et gitter |
| `animer_partikler(positioner, ...)` | Animation af prikker, der bevæger sig |

De fleste `plot_*`-funktioner tager gerne hele resultatet: `plot_rumtid(res)`,
`animer_partikler(res)`.

Slå en enkelt funktion op med `help(simuler_gillespie)`.

## Kursusmaterialet

Selve emnet — to Colab-notebooks med 51 opgaver, opgaveteksterne og de forklarende
diagrammer — ligger i kursus-repoet
[UNF-Science-Camps/FysikCamp2026](https://github.com/UNF-Science-Camps/FysikCamp2026).

---

## Udvikling

```
setup.py                      # metadata + afhængigheder for HELE distributionen
pyproject.toml                # kun byggesystemet
src/modekjar/                 # den oprindelige funktionssamling
src/unf_fysisk_simulering/    # simulerings-pakken
    simulering.py             #   hoved-API
    _kerner.py                #   numba-kerner med ren-Python-fallback
tests/
    test_simulering.py        #   tjekker hver funktion mod det, fysikken forudsiger
.github/workflows/
    test.yml                  #   CI: begge kerne-veje, og at hjulet kan installeres
    publish.yml               #   udgiver til PyPI på en GitHub Release
```

```bash
pip install -e ".[hurtig]"                                  # installér til udvikling

python tests/test_simulering.py                             # selvtest (numba aktiv)
SIMULERING_INGEN_NUMBA=1 python tests/test_simulering.py    # samme, ren Python
```

### Udgiv en ny version

Uploaden bruger **PyPI Trusted Publishing** — der ligger ingen API-token i repoet.
Det skal sættes op én gang:

1. På PyPI → *Your projects* → *Publishing* → **Add a pending publisher**:

   | Felt | Værdi |
   |---|---|
   | PyPI Project Name | `modekjar` |
   | Owner | `Zaptos27` |
   | Repository name | `functions_moedekjaer` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

2. På GitHub → *Settings* → *Environments* → **New environment** → `pypi`.

Derefter er en udgivelse:

```bash
# 1) sæt versionen i setup.py
git commit -am "v0.0.4" && git tag v0.0.4 && git push --follow-tags
# 2) lav en Release på taggen i GitHub  ->  publish.yml kører
```

`publish.yml` nægter at uploade, hvis taggen og `setup.py` ikke er enige om
versionsnummeret. PyPI tillader nemlig ikke, at det samme versionsnummer uploades to
gange — så en fejl her kan ikke fortrydes.

## Licens

MIT — se [LICENSE](LICENSE).
