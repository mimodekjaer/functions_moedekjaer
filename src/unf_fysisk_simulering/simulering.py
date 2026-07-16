"""
simulering.py  —  Fysik Camp 2026, emnet "Fysisk Simulering"
=============================================================

Et lille bibliotek med færdige simulerings-funktioner. Idéen er, at *du* kun
skal beskrive **systemet** (fx en differentialligning eller nogle kemiske
reaktioner) som en lille funktion, og så klarer biblioteket selve simuleringen
og tegner pæne grafer.

Alle funktioner, du skal bruge, hedder noget med ``simuler_...`` eller
``plot_...`` / ``vis_...``:

Simulering
----------
- ``simuler_differential(f, start, t_slut, dt, metode)`` – differentialligninger
  (Eulers metode, RK4, RK45).
- ``simuler_maruyama(drift, stoej, start, t_slut, dt, antal_baner)`` – stokastiske
  ligninger (Euler-Maruyama).
- ``simuler_brownsk(D, start, t_slut, dt, antal_baner)`` – Brownsk bevægelse.
- ``simuler_tilfaeldig_gang(antal_skridt, antal_baner, dim)`` – tilfældig gang.
- ``simuler_monte_carlo(forsoeg, antal)`` – kør et tilfældigt forsøg mange gange.
- ``simuler_poisson(rate, t_slut, dt, metode)`` – Poisson-proces (tids-/hændelses-drevet).
- ``simuler_gillespie(start_tilstand, reaktioner, t_slut)`` – Gillespie-algoritmen.
- ``simuler_tasep(L, alpha, beta, t_slut, dt)`` – TASEP / trafik / ribosomer.
- ``simuler_ising(L, T, antal_fejecyklusser)`` – Ising-magnetisme.
- ``simuler_skovbrand(L, p_vaekst, p_lyn, antal_skridt)`` – skovbrand (bonus).

Biologi
-------
- ``simuler_epidemi_gitter(L, p_smitte, p_rask, p_vaccineret, ...)`` – epidemi på et
  gitter (rumlig SIR, celleautomat).
- ``simuler_epidemi_agenter(antal_personer, fart, radius, ...)`` – epidemi blandt
  personer, der bevæger sig rundt (agent-baseret SIR).
- ``simuler_wright_fisher(antal_individer, start_frekvens, s, ...)`` – genetisk drift.
- ``simuler_neuron(stroem, stoej, ...)`` – en neuron, der fyrer (integrate-and-fire).
- ``simuler_reaktion_diffusion(F, k, ...)`` – Turing-mønstre (pletter og striber).
- ``simuler_flok(antal_fugle, stoej, ...)`` – en fugleflok uden leder (Vicsek).

Grafer
------
- ``plot_baner(t, x, ...)`` – tegn kurver/baner.
- ``plot_histogram(data, ...)`` – tegn en fordeling.
- ``plot_rumtid(historik, ...)`` – rum-tid-diagram (trafik).
- ``plot_faseplan(x, y, ...)`` – tegn to variable mod hinanden (faseplan).
- ``vis_gitter(gitter, ...)`` – vis et gitter (øjebliksbillede).
- ``animer_gitter(historik, ...)`` – lav en animation af et gitter.
- ``animer_partikler(positioner, ...)`` – animér prikker, der bevæger sig (agenter, fugle).
"""

from collections import namedtuple
import warnings
import numpy as np

from ._kerner import (
    HAR_NUMBA,
    kerne_tilfaeldig_gang,
    kerne_tasep,
    kerne_ising,
    kerne_skovbrand,
)

__all__ = [
    "simuler_differential",
    "simuler_maruyama",
    "simuler_brownsk",
    "simuler_tilfaeldig_gang",
    "simuler_monte_carlo",
    "simuler_poisson",
    "simuler_gillespie",
    "simuler_tasep",
    "simuler_trafik",
    "simuler_ising",
    "simuler_skovbrand",
    "simuler_epidemi_gitter",
    "simuler_epidemi_agenter",
    "simuler_wright_fisher",
    "simuler_neuron",
    "simuler_reaktion_diffusion",
    "simuler_flok",
    "plot_baner",
    "plot_histogram",
    "plot_rumtid",
    "plot_faseplan",
    "vis_gitter",
    "animer_gitter",
    "animer_partikler",
    "HAR_NUMBA",
]


# ========================================================================
# Hjælpere (du behøver ikke kigge her)
# ========================================================================
def _koer_kerne(kerne, *args):
    """Kør en numba-kerne. Hvis den fejler, køres den rene Python-udgave,
    så man får en læsbar fejlmeddelelse i stedet for en kryptisk numba-fejl."""
    try:
        return kerne(*args)
    except Exception as fejl:  # pragma: no cover - kun ved numba-problemer
        py = getattr(kerne, "py_func", None)
        if py is None:
            raise
        warnings.warn(
            "Den hurtige (numba) udgave fejlede – kører ren Python i stedet "
            "for at vise en tydeligere fejl. Oprindelig fejl: %r" % (fejl,)
        )
        return py(*args)


def _lav_seed(seed):
    """Lav et heltals-frø til de hurtige kerner (numba bruger sit eget RNG)."""
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    return int(seed)


def _kraev_positiv(navn, vaerdi):
    if vaerdi <= 0:
        raise ValueError("%s skal være større end 0 (du gav %r)." % (navn, vaerdi))


def _kraev_positivt_heltal(navn, vaerdi):
    if not isinstance(vaerdi, (int, np.integer)) or vaerdi <= 0:
        raise ValueError("%s skal være et positivt heltal (du gav %r)." % (navn, vaerdi))


def _kraev_sandsynlighed(navn, vaerdi):
    if not (0.0 <= vaerdi <= 1.0):
        raise ValueError(
            "%s er en sandsynlighed og skal ligge mellem 0 og 1 (du gav %r)."
            % (navn, vaerdi))


def _kraev_ikke_negativ(navn, vaerdi):
    if vaerdi < 0:
        raise ValueError("%s må ikke være negativ (du gav %r)." % (navn, vaerdi))


# ========================================================================
# 1) Differentialligninger:  dx/dt = f(t, x)
# ========================================================================
def simuler_differential(f, start, t_slut, dt, metode="rk4", tolerance=1e-4):
    """Løs en differentialligning ``dx/dt = f(t, x)`` numerisk.

    Parametre
    ---------
    f : funktion ``f(t, x)`` der returnerer dx/dt (samme form som x).
    start : startværdien x(0). Et tal (fx 1.0) eller en liste/array for et
            system af ligninger (fx [S, I, R]).
    t_slut : hvor længe der simuleres.
    dt : tidsskridtet.
    metode : "euler", "rk4" eller "rk45".
    tolerance : kun for "rk45" – hvor stor fejl pr. skridt der accepteres.

    Returnerer
    ----------
    t : array med tidspunkter.
    x : array med værdier. For ét tal pr. tid har x form (antal_tider,);
        for et system har x form (antal_tider, antal_variable).
    """
    _kraev_positiv("dt", dt)
    _kraev_positiv("t_slut", t_slut)
    metode = str(metode).lower()
    if metode not in ("euler", "rk4", "rk45"):
        raise ValueError("metode skal være 'euler', 'rk4' eller 'rk45' "
                         "(du gav %r)." % metode)

    start = np.atleast_1d(np.asarray(start, dtype=float))
    d = start.size

    # Tjek brugerens funktion én gang, så fejl opdages tidligt med en pæn besked
    try:
        proeve = np.atleast_1d(np.asarray(f(0.0, start), dtype=float))
    except Exception as e:
        raise ValueError(
            "Kunne ikke kalde din funktion f(t, x). Tjek at den tager to "
            "argumenter (t og x) og returnerer dx/dt. Oprindelig fejl: %r" % (e,)
        )
    if proeve.shape != start.shape:
        raise ValueError(
            "Din funktion f(t, x) returnerede noget med form %s, men 'start' "
            "har form %s. De skal have samme form." % (proeve.shape, start.shape)
        )

    if metode in ("euler", "rk4"):
        antal = int(np.ceil(t_slut / dt))
        t = np.arange(antal + 1) * dt
        x = np.zeros((antal + 1, d))
        x[0] = start
        for n in range(antal):
            tn, xn = t[n], x[n]
            if metode == "euler":
                x[n + 1] = xn + np.asarray(f(tn, xn), dtype=float) * dt
            else:  # rk4
                k1 = np.asarray(f(tn, xn), dtype=float)
                k2 = np.asarray(f(tn + dt / 2, xn + k1 * dt / 2), dtype=float)
                k3 = np.asarray(f(tn + dt / 2, xn + k2 * dt / 2), dtype=float)
                k4 = np.asarray(f(tn + dt, xn + k3 * dt), dtype=float)
                x[n + 1] = xn + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        return t, (x[:, 0] if d == 1 else x)

    # ---- RK45 (Runge-Kutta-Fehlberg, adaptivt tidsskridt) ----
    c2, c3, c4, c5, c6 = 1/4, 3/8, 12/13, 1.0, 1/2
    a21 = 1/4
    a31, a32 = 3/32, 9/32
    a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
    a51, a52, a53, a54 = 439/216, -8.0, 3680/513, -845/4104
    a61, a62, a63, a64, a65 = -8/27, 2.0, -3544/2565, 1859/4104, -11/40
    b1, b3, b4, b5 = 25/216, 1408/2565, 2197/4104, -1/5            # 4. orden
    d1, d3, d4, d5, d6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55  # 5. orden

    t_liste = [0.0]
    x_liste = [start.copy()]
    t, x, h = 0.0, start.copy(), dt
    sikkerhed = 0
    while t < t_slut:
        if t + h > t_slut:
            h = t_slut - t
        k1 = np.asarray(f(t, x), dtype=float)
        k2 = np.asarray(f(t + c2*h, x + h*(a21*k1)), dtype=float)
        k3 = np.asarray(f(t + c3*h, x + h*(a31*k1 + a32*k2)), dtype=float)
        k4 = np.asarray(f(t + c4*h, x + h*(a41*k1 + a42*k2 + a43*k3)), dtype=float)
        k5 = np.asarray(f(t + c5*h, x + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4)), dtype=float)
        k6 = np.asarray(f(t + c6*h, x + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)), dtype=float)
        x4 = x + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5)
        x5 = x + h*(d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        TE = float(np.max(np.abs(x5 - x4)))
        if TE < tolerance or h <= 1e-12:
            t += h
            x = x5
            t_liste.append(t)
            x_liste.append(x.copy())
        # juster tidsskridtet til næste forsøg
        if TE > 0:
            h = 0.9 * h * (tolerance / TE) ** 0.2
        else:
            h = 2 * h
        sikkerhed += 1
        if sikkerhed > 5_000_000:
            raise RuntimeError(
                "RK45 brugte for mange skridt. Prøv en større tolerance eller "
                "en kortere t_slut.")
    t = np.array(t_liste)
    x = np.array(x_liste)
    return t, (x[:, 0] if d == 1 else x)


# ========================================================================
# 2) Stokastiske ligninger:  dX = drift*dt + stoej*dW   (Euler-Maruyama)
# ========================================================================
def simuler_maruyama(drift, stoej, start, t_slut, dt, antal_baner=1, seed=None):
    """Simulér en stokastisk differentialligning med Euler-Maruyama-metoden.

    ``X_{n+1} = X_n + drift(t, X)*dt + stoej(t, X)*sqrt(dt)*xi``

    hvor ``xi`` er et tilfældigt tal fra en normalfordeling (middel 0, spredning 1).

    Parametre
    ---------
    drift : funktion ``drift(t, X)`` – den "glatte" del (kan være 0).
    stoej : funktion ``stoej(t, X)`` – hvor meget tilfældighed (fx sqrt(2D)).
    start : startværdien (et tal).
    t_slut, dt : simuleringstid og tidsskridt.
    antal_baner : hvor mange uafhængige baner der simuleres på én gang.

    Returnerer
    ----------
    t : array med tidspunkter, form (antal_tider,).
    X : array med form (antal_baner, antal_tider).
    """
    _kraev_positiv("dt", dt)
    _kraev_positiv("t_slut", t_slut)
    _kraev_positivt_heltal("antal_baner", antal_baner)
    rng = np.random.default_rng(seed)

    antal = int(np.ceil(t_slut / dt))
    t = np.arange(antal + 1) * dt
    X = np.zeros((antal_baner, antal + 1))
    X[:, 0] = float(start)

    # Tjek brugerens funktioner én gang
    for navn, func in (("drift", drift), ("stoej", stoej)):
        try:
            ud = np.asarray(func(0.0, X[:, 0]), dtype=float)
            np.broadcast_to(ud, (antal_baner,))
        except Exception as e:
            raise ValueError(
                "Kunne ikke bruge din '%s'-funktion. Den skal kunne kaldes som "
                "%s(t, X) og returnere et tal eller et array, der passer til "
                "antal_baner. Oprindelig fejl: %r" % (navn, navn, e))

    sqdt = np.sqrt(dt)
    for n in range(antal):
        tn, Xn = t[n], X[:, n]
        a = np.asarray(drift(tn, Xn), dtype=float)
        b = np.asarray(stoej(tn, Xn), dtype=float)
        xi = rng.standard_normal(antal_baner)
        X[:, n + 1] = Xn + a * dt + b * sqdt * xi
    return t, X


def simuler_brownsk(D, start=0.0, t_slut=1.0, dt=0.01, antal_baner=1, seed=None):
    """Brownsk bevægelse med diffusionskonstant ``D``.

    Det er bare ``simuler_maruyama`` med drift = 0 og stoej = sqrt(2D)."""
    _kraev_positiv("D", D)
    return simuler_maruyama(
        drift=lambda t, X: 0.0,
        stoej=lambda t, X: np.sqrt(2.0 * D),
        start=start, t_slut=t_slut, dt=dt, antal_baner=antal_baner, seed=seed,
    )


# ========================================================================
# 3) Tilfældig gang på et gitter
# ========================================================================
def simuler_tilfaeldig_gang(antal_skridt, antal_baner, dim=1, seed=None):
    """Simulér ``antal_baner`` tilfældige gange med ``antal_skridt`` skridt hver.

    Hvert skridt går +1 eller -1 langs en tilfældig akse.

    Returnerer
    ----------
    baner : for dim=1 et array med form (antal_baner, antal_skridt+1).
            For dim=2 eller 3 form (antal_baner, antal_skridt+1, dim).
    """
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    _kraev_positivt_heltal("antal_baner", antal_baner)
    if dim not in (1, 2, 3):
        raise ValueError("dim skal være 1, 2 eller 3 (du gav %r)." % (dim,))
    if antal_baner * (antal_skridt + 1) * dim > 50_000_000:
        raise ValueError(
            "Det bliver for stort (antal_baner * antal_skridt * dim). "
            "Prøv færre baner eller færre skridt.")

    seed = _lav_seed(seed)
    baner = _koer_kerne(kerne_tilfaeldig_gang, antal_skridt, antal_baner, dim, seed)
    if dim == 1:
        return baner[:, :, 0]
    return baner


# ========================================================================
# 4) Generisk Monte Carlo: kør et tilfældigt forsøg mange gange
# ========================================================================
def simuler_monte_carlo(forsoeg, antal, seed=None):
    """Kør funktionen ``forsoeg()`` ``antal`` gange og saml resultaterne.

    ``forsoeg`` skal kunne kaldes uden argumenter og returnere et tal.

    Returnerer et array med ``antal`` resultater (fx til et gennemsnit eller
    et histogram).
    """
    _kraev_positivt_heltal("antal", antal)
    if seed is not None:
        np.random.seed(int(seed))

    try:
        foerste = float(forsoeg())
    except TypeError:
        raise ValueError(
            "Din 'forsoeg'-funktion skal kunne kaldes uden argumenter, fx "
            "forsoeg(), og returnere ét tal.")
    except Exception as e:
        raise ValueError("Kunne ikke køre 'forsoeg()'. Oprindelig fejl: %r" % (e,))

    resultater = np.empty(antal)
    resultater[0] = foerste
    for i in range(1, antal):
        resultater[i] = float(forsoeg())
    return resultater


# ========================================================================
# 5) Poisson-proces
# ========================================================================
def simuler_poisson(rate, t_slut, dt=None, metode="haendelse", seed=None):
    """Simulér en Poisson-proces med gennemsnitlig rate ``rate``.

    metode="haendelse" : hændelses-drevet (springer direkte til næste hændelse).
    metode="tid"       : tids-drevet (kræver et tidsskridt ``dt``, og rate*dt < 1).

    Returnerer et array med tidspunkterne for hændelserne. Antallet af
    hændelser er ``len(tider)``.
    """
    _kraev_positiv("rate", rate)
    _kraev_positiv("t_slut", t_slut)
    rng = np.random.default_rng(seed)
    metode = str(metode).lower()

    if metode in ("haendelse", "hændelse", "event"):
        tider = []
        t = 0.0
        while True:
            tau = -np.log(rng.random()) / rate
            t += tau
            if t >= t_slut:
                break
            tider.append(t)
        return np.array(tider)

    if metode in ("tid", "time"):
        if dt is None:
            raise ValueError("Tids-drevet Poisson kræver et tidsskridt dt.")
        _kraev_positiv("dt", dt)
        if rate * dt >= 1:
            raise ValueError(
                "rate*dt = %.3g er ikke mindre end 1. Vælg et mindre dt, så "
                "rate*dt < 1 (ellers kan der ske mere end én hændelse pr. skridt)."
                % (rate * dt))
        antal = int(np.ceil(t_slut / dt))
        u = rng.random(antal)
        skridt_med_haendelse = np.nonzero(u < rate * dt)[0]
        return (skridt_med_haendelse + 1) * dt

    raise ValueError("metode skal være 'haendelse' eller 'tid' (du gav %r)." % metode)


# ========================================================================
# 6) Gillespie-algoritmen (hændelses-drevet for flere reaktioner)
# ========================================================================
def simuler_gillespie(start_tilstand, reaktioner, t_slut, seed=None,
                      maks_skridt=1_000_000):
    """Simulér et system af tilfældige hændelser med Gillespie-algoritmen.

    Parametre
    ---------
    start_tilstand : liste/array med starttal, fx [S, I, R] = [990, 10, 0].
    reaktioner : liste af par ``(rate_funktion, aendring)``:
        - ``rate_funktion(tilstand)`` returnerer hvor "hurtigt" reaktionen sker
          ud fra den nuværende tilstand.
        - ``aendring`` er et array, der lægges til tilstanden, når reaktionen
          sker, fx [-1, +1, 0] (én S bliver til én I).
    t_slut : hvor længe der simuleres.

    Returnerer
    ----------
    tider : array med tidspunkter, form (antal_hændelser+1,).
    tilstande : array med form (antal_hændelser+1, antal_variable).
    """
    _kraev_positiv("t_slut", t_slut)
    tilstand = np.asarray(start_tilstand, dtype=float).copy()
    d = tilstand.size

    if not reaktioner:
        raise ValueError("Du skal give mindst én reaktion.")
    rate_funktioner = []
    aendringer = []
    for idx, par in enumerate(reaktioner):
        try:
            rate_funktion, aendring = par
        except Exception:
            raise ValueError(
                "Reaktion nr. %d skal være et par (rate_funktion, aendring)." % idx)
        aendring = np.asarray(aendring, dtype=float)
        if aendring.shape != (d,):
            raise ValueError(
                "Reaktion nr. %d: ændrings-vektoren har form %s, men tilstanden "
                "har %d komponenter." % (idx, aendring.shape, d))
        try:
            r = float(rate_funktion(tilstand))
        except Exception as e:
            raise ValueError(
                "Reaktion nr. %d: kunne ikke beregne raten. Tjek at rate-"
                "funktionen tager tilstanden som argument. Fejl: %r" % (idx, e))
        if r < 0:
            raise ValueError("Reaktion nr. %d gav en negativ rate." % idx)
        rate_funktioner.append(rate_funktion)
        aendringer.append(aendring)
    aendringer = np.array(aendringer)

    rng = np.random.default_rng(seed)
    tider = [0.0]
    tilstande = [tilstand.copy()]
    t = 0.0
    for skridt in range(maks_skridt):
        if t >= t_slut:
            break
        rater = np.array([float(rf(tilstand)) for rf in rate_funktioner])
        if np.any(rater < 0):
            raise ValueError("En rate blev negativ undervejs – tjek dine rate-funktioner.")
        a0 = rater.sum()
        if a0 <= 0:
            break  # ingen flere hændelser kan ske
        tau = -np.log(rng.random()) / a0
        t += tau
        if t >= t_slut:
            break
        j = rng.choice(len(rater), p=rater / a0)
        tilstand = tilstand + aendringer[j]
        tider.append(t)
        tilstande.append(tilstand.copy())
    else:
        raise RuntimeError(
            "Gillespie nåede %d skridt uden at nå t_slut. Måske er raterne meget "
            "høje, eller t_slut for stor." % maks_skridt)
    return np.array(tider), np.array(tilstande)


# ========================================================================
# 7) TASEP / trafik
# ========================================================================
TasepResultat = namedtuple(
    "TasepResultat",
    ["rumtid", "tider", "taethedsprofil", "middel_taethed", "stroem"])


def simuler_tasep(L, alpha, beta, t_slut, dt, p=1.0, seed=None):
    """Simulér TASEP: trafik på en ensrettet vej med ét spor og ``L`` felter.

    Parametre
    ---------
    L : antal felter på vejen.
    alpha : rate for at en ny bil kører ind ved felt 1 (hvis tomt).
    beta : rate for at en bil forlader vejen ved felt L (hvis optaget).
    p : rate for at en bil rykker ét felt frem (hvis feltet forude er tomt).
    t_slut, dt : simuleringstid og tidsskridt. Kravet er alpha*dt, beta*dt,
                 p*dt < 1.

    Bemærk: opdateringen er tids-diskret (felterne fejes igennem i hvert skridt), så
    modellen svarer til den kontinuerte TASEP i grænsen dt -> 0. For små dt (som her)
    rammer den fasediagrammet og strømmen korrekt.

    Returnerer (et navngivet par – du kan skrive res.rumtid osv.)
    ----------
    rumtid : array (antal_tider, L) med 0/1 = tomt/bil. Brug ``plot_rumtid``.
    tider : tidspunkterne.
    taethedsprofil : gennemsnitlig biltæthed pr. felt (efter indkøring).
    middel_taethed : den samlede gennemsnitlige tæthed (ét tal).
    stroem : den gennemsnitlige bilstrøm (biler pr. tid).
    """
    _kraev_positivt_heltal("L", L)
    _kraev_positiv("dt", dt)
    _kraev_positiv("t_slut", t_slut)
    for navn, vaerdi in (("alpha", alpha), ("beta", beta), ("p", p)):
        if vaerdi < 0:
            raise ValueError("%s må ikke være negativ." % navn)
        if vaerdi * dt >= 1:
            raise ValueError(
                "%s*dt = %.3g er ikke mindre end 1. Vælg et mindre dt."
                % (navn, vaerdi * dt))

    antal = int(np.ceil(t_slut / dt))
    seed = _lav_seed(seed)
    historik = _koer_kerne(
        kerne_tasep, L, float(alpha * dt), float(p * dt), float(beta * dt),
        antal, seed)
    tider = np.arange(antal + 1) * dt

    # statistik efter "indkøring" (anden halvdel af tiden)
    burn = antal // 2
    del_h = historik[burn:].astype(float)
    taethedsprofil = del_h.mean(axis=0)
    middel_taethed = float(taethedsprofil.mean())
    par_10 = (del_h[:, :-1] == 1) & (del_h[:, 1:] == 0)
    stroem = float(p * par_10.mean())
    return TasepResultat(historik, tider, taethedsprofil, middel_taethed, stroem)


# ========================================================================
# 7b) Trafik med DIN EGEN hastighedsmodel (ringvej, "bil-følge-model")
# ========================================================================
TrafikResultat = namedtuple("TrafikResultat", ["rumtid", "middel_fart", "stroem"])


def simuler_trafik(hastighedsmodel, L, antal_biler, antal_skridt,
                   fart_start=0, seed=None):
    """Simulér trafik på en ringvej, hvor DU bestemmer hastighedsmodellen.

    Hver bil har en position og en fart (hvor mange felter den rykker pr. skridt).
    For hvert skridt spørger vi din funktion, hvor hurtigt hver bil vil køre:

        hastighedsmodel(fart, afstand)  ->  ny fart (et helt tal)

    hvor
        fart    = bilens nuværende fart
        afstand = antal tomme felter frem til bilen foran

    Farten begrænses automatisk til ``afstand``, så biler aldrig kører ind i
    hinanden -- så du kan frit lege med modellen uden at lave uheld.

    Eksempler på modeller (prøv dem!):
        def model(fart, afstand): return afstand        # uendelig acceleration
        def model(fart, afstand): return fart + 1       # langsom acceleration
        def model(fart, afstand): return min(fart + 1, 5)  # med fartgrænse

    Returnerer (navngivet par)
    ----------
    rumtid : array (antal_skridt+1, L) med 0/1 = tomt/bil. Brug ``plot_rumtid``.
    middel_fart : bilernes gennemsnitsfart (efter indkøring).
    stroem : trafikstrømmen = tæthed * middel_fart.
    """
    _kraev_positivt_heltal("L", L)
    _kraev_positivt_heltal("antal_biler", antal_biler)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    if antal_biler >= L:
        raise ValueError(
            "antal_biler skal være mindre end L, ellers er der ikke plads til "
            "at køre (du gav antal_biler=%d, L=%d)." % (antal_biler, L))

    # jævn startplacering på ringvejen
    pos = (np.arange(antal_biler) * L // antal_biler).astype(int)
    if len(np.unique(pos)) < antal_biler:
        pos = np.arange(antal_biler)        # nødplan: de første felter
    fart = np.full(antal_biler, int(fart_start))

    # tjek brugerens model én gang
    try:
        int(hastighedsmodel(int(fart[0]), 1))
    except Exception as e:
        raise ValueError(
            "Kunne ikke bruge din hastighedsmodel. Den skal kunne kaldes som "
            "hastighedsmodel(fart, afstand) og returnere et helt tal (den nye "
            "fart). Oprindelig fejl: %r" % (e,))

    rumtid = np.zeros((antal_skridt + 1, L), dtype=np.int8)
    fart_per_skridt = []
    for n in range(antal_skridt + 1):
        orden = np.argsort(pos)
        pos = pos[orden]
        fart = fart[orden]
        rumtid[n, pos] = 1
        if n == antal_skridt:
            break
        # afstand til bilen foran (rundt om ringen)
        afstand = (np.roll(pos, -1) - pos - 1) % L
        ny_fart = np.empty(antal_biler, dtype=int)
        for i in range(antal_biler):
            oensket = int(hastighedsmodel(int(fart[i]), int(afstand[i])))
            if oensket < 0:
                oensket = 0
            ny_fart[i] = min(oensket, int(afstand[i]))  # ingen sammenstød
        pos = (pos + ny_fart) % L
        fart = ny_fart
        fart_per_skridt.append(ny_fart.mean())

    halv = len(fart_per_skridt) // 2
    middel_fart = float(np.mean(fart_per_skridt[halv:])) if fart_per_skridt else 0.0
    stroem = float((antal_biler / L) * middel_fart)
    return TrafikResultat(rumtid, middel_fart, stroem)


# ========================================================================
# 8) Ising-model (magnetisme)
# ========================================================================
IsingResultat = namedtuple(
    "IsingResultat", ["gitter", "magnetisering", "middel_magnetisering"])


def simuler_ising(L, T, antal_fejecyklusser=200, seed=None):
    """Simulér en 2D Ising-model (magnetisme) ved temperatur ``T``.

    Returnerer (navngivet par)
    ----------
    gitter : det endelige gitter af spins (+1 / -1), form (L, L).
    magnetisering : gennemsnitlig magnetisering pr. fejecyklus.
    middel_magnetisering : |magnetisering| i gennemsnit (efter indkøring).
    """
    _kraev_positivt_heltal("L", L)
    _kraev_positiv("T", T)
    _kraev_positivt_heltal("antal_fejecyklusser", antal_fejecyklusser)
    seed = _lav_seed(seed)
    gitter, magnetisering = _koer_kerne(
        kerne_ising, L, float(T), antal_fejecyklusser, seed)
    burn = antal_fejecyklusser // 2
    middel = float(np.mean(np.abs(magnetisering[burn:])))
    return IsingResultat(gitter, magnetisering, middel)


# ========================================================================
# 9) Skovbrand (bonus)
# ========================================================================
def simuler_skovbrand(L, p_vaekst=0.01, p_lyn=0.0006, antal_skridt=200, seed=None):
    """Simulér en skovbrand-celleautomat.

    Returnerer ``historik`` med form (antal_skridt+1, L, L), hvor værdierne er
    0 = tom, 1 = træ, 2 = brændende. Brug ``animer_gitter`` til at se den."""
    _kraev_positivt_heltal("L", L)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    seed = _lav_seed(seed)
    return _koer_kerne(kerne_skovbrand, L, float(p_vaekst), float(p_lyn),
                       antal_skridt, seed)


# ========================================================================
# 10) Epidemi på et gitter (rumlig SIR-celleautomat)
# ========================================================================
EpidemiGitterResultat = namedtuple(
    "EpidemiGitterResultat",
    ["historik", "gitter", "tid", "antal", "S", "I", "R"])


def simuler_epidemi_gitter(L=100, p_smitte=0.3, p_rask=0.1, p_vaccineret=0.0,
                           antal_smittede_start=5, antal_skridt=200,
                           naboer=4, seed=None):
    """Simulér en epidemi på et gitter, hvor hver celle er én person.

    Hver person er enten **modtagelig** (0), **smittet** (1) eller **immun** (2).
    For hvert tidsskridt:

    - En smittet bliver rask (og immun) med sandsynligheden ``p_rask``.
    - En modtagelig person smittes af hver smittet nabo med sandsynligheden
      ``p_smitte``. Har man flere smittede naboer, er risikoen større.

    Til forskel fra SIR-differentialligningen møder man her kun sine **naboer** --
    ikke hele befolkningen. Derfor breder smitten sig som en bølge.

    Parametre
    ---------
    L : gitteret er L x L personer.
    p_smitte : smitte-sandsynlighed pr. smittet nabo pr. skridt.
    p_rask : sandsynlighed for at en smittet bliver rask pr. skridt.
    p_vaccineret : andelen der er immune fra start (vaccineret).
    antal_smittede_start : hvor mange der er smittede til at begynde med.
    naboer : 4 (op/ned/venstre/højre) eller 8 (også diagonalt).

    Returnerer (navngivet par)
    ----------
    historik : (antal_skridt+1, L, L) med 0/1/2. Brug ``animer_gitter``.
    gitter : slutbilledet. Brug ``vis_gitter``.
    tid : array med tidsskridtene.
    antal : (antal_skridt+1, 3) med antal [S, I, R] til hvert tidspunkt.
            Brug ``plot_baner(res.tid, res.antal, labels=["S", "I", "R"])``.
    S, I, R : de tre søjler i ``antal`` hver for sig.
    """
    _kraev_positivt_heltal("L", L)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    _kraev_positivt_heltal("antal_smittede_start", antal_smittede_start)
    _kraev_sandsynlighed("p_smitte", p_smitte)
    _kraev_sandsynlighed("p_rask", p_rask)
    _kraev_sandsynlighed("p_vaccineret", p_vaccineret)
    if naboer not in (4, 8):
        raise ValueError(
            "naboer skal være 4 (op/ned/venstre/højre) eller 8 (også skråt) "
            "-- du gav %r." % (naboer,))
    if (antal_skridt + 1) * L * L > 100_000_000:
        raise ValueError(
            "Det bliver for stort: (antal_skridt+1) * L * L = %d. Prøv et "
            "mindre gitter eller færre skridt."
            % ((antal_skridt + 1) * L * L))

    rng = np.random.default_rng(seed)
    gitter = np.zeros((L, L), dtype=np.int8)          # alle er modtagelige
    gitter[rng.random((L, L)) < p_vaccineret] = 2     # de vaccinerede er immune

    ledige = np.flatnonzero(gitter == 0)
    if antal_smittede_start > ledige.size:
        raise ValueError(
            "Der er kun %d personer, som ikke er vaccineret, så der er ikke "
            "plads til %d smittede. Vælg færre smittede eller en lavere "
            "p_vaccineret." % (ledige.size, antal_smittede_start))
    gitter.ravel()[rng.choice(ledige, size=antal_smittede_start, replace=False)] = 1

    historik = np.zeros((antal_skridt + 1, L, L), dtype=np.int8)
    antal = np.zeros((antal_skridt + 1, 3), dtype=np.int64)

    for n in range(antal_skridt + 1):
        historik[n] = gitter
        antal[n] = [int(np.count_nonzero(gitter == k)) for k in (0, 1, 2)]
        if n == antal_skridt:
            break

        smittet = (gitter == 1)
        # tæl smittede naboer (periodiske rande, som i skovbranden)
        naboer_smittet = (np.roll(smittet, 1, 0).astype(np.int16)
                          + np.roll(smittet, -1, 0)
                          + np.roll(smittet, 1, 1)
                          + np.roll(smittet, -1, 1))
        if naboer == 8:
            naboer_smittet = (naboer_smittet
                              + np.roll(smittet, (1, 1), (0, 1))
                              + np.roll(smittet, (1, -1), (0, 1))
                              + np.roll(smittet, (-1, 1), (0, 1))
                              + np.roll(smittet, (-1, -1), (0, 1)))

        # hver smittet nabo er et selvstændigt forsøg
        p_bliver_smittet = 1.0 - (1.0 - p_smitte) ** naboer_smittet
        ny_smittet = (gitter == 0) & (rng.random((L, L)) < p_bliver_smittet)
        ny_rask = smittet & (rng.random((L, L)) < p_rask)

        # begge er regnet ud fra det GAMLE gitter, så ingen når at blive
        # smittet og rask i samme skridt
        gitter[ny_rask] = 2
        gitter[ny_smittet] = 1

    tid = np.arange(antal_skridt + 1)
    return EpidemiGitterResultat(historik, historik[-1], tid, antal,
                                 antal[:, 0], antal[:, 1], antal[:, 2])


# ========================================================================
# 11) Epidemi blandt personer, der bevæger sig (agent-baseret SIR)
# ========================================================================
EpidemiAgentResultat = namedtuple(
    "EpidemiAgentResultat",
    ["positioner", "status", "tid", "antal", "S", "I", "R", "L"])


def simuler_epidemi_agenter(antal_personer=400, L=20.0, fart=0.25, radius=0.5,
                            p_smitte=0.3, p_rask=0.02, antal_smittede_start=5,
                            andel_hjemme=0.0, andel_vaccineret=0.0,
                            antal_skridt=600, seed=None):
    """Simulér en epidemi blandt personer, der går tilfældigt rundt i en kasse.

    Hver person er **modtagelig** (0), **smittet** (1) eller **immun** (2) og tager
    et tilfældigt skridt af længden ``fart`` for hvert tidsskridt (en tilfældig gang!).
    Er en modtagelig person tættere end ``radius`` på en smittet, kan hun blive
    smittet med sandsynligheden ``p_smitte``.

    Parametre
    ---------
    antal_personer : hvor mange personer der er.
    L : kassen er L x L stor (personerne studser mod væggene).
    fart : hvor langt man går pr. skridt. **Sæt den ned = bliv mere hjemme.**
    radius : hvor tæt man skal være på hinanden for at smitte.
    p_smitte : smitte-sandsynlighed pr. smittet person indenfor radius pr. skridt.
    p_rask : sandsynlighed for at blive rask pr. skridt.
    andel_hjemme : andelen der slet ikke bevæger sig (nedlukning).
    andel_vaccineret : andelen der er immune fra start (vaccination).

    Med ``andel_hjemme`` og ``andel_vaccineret`` kan du sammenligne to måder at
    stoppe en epidemi på: at folk mødes mindre, eller at færre kan smittes.

    Returnerer (navngivet par)
    ----------
    positioner : (antal_skridt+1, antal_personer, 2). Brug ``animer_partikler``.
    status : (antal_skridt+1, antal_personer) med 0/1/2.
    tid, antal, S, I, R : som i ``simuler_epidemi_gitter``.
    L : kassens størrelse (så ``animer_partikler`` selv kan sætte akserne).
    """
    _kraev_positivt_heltal("antal_personer", antal_personer)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    _kraev_positivt_heltal("antal_smittede_start", antal_smittede_start)
    _kraev_positiv("L", L)
    _kraev_positiv("radius", radius)
    _kraev_ikke_negativ("fart", fart)
    _kraev_sandsynlighed("p_smitte", p_smitte)
    _kraev_sandsynlighed("p_rask", p_rask)
    _kraev_sandsynlighed("andel_hjemme", andel_hjemme)
    _kraev_sandsynlighed("andel_vaccineret", andel_vaccineret)
    if fart > L:
        raise ValueError(
            "fart (%.3g) er større end kassen L (%.3g). Så springer personerne "
            "ud af kassen. Vælg en mindre fart." % (fart, L))
    if antal_smittede_start > antal_personer:
        raise ValueError(
            "Der kan ikke være %d smittede, når der kun er %d personer."
            % (antal_smittede_start, antal_personer))
    if antal_personer > 2000:
        raise ValueError(
            "antal_personer = %d er for mange (animationen bliver ubrugelig). "
            "Hold dig under 2000." % antal_personer)

    rng = np.random.default_rng(seed)
    N = antal_personer
    pos = rng.random((N, 2)) * L
    status = np.zeros(N, dtype=np.int8)

    antal_vaccineret = int(round(andel_vaccineret * N))
    if antal_vaccineret > 0:
        status[rng.choice(N, size=antal_vaccineret, replace=False)] = 2
    ledige = np.flatnonzero(status == 0)
    if antal_smittede_start > ledige.size:
        raise ValueError(
            "Der er kun %d personer, som ikke er vaccineret, så der er ikke "
            "plads til %d smittede. Vælg færre smittede eller en lavere "
            "andel_vaccineret." % (ledige.size, antal_smittede_start))
    status[rng.choice(ledige, size=antal_smittede_start, replace=False)] = 1

    hjemme = np.zeros(N, dtype=bool)
    antal_hjemme = int(round(andel_hjemme * N))
    if antal_hjemme > 0:
        hjemme[rng.choice(N, size=antal_hjemme, replace=False)] = True
    mobil = ~hjemme

    positioner = np.zeros((antal_skridt + 1, N, 2), dtype=np.float32)
    statusser = np.zeros((antal_skridt + 1, N), dtype=np.int8)
    antal = np.zeros((antal_skridt + 1, 3), dtype=np.int64)

    for n in range(antal_skridt + 1):
        positioner[n] = pos
        statusser[n] = status
        antal[n] = [int(np.count_nonzero(status == k)) for k in (0, 1, 2)]
        if n == antal_skridt:
            break

        # 1) alle, der ikke er hjemme, tager et tilfældigt skridt
        antal_mobile = int(np.count_nonzero(mobil))
        if antal_mobile > 0 and fart > 0:
            vinkel = rng.random(antal_mobile) * 2.0 * np.pi
            pos[mobil] += fart * np.stack([np.cos(vinkel), np.sin(vinkel)], axis=1)
            pos = np.abs(pos)              # studs mod venstre/nederste væg
            pos = L - np.abs(L - pos)      # studs mod højre/øverste væg

        # 2) hvem bliver rask, og hvem bliver smittet? (begge ud fra NUVÆRENDE status)
        modtagelige = np.flatnonzero(status == 0)
        smittede = np.flatnonzero(status == 1)
        ny_rask = smittede[rng.random(smittede.size) < p_rask]

        ny_smittet = np.empty(0, dtype=np.int64)
        if modtagelige.size > 0 and smittede.size > 0:
            forskel = pos[modtagelige][:, None, :] - pos[smittede][None, :, :]
            naer = ((forskel ** 2).sum(axis=-1) < radius * radius).sum(axis=1)
            p_bliver_smittet = 1.0 - (1.0 - p_smitte) ** naer
            ny_smittet = modtagelige[rng.random(modtagelige.size) < p_bliver_smittet]

        status[ny_rask] = 2
        status[ny_smittet] = 1

    tid = np.arange(antal_skridt + 1)
    return EpidemiAgentResultat(positioner, statusser, tid, antal,
                                antal[:, 0], antal[:, 1], antal[:, 2], float(L))


# ========================================================================
# 12) Wright-Fisher: genetisk drift
# ========================================================================
WrightFisherResultat = namedtuple(
    "WrightFisherResultat",
    ["frekvenser", "tid", "andel_fikseret", "andel_tabt", "heterozygositet"])


def simuler_wright_fisher(antal_individer=100, start_frekvens=0.5,
                          generationer=200, antal_populationer=200,
                          s=0.0, seed=None):
    """Simulér genetisk drift: hvordan en gen-variant spreder sig ved ren tilfældighed.

    En population har ``antal_individer`` gen-kopier. En andel ``p`` af dem er den
    variant, vi følger. Hver ny generation trækkes ved at vælge ``antal_individer``
    forældre **tilfældigt med tilbagelægning** -- ren tilfældighed, ingen "bedste"
    vinder. Med tiden ender ``p`` altid enten i 0 (varianten er **tabt**) eller i
    1 (varianten er **fikseret**).

    Med en selektions-fordel ``s`` er varianten ``(1+s)`` gange så god til at blive
    valgt som forælder. ``s = 0`` er ren drift; ``s = 0.1`` er 10 % bedre.

    Bemærk: ``antal_individer`` tælles i **gen-kopier** (haploid), så variationen
    forsvinder som ``(1 - 1/N)^t`` -- ikke ``(1 - 1/2N)^t``.

    Returnerer (navngivet par)
    ----------
    frekvenser : (antal_populationer, generationer+1). Brug ``plot_baner``.
    tid : generationsnumrene.
    andel_fikseret : hvor stor en del af populationerne der endte i p = 1.
    andel_tabt : hvor stor en del der endte i p = 0.
    heterozygositet : 2p(1-p) midlet over populationerne -- et mål for, hvor meget
                      genetisk variation der er tilbage.
    """
    _kraev_positivt_heltal("antal_individer", antal_individer)
    _kraev_positivt_heltal("generationer", generationer)
    _kraev_positivt_heltal("antal_populationer", antal_populationer)
    _kraev_sandsynlighed("start_frekvens", start_frekvens)
    if s <= -1:
        raise ValueError(
            "s skal være større end -1 (ellers ville varianten have negativ "
            "fitness) -- du gav %r." % (s,))
    if antal_populationer * (generationer + 1) > 20_000_000:
        raise ValueError(
            "Det bliver for stort: antal_populationer * generationer = %d. "
            "Prøv færre populationer eller færre generationer."
            % (antal_populationer * (generationer + 1)))

    rng = np.random.default_rng(seed)
    frekvenser = np.zeros((antal_populationer, generationer + 1), dtype=np.float32)
    p = np.full(antal_populationer, float(start_frekvens))
    frekvenser[:, 0] = p

    for g in range(generationer):
        vaegt = p * (1.0 + s)
        andel = vaegt / (vaegt + (1.0 - p))     # chancen for at blive valgt
        p = rng.binomial(antal_individer, andel) / antal_individer
        frekvenser[:, g + 1] = p

    heterozygositet = (2.0 * frekvenser * (1.0 - frekvenser)).mean(axis=0)
    return WrightFisherResultat(
        frekvenser,
        np.arange(generationer + 1),
        float(np.mean(p >= 1.0)),
        float(np.mean(p <= 0.0)),
        heterozygositet,
    )


# ========================================================================
# 13) En neuron, der fyrer (integrate-and-fire med støj)
# ========================================================================
NeuronResultat = namedtuple(
    "NeuronResultat", ["t", "V", "spike_tider", "fyringsrate"])


def simuler_neuron(stroem=1.2, stoej=0.5, t_slut=1000.0, dt=0.1, tau=20.0,
                   modstand=1.0, V_hvile=0.0, V_taerskel=1.0, V_reset=0.0,
                   t_refraktaer=2.0, seed=None):
    """Simulér en neuron med "integrate-and-fire"-modellen.

    Neuronens spænding ``V`` lækker langsomt tilbage mod hvilepotentialet, mens den
    strøm, den får ind, skubber den op::

        dV/dt = ( -(V - V_hvile) + modstand * stroem ) / tau   + støj

    Når ``V`` når tærsklen ``V_taerskel``, **fyrer** neuronen (et "spike"), spændingen
    nulstilles til ``V_reset``, og den holder pause i ``t_refraktaer``. Selve spiket
    tegnes ikke -- det er bare tidspunktet, der tælles.

    Støjen lægges til med Euler-Maruyama -- skaleret med sqrt(dt/tau), altså kvadratroden
    af tidsskridtet (1/sqrt(tau)-faktoren holder spændingens spredning uafhængig af tau).

    Tiden måles i millisekunder, så ``fyringsrate`` kommer ud i hertz (spikes/sekund).

    Parametre
    ---------
    stroem : den strøm, neuronen får ind. Enten et tal ELLER en funktion
             ``stroem(t)``, hvis du vil skrue op og ned undervejs.
    stoej : hvor meget tilfældig baggrundsstøj neuronen får (0 = ingen).
    tau : membranens tidskonstant (hvor hurtigt spændingen lækker tilbage).

    Returnerer (navngivet par)
    ----------
    t, V : tid og spænding. Brug ``plot_baner(res.t, res.V)``.
    spike_tider : tidspunkterne for spikes.
    fyringsrate : antal spikes pr. sekund (Hz).
    """
    _kraev_positiv("dt", dt)
    _kraev_positiv("t_slut", t_slut)
    _kraev_positiv("tau", tau)
    _kraev_ikke_negativ("stoej", stoej)
    _kraev_ikke_negativ("t_refraktaer", t_refraktaer)
    if V_taerskel <= V_hvile:
        raise ValueError(
            "V_taerskel (%.3g) skal ligge over V_hvile (%.3g). Ellers ville "
            "neuronen fyre hele tiden." % (V_taerskel, V_hvile))
    if dt > tau / 5.0:
        raise ValueError(
            "dt = %.3g er for stort i forhold til tidskonstanten tau = %.3g. "
            "Vælg højst dt = tau/5 = %.3g, ellers bliver simuleringen unøjagtig."
            % (dt, tau, tau / 5.0))

    if callable(stroem):
        stroem_funktion = stroem
        try:
            float(stroem_funktion(0.0))
        except Exception as e:
            raise ValueError(
                "Kunne ikke bruge din strøm-funktion. Den skal kunne kaldes som "
                "stroem(t) og returnere ét tal. Oprindelig fejl: %r" % (e,))
    else:
        _fast_stroem = float(stroem)
        stroem_funktion = lambda t: _fast_stroem

    rng = np.random.default_rng(seed)
    antal = int(np.ceil(t_slut / dt))
    t = np.arange(antal + 1) * dt
    V = np.zeros(antal + 1)
    V[0] = V_hvile
    xi = rng.standard_normal(antal)
    sqdt = np.sqrt(dt / tau)

    spike_tider = []
    v = float(V_hvile)
    pause_indtil = -1.0
    for n in range(antal):
        if t[n] < pause_indtil:
            v = V_reset                      # neuronen holder pause (refraktær)
        else:
            I = float(stroem_funktion(t[n]))
            v = (v + (-(v - V_hvile) + modstand * I) * dt / tau
                 + stoej * sqdt * xi[n])
            if v >= V_taerskel:
                spike_tider.append(t[n + 1])
                v = V_reset
                pause_indtil = t[n + 1] + t_refraktaer
        V[n + 1] = v

    fyringsrate = len(spike_tider) / (t_slut / 1000.0)
    return NeuronResultat(t, V, np.array(spike_tider), float(fyringsrate))


# ========================================================================
# 14) Reaktion-diffusion: Turing-mønstre (pletter og striber)
# ========================================================================
ReaktionDiffusionResultat = namedtuple(
    "ReaktionDiffusionResultat", ["historik", "gitter", "U", "V", "tid"])

# (F, k)-værdier der giver kendte mønstre (Du=0.16, Dv=0.08, dt=1)
MOENSTRE = {
    "pletter":  (0.035, 0.065),
    "mitose":   (0.0367, 0.0649),
    "striber":  (0.035, 0.060),
    "labyrint": (0.029, 0.057),
    "koral":    (0.055, 0.062),
    "bobler":   (0.014, 0.054),
}


def simuler_reaktion_diffusion(L=128, F=0.035, k=0.065, Du=0.16, Dv=0.08,
                               dt=1.0, antal_skridt=8000, gem_hvert=80,
                               start="pletter", seed=None):
    """Simulér to stoffer, der reagerer med hinanden OG diffunderer (Gray-Scott).

    To stoffer ligger i hvert punkt på et gitter: ``U`` (føde) og ``V`` (mønster).
    ``V`` laver mere af sig selv ved at spise ``U`` (``U + 2V -> 3V``), ``U`` fyldes
    hele tiden op med raten ``F``, og ``V`` fjernes med raten ``F + k``. Samtidig
    **diffunderer** begge stoffer -- og ``U`` diffunderer hurtigst.

    Det er hele opskriften. Alan Turing viste i 1952, at netop den slags system helt
    af sig selv laver **mønstre**: pletter, striber og labyrinter. Det er den bedste
    forklaring, vi har, på hvorfor geparden har pletter og zebraen striber.

    Metoden er **Eulers metode -- bare i hvert punkt på gitteret** i stedet for i én
    variabel. Diffusion regnes ud ved at sammenligne hvert punkt med dets fire naboer.

    Prøv disse (F, k)-værdier::

        pletter   F=0.035,  k=0.065     mitose    F=0.0367, k=0.0649
        striber   F=0.035,  k=0.060     labyrint  F=0.029,  k=0.057
        koral     F=0.055,  k=0.062     bobler    F=0.014,  k=0.054

    Returnerer (navngivet par)
    ----------
    historik : (antal_billeder, L, L) med V-feltet. Brug
               ``animer_gitter(res, cmap="inferno")``.
    gitter : slutbilledet af V. Brug ``vis_gitter(res, cmap="inferno")``.
    U, V : de to stof-felter til slut.
    tid : tidspunktet for hvert gemt billede.
    """
    _kraev_positivt_heltal("L", L)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    _kraev_positivt_heltal("gem_hvert", gem_hvert)
    _kraev_positiv("dt", dt)
    for navn, vaerdi in (("F", F), ("k", k), ("Du", Du), ("Dv", Dv)):
        _kraev_ikke_negativ(navn, vaerdi)
    if start not in ("pletter", "plet"):
        raise ValueError(
            "start skal være \"pletter\" (mange tilfældige klatter, fylder hele "
            "gitteret) eller \"plet\" (én klat i midten) -- du gav %r." % (start,))
    if max(Du, Dv) * dt > 0.25:
        raise ValueError(
            "max(Du, Dv) * dt = %.3g er større end 0.25. Så bliver simuleringen "
            "ustabil, og tallene eksploderer (præcis samme problem som et for "
            "stort tidsskridt i Eulers metode). Vælg et mindre dt eller en mindre "
            "diffusion." % (max(Du, Dv) * dt))

    rng = np.random.default_rng(seed)
    U = np.ones((L, L))
    V = np.zeros((L, L))

    if start == "plet":
        r = max(L // 10, 3)
        m = L // 2
        U[m - r:m + r, m - r:m + r] = 0.50
        V[m - r:m + r, m - r:m + r] = 0.25
    else:
        r = max(L // 25, 2)
        for _ in range(25):
            i, j = rng.integers(0, max(L - 2 * r, 1), size=2)
            U[i:i + 2 * r, j:j + 2 * r] = 0.50
            V[i:i + 2 * r, j:j + 2 * r] = 0.25
    U += 0.02 * rng.standard_normal((L, L))
    V += 0.02 * rng.standard_normal((L, L))
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)

    antal_billeder = antal_skridt // gem_hvert + 1
    historik = np.zeros((antal_billeder, L, L), dtype=np.float32)
    tid = np.zeros(antal_billeder)
    historik[0] = V
    naeste = 1

    for n in range(1, antal_skridt + 1):
        # diffusion: sammenlign hvert punkt med dets fire naboer
        lap_U = (np.roll(U, 1, 0) + np.roll(U, -1, 0)
                 + np.roll(U, 1, 1) + np.roll(U, -1, 1) - 4.0 * U)
        lap_V = (np.roll(V, 1, 0) + np.roll(V, -1, 0)
                 + np.roll(V, 1, 1) + np.roll(V, -1, 1) - 4.0 * V)
        reaktion = U * V * V                       # U + 2V -> 3V
        U += dt * (Du * lap_U - reaktion + F * (1.0 - U))
        V += dt * (Dv * lap_V + reaktion - (F + k) * V)

        if n % gem_hvert == 0 and naeste < antal_billeder:
            historik[naeste] = V
            tid[naeste] = n * dt
            naeste += 1

    return ReaktionDiffusionResultat(historik, V.astype(np.float32), U, V, tid)


# ========================================================================
# 15) Fugleflok uden leder (Vicsek-modellen)
# ========================================================================
FlokResultat = namedtuple(
    "FlokResultat",
    ["positioner", "retninger", "orden", "tid", "middel_orden", "L"])


def simuler_flok(antal_fugle=300, L=10.0, fart=0.3, radius=1.0, stoej=0.5,
                 antal_skridt=400, dt=1.0, seed=None):
    """Simulér en fugleflok med Vicsek-modellen -- tre regler, ingen leder.

    Hver fugle flyver med **konstant fart**, og for hvert tidsskridt gør den kun ét:
    den kigger på alle fugle indenfor ``radius`` og drejer i den **gennemsnitlige
    retning**, de flyver -- plus lidt tilfældig støj (fuglen ser ikke helt præcist).

    Der er ingen leder og ingen plan. Alligevel opstår der en flok. Skruer man op for
    støjen, går flokken i opløsning: det er en **faseovergang**, præcis som i
    Ising-modellen (notebook 05), bare med fugle i stedet for magneter.

    Parametre
    ---------
    antal_fugle : hvor mange fugle.
    L : kassen er L x L (fuglene flyver ud på den ene side og ind på den anden).
    fart : hvor langt en fugl flyver pr. skridt.
    radius : hvor langt en fugl kan se.
    stoej : hvor upræcist fuglen drejer. 0 = perfekt, 6.28 = fuldstændig tilfældigt.

    Returnerer (navngivet par)
    ----------
    positioner : (antal_skridt+1, antal_fugle, 2). Brug ``animer_partikler``.
    retninger : (antal_skridt+1, antal_fugle) med hver fugls vinkel.
    orden : "hvor meget flok" der er, til hvert tidspunkt. 1 = alle flyver samme vej,
            0 = kaos. NB: ved fuldt kaos ender den omkring 1/sqrt(antal_fugle),
            ikke præcis 0 -- det er ikke en fejl.
    middel_orden : orden midlet over anden halvdel (efter indkøring).
    tid, L : tidspunkter og kassens størrelse.
    """
    _kraev_positivt_heltal("antal_fugle", antal_fugle)
    _kraev_positivt_heltal("antal_skridt", antal_skridt)
    _kraev_positiv("L", L)
    _kraev_positiv("radius", radius)
    _kraev_positiv("dt", dt)
    _kraev_ikke_negativ("fart", fart)
    _kraev_ikke_negativ("stoej", stoej)
    if radius > L / 2.0:
        raise ValueError(
            "radius (%.3g) må højst være det halve af L (%.3g). Ellers kan en fugl "
            "se hele vejen rundt om kassen og se de samme fugle to gange."
            % (radius, L / 2.0))
    if antal_fugle ** 2 * antal_skridt > 200_000_000:
        raise ValueError(
            "Det bliver for langsomt: antal_fugle^2 * antal_skridt = %d. Hver fugl "
            "skal kigge på alle de andre, så prøv færre fugle eller færre skridt."
            % (antal_fugle ** 2 * antal_skridt))

    rng = np.random.default_rng(seed)
    N = antal_fugle
    pos = rng.random((N, 2)) * L
    retning = rng.random(N) * 2.0 * np.pi - np.pi

    positioner = np.zeros((antal_skridt + 1, N, 2), dtype=np.float32)
    retninger = np.zeros((antal_skridt + 1, N), dtype=np.float32)
    orden = np.zeros(antal_skridt + 1)

    for n in range(antal_skridt + 1):
        positioner[n] = pos
        retninger[n] = retning
        orden[n] = float(np.abs(np.exp(1j * retning).mean()))
        if n == antal_skridt:
            break

        # hvem kan se hvem? (kassen er periodisk, så vi tager den korteste vej rundt)
        forskel = pos[:, None, :] - pos[None, :, :]
        forskel -= L * np.round(forskel / L)
        ser = ((forskel ** 2).sum(axis=-1) < radius * radius).astype(float)

        # midl naboernes retninger som VEKTORER -- ikke som vinkler!
        # (gennemsnittet af 179 grader og -179 grader er 180, ikke 0)
        sum_sin = ser @ np.sin(retning)
        sum_cos = ser @ np.cos(retning)
        retning = (np.arctan2(sum_sin, sum_cos)
                   + stoej * (rng.random(N) - 0.5))
        retning = (retning + np.pi) % (2.0 * np.pi) - np.pi

        pos = (pos + fart * dt
               * np.stack([np.cos(retning), np.sin(retning)], axis=1)) % L

    halv = (antal_skridt + 1) // 2
    return FlokResultat(positioner, retninger, orden,
                        np.arange(antal_skridt + 1) * dt,
                        float(orden[halv:].mean()), float(L))


# ========================================================================
# Grafer
# ========================================================================
def plot_baner(t, x, labels=None, titel="", xlabel="t", ylabel="x",
               maks_kurver=60):
    """Tegn én eller flere kurver/baner over tid.

    ``x`` kan være 1D (én kurve), (antal_tider, antal_variable) eller
    (antal_baner, antal_tider). Funktionen finder selv ud af det ud fra ``t``."""
    import matplotlib.pyplot as plt
    t = np.asarray(t)
    x = np.asarray(x)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if x.ndim == 1:
        ax.plot(t, x, label=(labels[0] if labels else None))
    else:
        if x.shape[0] == len(t):
            kurver = [x[:, k] for k in range(x.shape[1])]
        elif x.shape[1] == len(t):
            kurver = [x[k, :] for k in range(x.shape[0])]
        else:
            raise ValueError(
                "Formen på x %s passer ikke med længden af t (%d)."
                % (x.shape, len(t)))
        mange = len(kurver) > maks_kurver
        for k, kurve in enumerate(kurver[:maks_kurver]):
            lab = labels[k] if (labels and k < len(labels)) else None
            ax.plot(t, kurve, label=lab, alpha=(0.3 if mange else 0.9))
        if mange:
            ax.set_title((titel + "  (viser %d af %d baner)" % (maks_kurver, len(kurver))).strip())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if titel and not (x.ndim > 1 and len(kurver) > maks_kurver):
        ax.set_title(titel)
    ax.grid(True, alpha=0.3)
    if labels:
        ax.legend()
    return ax


def plot_histogram(data, bins=30, titel="", xlabel="", ylabel="antal"):
    """Tegn et histogram (en fordeling) af nogle tal."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.asarray(data).ravel(), bins=bins, color="#4C72B0",
            edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titel)
    ax.grid(True, alpha=0.3)
    return ax


def plot_rumtid(historik, titel="Rum-tid-diagram", maks_raekker=1200,
                xlabel="felt (position)", ylabel="tid  →"):
    """Tegn et rum-tid-diagram. Sort = bil/optaget, hvid = tom.

    Til trafik: en trafikprop ses som et sort bånd, der driver baglæns."""
    import matplotlib.pyplot as plt
    if hasattr(historik, "rumtid"):
        historik = historik.rumtid
    H = np.asarray(historik)
    if H.shape[0] > maks_raekker:
        idx = np.linspace(0, H.shape[0] - 1, maks_raekker).astype(int)
        H = H[idx]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(H, aspect="auto", cmap="binary", interpolation="nearest",
              origin="upper")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titel)
    return ax


def plot_faseplan(x, y=None, titel="", xlabel="x", ylabel="y", pil=True):
    """Tegn to variable mod hinanden i stedet for mod tiden (et *faseplan*).

    Et faseplan viser systemets "bane": i stedet for at se hver variabel svinge op og
    ned over tid, ser man dem mod hinanden. En lukket sløjfe betyder, at systemet
    gentager sig selv igen og igen (fx rovdyr og byttedyr, eller en neuron, der fyrer).

    Du kan enten give to arrays (``plot_faseplan(x, y)``) eller bare resultatet fra
    ``simuler_differential`` med to variable (``plot_faseplan(x)``).
    """
    import matplotlib.pyplot as plt
    x = np.asarray(x)
    if y is None:
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(
                "plot_faseplan(x) uden y kræver, at x har form (antal_tider, 2) "
                "-- fx resultatet af simuler_differential med to variable. Du gav "
                "noget med form %s. Ellers kald plot_faseplan(x, y) med to arrays."
                % (x.shape,))
        x, y = x[:, 0], x[:, 1]
    else:
        y = np.asarray(y)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(x, y, lw=1.2, color="#4C72B0")
    if pil and len(x) > 24:
        skridt = len(x) // 12
        idx = np.arange(skridt, len(x) - 1, skridt)
        dx, dy = x[idx + 1] - x[idx], y[idx + 1] - y[idx]
        laengde = np.hypot(dx, dy)
        laengde[laengde == 0] = 1.0
        ax.quiver(x[idx], y[idx], dx / laengde, dy / laengde, angles="xy",
                  pivot="mid", color="#4C72B0", width=0.006, scale=25)
    ax.plot(x[0], y[0], "o", color="#55A868", ms=9, label="start")
    ax.plot(x[-1], y[-1], "s", color="#C44E52", ms=9, label="slut")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def vis_gitter(gitter, titel="", cmap="viridis"):
    """Vis et gitter som et øjebliksbillede (fx Ising-spins eller en skov)."""
    import matplotlib.pyplot as plt
    if hasattr(gitter, "gitter"):
        gitter = gitter.gitter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.asarray(gitter), cmap=cmap, interpolation="nearest")
    ax.set_title(titel)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def animer_gitter(historik, interval=120, cmap="binary", titel=""):
    """Lav en animation af en historik (fx TASEP eller skovbrand).

    I en notebook vises animationen automatisk. ``historik`` skal have form
    (antal_billeder, ...). For TASEP kan du også bruge res.rumtid."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    if hasattr(historik, "rumtid"):
        historik = historik.rumtid
    elif hasattr(historik, "historik"):
        historik = historik.historik
    H = np.asarray(historik)
    if H.ndim == 2:
        # gør hver tidsrække til en 1-pixel-høj "stribe", så det kan animeres
        H = H[:, None, :]
    fig, ax = plt.subplots(figsize=(5, 5))
    billede = ax.imshow(H[0], cmap=cmap, interpolation="nearest",
                        vmin=float(H.min()), vmax=float(H.max()))
    ax.set_title(titel)
    ax.set_xticks([])
    ax.set_yticks([])

    def opdater(k):
        billede.set_data(H[k])
        return (billede,)

    anim = animation.FuncAnimation(fig, opdater, frames=len(H),
                                   interval=interval, blit=True)
    plt.close(fig)
    try:
        from IPython.display import HTML
        return HTML(anim.to_jshtml())
    except Exception:
        return anim


def animer_partikler(positioner, status=None, retninger=None, L=None,
                     interval=80, maks_billeder=120, titel="", stoerrelse=18,
                     farver=("#4C72B0", "#C44E52", "#55A868")):
    """Animér prikker, der bevæger sig rundt (personer i en epidemi, fugle i en flok).

    Du kan bare give hele resultatet: ``animer_partikler(res)`` -- så finder funktionen
    selv positioner, status, retninger og kassens størrelse.

    ``status`` (0/1/2) giver prikkerne farve: blå = modtagelig, rød = smittet,
    grøn = immun. Er der ``retninger``, tegnes små pile i stedet for prikker.

    Der vises højst ``maks_billeder`` billeder. Det er med vilje: laver man for mange,
    smider matplotlib i stilhed slutningen af animationen væk.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if hasattr(positioner, "positioner"):
        resultat = positioner
        positioner = resultat.positioner
        if status is None:
            status = getattr(resultat, "status", None)
        if retninger is None:
            retninger = getattr(resultat, "retninger", None)
        if L is None:
            L = getattr(resultat, "L", None)

    P = np.asarray(positioner, dtype=float)
    if P.ndim != 3 or P.shape[2] != 2:
        raise ValueError(
            "positioner skal have form (antal_billeder, antal_partikler, 2). "
            "Du gav noget med form %s." % (P.shape,))
    S = None if status is None else np.asarray(status)
    R = None if retninger is None else np.asarray(retninger)

    if P.shape[0] > maks_billeder:
        idx = np.linspace(0, P.shape[0] - 1, maks_billeder).astype(int)
        P = P[idx]
        if S is not None:
            S = S[idx]
        if R is not None:
            R = R[idx]

    if L is None:
        L = float(P.max())
    palet = np.array(farver)
    hent_farver = (lambda n: palet[np.clip(S[n], 0, len(farver) - 1)]
                   if S is not None else farver[0])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=72)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titel)

    if R is not None:
        kunst = ax.quiver(P[0][:, 0], P[0][:, 1], np.cos(R[0]), np.sin(R[0]),
                          color=hent_farver(0), angles="xy", pivot="mid",
                          scale=30, width=0.005)

        def opdater(n):
            kunst.set_offsets(P[n])
            kunst.set_UVC(np.cos(R[n]), np.sin(R[n]))
            if S is not None:
                kunst.set_color(hent_farver(n))
            return (kunst,)
    else:
        kunst = ax.scatter(P[0][:, 0], P[0][:, 1], s=stoerrelse,
                           c=hent_farver(0))

        def opdater(n):
            kunst.set_offsets(P[n])
            if S is not None:
                kunst.set_color(hent_farver(n))
            return (kunst,)

    anim = animation.FuncAnimation(fig, opdater, frames=len(P),
                                   interval=interval, blit=True)
    plt.close(fig)
    try:
        from IPython.display import HTML
        return HTML(anim.to_jshtml())
    except Exception:
        return anim
