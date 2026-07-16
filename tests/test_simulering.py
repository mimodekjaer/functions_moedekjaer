"""
test_simulering.py  —  selvtest af hele biblioteket.

Hver simulerings-funktion køres og tjekkes mod det, fysikken FORUDSIGER: den
tilfældige gang skal give <x^2> ~ N, Brownsk bevægelse <x^2> ~ 2Dt, Ising skal
være magnetiseret ved lav temperatur og ikke ved høj, flokimmunitet skal virke,
og så videre. Det er altså ikke bare "kører uden at kaste en exception".

Kør:  python tests/test_simulering.py
      SIMULERING_INGEN_NUMBA=1 python tests/test_simulering.py   # ren Python

Testen kræver, at pakken er installeret, fx med ``pip install -e .``.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")  # ingen skærm nødvendig

from unf_fysisk_simulering import *          # noqa: F401,F403


if __name__ == "__main__":
    print("numba aktiv:", HAR_NUMBA)

    # 1) differential: dx/dt = x  ->  e^t ; Euler skal afvige mere end RK4
    f = lambda t, x: x
    t, x_euler = simuler_differential(f, 1.0, 3.0, 0.5, metode="euler")
    _, x_rk4 = simuler_differential(f, 1.0, 3.0, 0.5, metode="rk4")
    sandt = np.exp(t[-1])
    fejl_euler = abs(x_euler[-1] - sandt)
    fejl_rk4 = abs(x_rk4[-1] - sandt)
    print("  differential  Euler-fejl=%.3f  RK4-fejl=%.5f" % (fejl_euler, fejl_rk4))
    assert fejl_rk4 < fejl_euler, "RK4 burde være mere præcis end Euler"

    # system: harmonisk svingning, RK45
    g = lambda t, v: np.array([v[1], -v[0]])
    tt, xx = simuler_differential(g, [1.0, 0.0], 6.28, 0.1, metode="rk45")
    assert xx.shape[1] == 2

    # 2) tilfældig gang: <x^2> ~ N
    N = 200
    baner = simuler_tilfaeldig_gang(N, 4000, dim=1, seed=1)
    msd = np.mean(baner[:, -1] ** 2)
    print("  tilfældig gang  <x^2>=%.1f (forventet ~%d)" % (msd, N))
    assert 0.6 * N < msd < 1.4 * N

    # 3) Brownsk: <x^2> ~ 2 D t
    D, T = 0.5, 4.0
    tb, X = simuler_brownsk(D, 0.0, T, 0.01, antal_baner=4000, seed=2)
    msd_b = np.mean(X[:, -1] ** 2)
    print("  brownsk  <x^2>=%.2f (forventet ~%.2f)" % (msd_b, 2 * D * T))
    assert 0.6 * (2 * D * T) < msd_b < 1.4 * (2 * D * T)

    # 4) Monte Carlo: estimér pi
    def pi_forsoeg():
        x, y = np.random.random(), np.random.random()
        return 4.0 if x * x + y * y < 1.0 else 0.0
    pi = simuler_monte_carlo(pi_forsoeg, 200000, seed=3).mean()
    print("  monte carlo  pi=%.3f" % pi)
    assert abs(pi - np.pi) < 0.1

    # 5) Poisson: antal ~ rate*t_slut, begge metoder
    r, tmax = 3.0, 50.0
    n_h = len(simuler_poisson(r, tmax, metode="haendelse", seed=4))
    n_t = len(simuler_poisson(r, tmax, dt=0.01, metode="tid", seed=5))
    print("  poisson  hændelse=%d  tid=%d  (forventet ~%.0f)" % (n_h, n_t, r * tmax))
    assert 0.6 * r * tmax < n_h < 1.4 * r * tmax
    assert 0.6 * r * tmax < n_t < 1.4 * r * tmax
    try:
        simuler_poisson(200.0, 1.0, dt=0.01, metode="tid")
        raise AssertionError("burde have klaget over rate*dt >= 1")
    except ValueError:
        pass

    # 6) Gillespie: rent henfald A -> 0, ende ~ 0
    start = [1000.0]
    reaktioner = [(lambda s: 1.0 * s[0], np.array([-1.0]))]
    tg, sg = simuler_gillespie(start, reaktioner, 10.0, seed=6)
    print("  gillespie  A: %d -> %d" % (start[0], sg[-1, 0]))
    assert sg[-1, 0] < 100  # næsten alt henfaldet

    # 7) TASEP: lav-tæthed-fase har lav tæthed
    res = simuler_tasep(L=50, alpha=0.2, beta=0.8, t_slut=200.0, dt=0.1, seed=7)
    print("  tasep  middel_tæthed=%.3f  strøm=%.3f" % (res.middel_taethed, res.stroem))
    assert res.rumtid.shape[1] == 50
    assert 0.0 <= res.middel_taethed <= 1.0

    # 7b) Trafik med egen hastighedsmodel
    def fart_model(fart, afstand):
        return fart + 1            # langsom acceleration (capped af afstand)
    rt = simuler_trafik(fart_model, L=60, antal_biler=20, antal_skridt=200, seed=12)
    print("  trafik  middel_fart=%.2f  strøm=%.2f" % (rt.middel_fart, rt.stroem))
    assert rt.rumtid.shape == (201, 60)
    assert rt.middel_fart > 0

    # 8) Ising: lav T -> stærkt magnetiseret, høj T -> svag
    lav = simuler_ising(16, 1.0, 120, seed=8).middel_magnetisering
    hoej = simuler_ising(16, 6.0, 120, seed=9).middel_magnetisering
    print("  ising  |M|(T=1)=%.2f  |M|(T=6)=%.2f" % (lav, hoej))
    assert lav > hoej

    # 9) Skovbrand: kører og giver rigtig form
    h = simuler_skovbrand(30, antal_skridt=40, seed=10)
    assert h.shape == (41, 30, 30)

    # 10) Epidemi på gitter: flokimmunitet virker, og ingen personer forsvinder
    uden = simuler_epidemi_gitter(60, p_smitte=0.4, p_rask=0.1, p_vaccineret=0.0,
                                  antal_smittede_start=3, antal_skridt=300, seed=20)
    med = simuler_epidemi_gitter(60, p_smitte=0.4, p_rask=0.1, p_vaccineret=0.7,
                                 antal_smittede_start=3, antal_skridt=300, seed=21)
    ramt_uden = (uden.R[-1] - uden.R[0]) / max(uden.S[0], 1)
    ramt_med = (med.R[-1] - med.R[0]) / max(med.S[0], 1)
    print("  epidemi-gitter  ramt: uden vaccine=%.2f  med 70%% vaccineret=%.2f"
          % (ramt_uden, ramt_med))
    assert uden.historik.shape == (301, 60, 60)
    assert np.all(uden.antal.sum(axis=1) == 60 * 60)   # ingen forsvinder
    assert ramt_med < 0.3 * ramt_uden                  # flokimmunitet

    # 11) Epidemi med agenter: at bevæge sig mindre flader kurven ud
    lav = simuler_epidemi_agenter(antal_personer=300, fart=0.05,
                                  antal_skridt=400, seed=22)
    hoej = simuler_epidemi_agenter(antal_personer=300, fart=1.0,
                                   antal_skridt=400, seed=23)
    print("  epidemi-agenter  maks samtidigt syge: fart=0.05 -> %d,  fart=1.0 -> %d"
          % (lav.I.max(), hoej.I.max()))
    assert np.all(lav.antal.sum(axis=1) == 300)
    assert hoej.I.max() > lav.I.max()
    assert lav.positioner.min() >= 0.0 and lav.positioner.max() <= 20.0

    # 12) Wright-Fisher: P(fiksering) = startfrekvensen; selektion hjælper
    wf = simuler_wright_fisher(50, 0.3, generationer=600,
                               antal_populationer=1000, seed=24)
    wfs = simuler_wright_fisher(50, 0.3, generationer=600,
                                antal_populationer=1000, s=0.1, seed=25)
    print("  wright-fisher  fikseret=%.3f (forventet 0.30)  med s=0.1: %.3f"
          % (wf.andel_fikseret, wfs.andel_fikseret))
    assert wf.frekvenser.shape == (1000, 601)
    assert abs(wf.andel_fikseret - 0.3) < 0.06
    assert wfs.andel_fikseret > wf.andel_fikseret

    # 13) Neuron: tavs under tærsklen -- men støj alene kan få den til at fyre
    stille = simuler_neuron(stroem=0.8, stoej=0.0, t_slut=1000.0, seed=26)
    larm = simuler_neuron(stroem=0.8, stoej=0.6, t_slut=1000.0, seed=27)
    staerk = simuler_neuron(stroem=1.5, stoej=0.0, t_slut=1000.0, seed=28)
    print("  neuron  tavs=%d spikes,  med støj=%d spikes,  stærk strøm=%.0f Hz"
          % (len(stille.spike_tider), len(larm.spike_tider), staerk.fyringsrate))
    assert len(stille.spike_tider) == 0
    assert len(larm.spike_tider) > 0
    assert staerk.fyringsrate > 0

    # 14) Reaktion-diffusion: striber dækker mere end pletter; ustabilt dt fanges
    pletter = simuler_reaktion_diffusion(L=64, F=0.035, k=0.065,
                                         antal_skridt=4000, gem_hvert=200, seed=29)
    striber = simuler_reaktion_diffusion(L=64, F=0.035, k=0.060,
                                         antal_skridt=4000, gem_hvert=200, seed=29)
    daek_p = float((pletter.V > 0.2).mean())
    daek_s = float((striber.V > 0.2).mean())
    print("  reaktion-diffusion  dækning: pletter=%.2f  striber=%.2f"
          % (daek_p, daek_s))
    assert np.all(np.isfinite(pletter.V))        # ikke eksploderet
    assert daek_s > daek_p
    try:
        simuler_reaktion_diffusion(L=32, Du=0.3, dt=1.0, antal_skridt=10)
        raise AssertionError("burde have klaget over max(Du, Dv)*dt > 0.25")
    except ValueError:
        pass

    # 15) Flok: lav støj -> ordnet flok, høj støj -> kaos (~1/sqrt(N))
    rolig = simuler_flok(antal_fugle=200, stoej=0.2, antal_skridt=300, seed=30)
    kaos = simuler_flok(antal_fugle=200, stoej=6.0, antal_skridt=300, seed=31)
    print("  flok  orden(støj=0.2)=%.2f  orden(støj=6.0)=%.2f  (1/sqrt(N)=%.2f)"
          % (rolig.middel_orden, kaos.middel_orden, 1 / np.sqrt(200)))
    assert rolig.middel_orden > 0.8
    assert kaos.middel_orden < 0.3

    # 16) plot-hjælpere kører uden fejl
    plot_baner(t, x_rk4, titel="test")
    plot_histogram(baner[:, -1], titel="test")
    plot_rumtid(res, titel="test")
    vis_gitter(simuler_ising(16, 1.0, 50, seed=11), titel="test")
    plot_baner(uden.tid, uden.antal, labels=["S", "I", "R"], titel="test")
    plot_faseplan(xx, titel="test")
    vis_gitter(pletter, cmap="inferno", titel="test")
    animer_gitter(pletter, cmap="inferno", titel="test")
    animer_partikler(lav, titel="test")
    animer_partikler(rolig, titel="test")

    print("ALLE SELVTESTS BESTÅET.")
