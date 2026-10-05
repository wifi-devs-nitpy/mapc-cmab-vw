"""
Epsilon hyper-parameter tuning for the hierarchical VW contextual-bandit MAPC agent.

Place this file next to ``vw_mixed_scenario.py`` (it re-uses MixScen and
run_single_experiment from there, so the evaluation protocol is identical).

Sweep modes
-----------
oat      One-level-at-a-time (default). Starting from --baseline, sweep the epsilon of
         one level over its grid while the other three stay at the baseline.
uniform  The same epsilon for all four levels, over --uniform-grid.
grid     Full cartesian product of the four per-level grids (can be very large!).

Fair comparison
---------------
Every configuration uses the SAME seeds / JAX keys for run i (common random numbers),
so differences between configurations are not caused by different random streams.

Outputs (in --output-dir)
-------------------------
cache/eps_*.npz         raw throughputs (n_runs, n_steps); reused on re-run (use --force to redo)
summary.csv             one row per configuration with all metrics
plots/*.png             throughput curves + metrics-vs-epsilon plots
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np
from scipy import stats
from tqdm import tqdm

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mapc_cmab_vw.agents.mapc_vw_cb_agent_factory import MapcVWAgentFactory  # noqa: E402
from mapc_cmab_vw.envs.scenario_impl import small_office_scenario  # noqa: E402
from vw_mixed_scenario import MixScen, run_single_experiment  # noqa: E402

LEVEL_NAMES = ["Level 1 (AP group)", "Level 2 (station)", "Level 3 (links)", "Level 4 (tx power)"]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Epsilon sweep for the hierarchical VW agent.")

    p.add_argument("--mode", choices=["oat", "uniform", "grid"], default="oat")
    p.add_argument("--baseline", type=float, nargs=4, default=[0.3, 0.3, 0.1, 0.05],
                   metavar=("E1", "E2", "E3", "E4"),
                   help="Baseline epsilon per level (used by oat mode).")
    p.add_argument("--grid-lvl1", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--grid-lvl2", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--grid-lvl3", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--grid-lvl4", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    p.add_argument("--uniform-grid", type=float, nargs="+",
                   default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--top-k", type=int, default=8,
                   help="grid mode: number of best configurations drawn in the curve plot.")

    p.add_argument("--d-ap", type=float, default=10.0)
    p.add_argument("--d-sta-1", type=int, default=2)
    p.add_argument("--d-sta-2", type=int, default=4)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=10_000)
    p.add_argument("--n-links", type=int, default=3)
    p.add_argument("--n-tx-power-levels", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--window-size", type=int, default=100, help="moving-average window")
    p.add_argument("--confidence", type=float, default=0.99)
    p.add_argument("--output-dir", type=str, default="./tuning_results")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
def build_configs(args):
    """Returns a list of unique 4-tuples (eps_lvl1..eps_lvl4)."""
    base = tuple(args.baseline)
    grids = [args.grid_lvl1, args.grid_lvl2, args.grid_lvl3, args.grid_lvl4]

    if args.mode == "oat":
        cfgs = [base]
        for lvl, grid in enumerate(grids):
            for v in grid:
                cfg = list(base)
                cfg[lvl] = v
                cfgs.append(tuple(cfg))
    elif args.mode == "uniform":
        cfgs = [(e, e, e, e) for e in args.uniform_grid]
    else:
        cfgs = list(itertools.product(*grids))

    return list(dict.fromkeys(cfgs))  # unique, order preserved


def cfg_stem(cfg):
    return "eps_" + "_".join(f"{e:g}" for e in cfg)


def cfg_label(cfg):
    return "ε=(" + ", ".join(f"{e:g}" for e in cfg) + ")"


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
def run_config(cfg, args, scenario, cache_dir: Path):
    """Runs (or loads from cache) n_runs independent experiments for one configuration."""
    path = cache_dir / f"{cfg_stem(cfg)}.npz"
    if path.exists() and not args.force:
        data = np.load(path)
        if data["throughputs"].shape == (args.n_runs, args.n_steps):
            print(f"  cached: {path.name}")
            return data["throughputs"]
        print(f"  cache shape mismatch for {path.name}, recomputing")

    throughputs = np.zeros((args.n_runs, args.n_steps), dtype=np.float32)
    # identical keys / seeds for every configuration (common random numbers)
    run_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_runs)

    for run in range(args.n_runs):
        factory = MapcVWAgentFactory(
            associations=scenario.associations,
            agent_params_lvl1={"epsilon": cfg[0]},
            agent_params_lvl2={"epsilon": cfg[1]},
            agent_params_lvl3={"epsilon": cfg[2]},
            agent_params_lvl4={"epsilon": cfg[3]},
            n_tx_power_levels=args.n_tx_power_levels,
            n_links=args.n_links,
            seed=args.seed + run,
        )
        throughputs[run] = run_single_experiment(
            agent_factory=factory,
            scenario=scenario,
            run_number=run,
            n_steps=args.n_steps,
            key=run_keys[run],
        )

    np.savez(path, throughputs=throughputs, cfg=np.asarray(cfg))
    return throughputs


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average over the last axis ('valid' mode): output length T - w + 1."""
    c = np.cumsum(x, axis=-1, dtype=np.float64)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), c], axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def mean_ci(x: np.ndarray, confidence: float):
    """Mean and confidence interval over axis 0 (runs)."""
    n = x.shape[0]
    m = x.mean(axis=0)
    if n < 2:
        return m, m, m
    se = x.std(axis=0, ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h


def phase_metrics(tp: np.ndarray, w: int) -> dict:
    """
    Metrics of one stationary phase. tp: (n_runs, T_phase).
    Smoothing is done inside the phase so the scenario switch does not leak into it.
    """
    sm = moving_average(tp, w)                 # (runs, T')
    curve = sm.mean(axis=0)                    # mean over runs
    tail = max(1, int(0.2 * sm.shape[1]))      # last 20 % of the phase = "plateau"
    plateau = curve[-tail:].mean()

    idx = np.nonzero(curve >= 0.9 * plateau)[0]
    t90_own = float(idx[0] + w - 1) if idx.size else np.nan

    stab_std = sm[:, -tail:].std(axis=1).mean()     # fluctuation inside plateau
    run_plateaus = sm[:, -tail:].mean(axis=1)

    return {
        "mean": float(tp.mean()),
        "plateau": float(plateau),
        "t90_own": t90_own,
        "stab_std": float(stab_std),
        "cv": float(stab_std / plateau) if plateau > 0 else np.nan,
        "run_std": float(run_plateaus.std(ddof=1)) if tp.shape[0] > 1 else 0.0,
        "curve": curve,
    }


def compute_metrics(results: dict, args) -> dict:
    switch = args.n_steps // 2
    w = args.window_size
    metrics = {}

    for cfg, tp in results.items():
        p1 = phase_metrics(tp[:, 1:switch], w)   # step 0 is never filled by the evaluator
        p2 = phase_metrics(tp[:, switch:], w)
        per_run_mean = tp[:, 1:].mean(axis=1)
        _, lo, hi = mean_ci(per_run_mean[:, None], args.confidence)
        metrics[cfg] = {
            "p1": p1,
            "p2": p2,
            "overall": float(per_run_mean.mean()),
            "overall_ci": float((hi - lo)[0] / 2),
        }

    # learning speed relative to the BEST plateau among all configurations (comparable across cfgs)
    for ph in ("p1", "p2"):
        ref = max(m[ph]["plateau"] for m in metrics.values())
        for m in metrics.values():
            idx = np.nonzero(m[ph]["curve"] >= 0.9 * ref)[0]
            m[ph]["t90_ref"] = float(idx[0] + w - 1) if idx.size else np.nan

    return metrics


def write_csv(metrics: dict, path: Path):
    cols = ["eps_lvl1", "eps_lvl2", "eps_lvl3", "eps_lvl4", "overall_mean", "overall_ci"]
    for ph in ("p1", "p2"):
        cols += [f"{ph}_{k}" for k in ("mean", "plateau", "t90_ref", "t90_own", "stab_std", "cv", "run_std")]

    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for cfg, m in metrics.items():
            row = list(cfg) + [m["overall"], m["overall_ci"]]
            for ph in ("p1", "p2"):
                row += [m[ph][k] for k in ("mean", "plateau", "t90_ref", "t90_own", "stab_std", "cv", "run_std")]
            wr.writerow(row)


def print_ranking(metrics: dict, top: int = 10):
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["overall"], reverse=True)[:top]
    print("\nTop configurations by mean throughput over the whole run")
    print(f"{'epsilon (L1,L2,L3,L4)':<28}{'overall':>10}{'plat P1':>10}{'plat P2':>10}"
          f"{'t90 P1':>9}{'t90 P2':>9}{'cv P1':>8}{'cv P2':>8}")
    for cfg, m in ranked:
        print(f"{str(cfg):<28}{m['overall']:>10.2f}{m['p1']['plateau']:>10.2f}{m['p2']['plateau']:>10.2f}"
              f"{m['p1']['t90_ref']:>9.0f}{m['p2']['t90_ref']:>9.0f}"
              f"{m['p1']['cv']:>8.3f}{m['p2']['cv']:>8.3f}")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def draw_curves(ax, results, cfgs, labels, args, title):
    w = args.window_size
    switch = args.n_steps // 2
    cmap = plt.get_cmap("viridis", max(len(cfgs), 2))

    for i, (cfg, label) in enumerate(zip(cfgs, labels)):
        tp = results[cfg][:, 1:]                      # step s is at column s-1
        sm = moving_average(tp, w)
        m, lo, hi = mean_ci(sm, args.confidence)
        x = np.arange(w, tp.shape[1] + 1)
        ax.plot(x, m, color=cmap(i), lw=1.6, label=label)
        ax.fill_between(x, lo, hi, color=cmap(i), alpha=0.15)

    ax.axvline(switch, color="k", ls="--", alpha=0.6)
    ax.text(switch, ax.get_ylim()[1], f" scenario switch\n d_sta {args.d_sta_1}→{args.d_sta_2}",
            va="top", ha="left", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("simulation step")
    ax.set_ylabel(f"throughput (moving avg, window={w})")
    ax.grid(alpha=0.3)


def subtitle(args):
    return (f"mean of {args.n_runs} runs, {int(args.confidence * 100)}% CI, "
            f"small office d_ap={args.d_ap:g}, {args.n_steps} steps")


def plot_oat(results, metrics, args, plot_dir: Path):
    base = tuple(args.baseline)
    fig_all, axes_all = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    axes_all = axes_all.ravel()

    for lvl in range(4):
        cfgs = [c for c in results if all(c[j] == base[j] for j in range(4) if j != lvl)]
        cfgs.sort(key=lambda c: c[lvl])
        labels = [f"ε{lvl + 1}={c[lvl]:g}" + (" (baseline)" if c == base else "") for c in cfgs]
        others = ", ".join(f"L{j + 1}={base[j]:g}" for j in range(4) if j != lvl)
        title = f"{LEVEL_NAMES[lvl]} epsilon sweep  |  fixed: {others}\n{subtitle(args)}"

        # 1) curves, one figure per level
        fig, ax = plt.subplots(figsize=(11, 6))
        draw_curves(ax, results, cfgs, labels, args, title)
        ax.legend(title=f"ε of {LEVEL_NAMES[lvl]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"curves_level{lvl + 1}.png", dpi=150)
        if not args.show:
            plt.close(fig)

        # 2) same curve in the overview figure
        draw_curves(axes_all[lvl], results, cfgs, labels, args, title)
        axes_all[lvl].legend(fontsize=7)

        # 3) metrics vs epsilon
        plot_metrics_vs_eps(lvl, cfgs, metrics, others, args, plot_dir)

    fig_all.suptitle("Epsilon sweeps, one level at a time", fontsize=13)
    fig_all.tight_layout()
    fig_all.savefig(plot_dir / "curves_overview_all_levels.png", dpi=130)
    if not args.show:
        plt.close(fig_all)


def plot_metrics_vs_eps(lvl, cfgs, metrics, others, args, plot_dir: Path):
    x = np.arange(len(cfgs))
    xt = [f"{c[lvl]:g}" for c in cfgs]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    def two_phase(ax, key, title, ylabel):
        for ph, name, mk in (("p1", f"phase 1 (d_sta={args.d_sta_1})", "o"),
                             ("p2", f"phase 2 (d_sta={args.d_sta_2})", "s")):
            ax.plot(x, [metrics[c][ph][key] for c in cfgs], marker=mk, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)

    two_phase(axes[0, 0], "plateau", "Final throughput of each phase (last 20% of phase)\nhigher = better", "throughput")
    two_phase(axes[0, 1], "t90_ref", "Learning speed: steps to reach 90% of the best plateau\nlower = faster (gap = never reached)", "steps since phase start")
    two_phase(axes[1, 0], "cv", "Stability: std/mean of smoothed throughput in plateau\nlower = more stable", "coefficient of variation")

    ax = axes[1, 1]
    ax.errorbar(x, [metrics[c]["overall"] for c in cfgs],
                yerr=[metrics[c]["overall_ci"] for c in cfgs], marker="o", capsize=4)
    ax.set_title(f"Mean throughput over the whole run (±{int(args.confidence * 100)}% CI over runs)\nhigher = better", fontsize=10)
    ax.set_ylabel("throughput")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(xt)
        ax.set_xlabel(f"epsilon of {LEVEL_NAMES[lvl]}")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    fig.suptitle(f"{LEVEL_NAMES[lvl]}: metrics vs epsilon  |  fixed: {others}\n{subtitle(args)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / f"metrics_level{lvl + 1}.png", dpi=150)
    if not args.show:
        plt.close(fig)


def plot_generic(results, metrics, args, plot_dir: Path):
    """uniform / grid modes: curves of (top-k) configurations."""
    cfgs = sorted(results, key=lambda c: metrics[c]["overall"], reverse=True)
    if args.mode == "grid":
        cfgs = cfgs[:args.top_k]
        what = f"top {len(cfgs)} of {len(results)} configurations (by mean throughput)"
    else:
        cfgs = sorted(cfgs, key=lambda c: c[0])
        what = "same epsilon on all levels"

    labels = [f"ε={c[0]:g}" if args.mode == "uniform" else cfg_label(c) for c in cfgs]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    draw_curves(ax, results, cfgs, labels, args, f"Epsilon tuning ({what})\n{subtitle(args)}")
    ax.legend(fontsize=8, title="epsilon (L1, L2, L3, L4)" if args.mode == "grid" else "epsilon (all levels)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"curves_{args.mode}.png", dpi=150)
    if not args.show:
        plt.close(fig)

    if args.mode == "uniform":
        plot_metrics_vs_eps(0, cfgs, metrics, "all levels share ε", args, plot_dir)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.output_dir)
    cache_dir, plot_dir = out / "cache", out / "plots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    scenario = MixScen(small_office_scenario, args.d_sta_1, args.d_sta_2, args.d_ap,
                       max_steps=args.n_steps)

    cfgs = build_configs(args)
    print("-" * 60)
    print(f"mode={args.mode}  configurations={len(cfgs)}  runs/cfg={args.n_runs}  steps={args.n_steps}")
    print(f"total simulated runs: {len(cfgs) * args.n_runs}")
    print("-" * 60)

    results = {}
    for i, cfg in enumerate(tqdm(cfgs, desc="configurations"), start=1):
        print(f"\n[{i}/{len(cfgs)}] {cfg_label(cfg)}")
        results[cfg] = run_config(cfg, args, scenario, cache_dir)

    metrics = compute_metrics(results, args)
    write_csv(metrics, out / "summary.csv")
    print_ranking(metrics)

    if args.mode == "oat":
        plot_oat(results, metrics, args, plot_dir)
    else:
        plot_generic(results, metrics, args, plot_dir)

    print(f"\nSaved: {out / 'summary.csv'}\n       {plot_dir}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
