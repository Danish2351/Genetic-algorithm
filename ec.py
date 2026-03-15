#%%
"""
==============================================================================
  EVOLUTIONARY ALGORITHM - ASSIGNMENT 1
==============================================================================

  FUNCTIONS:
    Function 1: f(x,y) = x² + y²          range: -5 < x,y < 5   (MAXIMIZE)
    Function 2: f(x,y) = Rosenbrock        range: -2<x<2, -1<y<3 (MAXIMIZE)

  SELECTION COMBINATIONS (6 total):
    FPS + Truncation
    RBS + Truncation
    BinaryTournament + Truncation
    FPS + BinaryTournament
    RBS + BinaryTournament
    BinaryTournament + BinaryTournament

  OUTPUT PLOTS (per function):
    - 6 individual charts (one per combo): avg best + avg avg on same plot
    - 1 combined chart: avg best-so-far for all 6 combos
    - 1 combined chart: avg average-fitness for all 6 combos

==============================================================================
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# ── SETTINGS ──────────────────────────────────────────────────────────────────
POP_SIZE    = 10      # individuals per generation
N_GENS      = 40      # generations per run
N_RUNS      = 10      # independent runs per combination
MUT_PROB    = 0.5     # per-gene mutation probability
MUT_STEP    = 0.25    # mutation step ± this value
SEED        = 42      # reproducibility

# Colors for the 6 combinations in the final combined chart
COMBO_COLORS = ["#D4550F", "#C428C4", "#8B8B00", "#2B5FC4", "#3DA63D", "#7B3DC4"]

# Output folders (created automatically)
CSV_DIR  = "csv_output"
PLOT_DIR = "plot_output"
os.makedirs(CSV_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1 — FITNESS FUNCTIONS  (we MAXIMIZE both)
# ==============================================================================

def function1(x, y):
    """
    f(x, y) = x² + y²
    Range : -5 < x, y < 5
    Max   : at corners ≈ 50  (EA should push individuals toward ±5)
    """
    return x**2 + y**2


def function2(x, y):
    """
    Rosenbrock: f(x, y) = 100*(x² - y)² + (1 - x)²
    Range : -2 < x < 2,  -1 < y < 3
    We MAXIMIZE it — EA seeks high-value regions near the corners.
    """
    return 100 * (x**2 - y)**2 + (1 - x)**2


# ==============================================================================
# SECTION 2 — POPULATION
# ==============================================================================

def init_population(x_range, y_range):
    """Create POP_SIZE random [x, y] individuals within the given bounds."""
    x = np.random.uniform(x_range[0], x_range[1], POP_SIZE)
    y = np.random.uniform(y_range[0], y_range[1], POP_SIZE)
    return np.column_stack((x, y))   # shape: (POP_SIZE, 2)


def evaluate(population, fitness_func):
    """Return fitness array for every individual. Higher = better (MAXIMIZE)."""
    return np.array([fitness_func(ind[0], ind[1]) for ind in population])


# ==============================================================================
# SECTION 3 — PARENT SELECTION  (all favour higher fitness)
# ==============================================================================

def fps(fitness, n):
    """
    Fitness Proportional Selection (Roulette Wheel).
    Selection probability ∝ fitness value.
    Shift values so all are positive before normalising.
    """
    shifted = fitness - fitness.min() + 1e-9
    probs   = shifted / shifted.sum()
    return np.random.choice(len(fitness), size=n, replace=True, p=probs)


def rbs(fitness, n):
    """
    Rank-Based Selection.
    Sort by fitness, assign rank 1 (worst) … N (best), select by rank probability.
    Reduces domination by a single very-fit individual.
    """
    ranks = np.argsort(np.argsort(fitness)) + 1   # 1 = worst, N = best
    probs = ranks / ranks.sum()
    return np.random.choice(len(fitness), size=n, replace=True, p=probs)


def binary_tournament(fitness, n):
    """
    Binary Tournament Selection.
    Pick 2 random individuals; the one with HIGHER fitness wins.
    Repeat n times to obtain n parent indices.
    """
    winners = []
    for _ in range(n):
        i, j = np.random.choice(len(fitness), size=2, replace=False)
        winners.append(i if fitness[i] > fitness[j] else j)   # MAXIMIZE
    return np.array(winners)


# ==============================================================================
# SECTION 4 — CROSSOVER + MUTATION
# ==============================================================================

def crossover(p1, p2):
    """Arithmetic crossover: child = α·p1 + (1−α)·p2  where α ~ Uniform[0,1]."""
    a = np.random.uniform(0, 1)
    return a * p1 + (1 - a) * p2


def mutate(individual, x_range, y_range):
    """
    Each gene is mutated independently with probability MUT_PROB.
    Mutation adds a random value in [−MUT_STEP, +MUT_STEP].
    Result is clamped back inside the valid search bounds.
    """
    child = individual.copy()
    if np.random.rand() < MUT_PROB:
        child[0] += np.random.uniform(-MUT_STEP, MUT_STEP)
        child[0]  = np.clip(child[0], x_range[0], x_range[1])
    if np.random.rand() < MUT_PROB:
        child[1] += np.random.uniform(-MUT_STEP, MUT_STEP)
        child[1]  = np.clip(child[1], y_range[0], y_range[1])
    return child


# ==============================================================================
# SECTION 5 — SURVIVAL SELECTION
# ==============================================================================

def truncation_survival(pop, fit, n):
    """
    Truncation: keep the n individuals with the HIGHEST fitness.
    Deterministic — always picks the same top-n from the pool.
    """
    best_idx = np.argsort(fit)[::-1][:n]   # descending order, take first n
    return pop[best_idx], fit[best_idx]


def bt_survival(pop, fit, n):
    """
    Binary Tournament Survival: run n tournaments on the combined pool.
    Stochastic — high-fitness individuals are likely but not guaranteed to survive.
    """
    idx = binary_tournament(fit, n)
    return pop[idx], fit[idx]


# ==============================================================================
# SECTION 6 — ONE COMPLETE EA RUN  (returns per-generation stats)
# ==============================================================================

def run_ea(fitness_func, x_range, y_range, parent_sel, survival_sel):
    """
    Executes the EA for exactly N_GENS generations.

    Returns two lists of length N_GENS:
        bsf_per_gen : Best-So-Far fitness at each generation
                      (monotonically non-decreasing — it never gets worse)
        avg_per_gen : Mean fitness of the surviving population at each generation
    """
    pop     = init_population(x_range, y_range)
    fitness = evaluate(pop, fitness_func)
    bsf     = fitness.max()          # global best seen so far

    bsf_per_gen = []
    avg_per_gen = []

    for _ in range(N_GENS):

        # ── Step 1: parent selection ───────────────────────────────────────
        p_idx = parent_sel(fitness, POP_SIZE)

        # ── Step 2: crossover + mutation → POP_SIZE offspring ─────────────
        offspring = []
        for i in range(0, POP_SIZE, 2):
            p1 = pop[p_idx[i]]
            p2 = pop[p_idx[(i + 1) % POP_SIZE]]
            offspring.append(mutate(crossover(p1, p2), x_range, y_range))
            offspring.append(mutate(crossover(p2, p1), x_range, y_range))
        offspring = np.array(offspring[:POP_SIZE])

        # ── Step 3: combine parents + offspring  (20 individuals) ─────────
        combined_pop = np.vstack([pop, offspring])
        combined_fit = evaluate(combined_pop, fitness_func)

        # ── Step 4: survival selection → keep best POP_SIZE ───────────────
        pop, fitness = survival_sel(combined_pop, combined_fit, POP_SIZE)

        # ── Step 5: record generation stats ───────────────────────────────
        bsf = max(bsf, fitness.max())   # BSF never decreases
        bsf_per_gen.append(bsf)
        avg_per_gen.append(fitness.mean())

    return bsf_per_gen, avg_per_gen


# ==============================================================================
# SECTION 7 — 10 INDEPENDENT RUNS  (stores every run separately)
# ==============================================================================

def run_10_times(fitness_func, x_range, y_range, parent_sel, survival_sel):
    """
    Runs the EA N_RUNS (=10) times, each starting from a fresh random population.

    Returns:
        all_bsf : 2D array  shape (N_RUNS, N_GENS)
                  all_bsf[r][g] = Best-So-Far of run r at generation g
        all_avg : 2D array  shape (N_RUNS, N_GENS)
                  all_avg[r][g] = Avg fitness of run r at generation g
    """
    all_bsf, all_avg = [], []

    for run in range(N_RUNS):
        bsf_list, avg_list = run_ea(
            fitness_func, x_range, y_range, parent_sel, survival_sel
        )
        all_bsf.append(bsf_list)
        all_avg.append(avg_list)
        print(f"    Run {run + 1:2d}/10  done  |  "
              f"final BSF = {bsf_list[-1]:.2f}  |  "
              f"final avg = {avg_list[-1]:.2f}")

    return np.array(all_bsf), np.array(all_avg)   # shapes: (10, 40)


# ==============================================================================
# SECTION 8 — CSV EXPORT
# ==============================================================================

def save_csv(all_bsf, all_avg, combo_label, func_label):
    """
    Saves two CSV files for one combination — one for BSF, one for avg fitness.

    CSV format (matches template table exactly):
        Column 0 : Generation  (1, 2, … 40)
        Columns 1-10 : Run 1 … Run 10  (individual run values)
        Column 11 : Average  (mean across the 10 runs for that generation)

    Files are named: <func>_<combo>_best_fit.csv
                     <func>_<combo>_avg_fit.csv
    """
    # Build safe filename prefix
    safe_combo = combo_label.replace(" ", "_").replace("+", "").replace("__", "_")
    safe_func  = func_label.replace(" ", "_").replace(":", "").replace("/", "")
    prefix     = f"{CSV_DIR}/{safe_func}_{safe_combo}"

    avg_bsf = all_bsf.mean(axis=0)   # shape (N_GENS,)
    avg_avg = all_avg.mean(axis=0)

    headers = ["Generation"] + [f"Run {r+1}" for r in range(N_RUNS)] + ["Average"]

    # ── CSV 1: Best-So-Far ────────────────────────────────────────────────
    with open(f"{prefix}_best_fit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for g in range(N_GENS):
            row = [g + 1]
            row += [round(all_bsf[r][g], 4) for r in range(N_RUNS)]
            row += [round(avg_bsf[g], 4)]
            writer.writerow(row)

    # ── CSV 2: Average Fitness ────────────────────────────────────────────
    with open(f"{prefix}_avg_fit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for g in range(N_GENS):
            row = [g + 1]
            row += [round(all_avg[r][g], 4) for r in range(N_RUNS)]
            row += [round(avg_avg[g], 4)]
            writer.writerow(row)

    print(f"  CSV saved: {prefix}_best_fit.csv")
    print(f"  CSV saved: {prefix}_avg_fit.csv")

    return avg_bsf, avg_avg


# ==============================================================================
# SECTION 9 — SELECTION MAPS + COMBINATION LIST
# ==============================================================================

PARENT_SEL = {
    "FPS": fps,
    "RBS": rbs,
    "BT":  binary_tournament,
}
SURVIVAL_SEL = {
    "Truncation": truncation_survival,
    "BT":         bt_survival,
}

COMBINATIONS = [
    ("FPS", "Truncation"),
    ("RBS", "Truncation"),
    ("BT",  "Truncation"),
    ("FPS", "BT"),
    ("RBS", "BT"),
    ("BT",  "BT"),
]


# ==============================================================================
# SECTION 10 — PLOTTING
# ==============================================================================

def plot_single_combo(avg_bsf, avg_avg, combo_label, func_label, idx, total):
    """
    One separate window per combination.
    Blue  = Average Best-So-Far across 10 runs
    Orange = Average Avg-Fitness across 10 runs
    Shaded gap between the two lines shows how much the best exceeds the average.
    """
    gens = range(1, N_GENS + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"{func_label}  |  Combination {idx}/{total}",
        fontsize=12, fontweight="bold"
    )

    ax.plot(gens, avg_bsf, color="#2B5FC4", linewidth=2, label="average best")
    ax.plot(gens, avg_avg, color="#E87722", linewidth=2, label="average avg")
    ax.fill_between(gens, avg_avg, avg_bsf, alpha=0.10, color="#2B5FC4")

    ax.set_title(combo_label, fontsize=12)
    ax.set_xlabel("generations", fontsize=10)
    ax.set_ylabel("fitness values", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, N_GENS)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    # Save individual combo plot before displaying
    safe_combo = combo_label.replace(" ", "_").replace("+", "").replace("__", "_")
    safe_func  = func_label.replace(" ", "_").replace(":", "").replace("/", "")
    fname = f"{PLOT_DIR}/{safe_func}_{safe_combo}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {fname}")

    plt.show()


def plot_combined(all_results, func_label):
    """
    Two side-by-side combined charts (all 6 combinations on each).
    Left  : Average Best-So-Far for all 6 combos — matches template slide 8 / 15
    Right : Average Avg-Fitness  for all 6 combos
    """
    gens = range(1, N_GENS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{func_label} — All Combinations", fontsize=13, fontweight="bold")

    for i, res in enumerate(all_results):
        color = COMBO_COLORS[i]
        ax1.plot(gens, res["avg_bsf"], color=color, linewidth=2, label=res["label"])
        ax2.plot(gens, res["avg_avg"], color=color, linewidth=2, label=res["label"])

    for ax, title in [
        (ax1, "Best fit of all combinations"),
        (ax2, "Average fit of all combinations"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("generation", fontsize=9)
        ax.set_ylabel("fitness values", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(1, N_GENS)
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save combined chart before displaying
    safe_func = func_label.replace(" ", "_").replace(":", "").replace("/", "")
    fname = f"{PLOT_DIR}/{safe_func}_combined.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {fname}")

    plt.show()


# ==============================================================================
# SECTION 11 — MASTER RUNNER (one function, all 6 combinations)
# ==============================================================================

def run_function(fitness_func, x_range, y_range, func_label):
    """
    For a single fitness function:
      1. Loop over all 6 combinations
      2. Run 10 independent EA runs each
      3. Save two CSV files per combination
      4. Show individual plot per combination
      5. Collect results for the final combined chart
    """
    all_results = []   # will hold avg_bsf + avg_avg for every combo

    print(f"\n{'='*60}")
    print(f"  {func_label}")
    print(f"{'='*60}")

    for idx, (ps_name, ss_name) in enumerate(COMBINATIONS, start=1):
        combo_label = f"{ps_name} + {ss_name}"
        print(f"\n  [{idx}/6]  {combo_label}")

        # ── Run 10 times, get per-run per-generation data ──────────────────
        all_bsf, all_avg = run_10_times(
            fitness_func, x_range, y_range,
            PARENT_SEL[ps_name],
            SURVIVAL_SEL[ss_name],
        )

        # ── Save CSVs, also returns column-wise averages ───────────────────
        avg_bsf, avg_avg = save_csv(all_bsf, all_avg, combo_label, func_label)

        # ── Plot this combination in its own window ────────────────────────
        plot_single_combo(avg_bsf, avg_avg, combo_label, func_label,
                          idx, len(COMBINATIONS))

        # ── Store for combined chart later ────────────────────────────────
        all_results.append({
            "label":   combo_label,
            "avg_bsf": avg_bsf,
            "avg_avg": avg_avg,
        })

    # ── Final combined chart (all 6 on one figure) ────────────────────────
    plot_combined(all_results, func_label)


# ==============================================================================
# SECTION 12 — MAIN
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(SEED)

    # ── FUNCTION 1 ────────────────────────────────────────────────────────
    run_function(
        fitness_func = function1,
        x_range      = (-5, 5),
        y_range      = (-5, 5),
        func_label   = "Function 1  f(x,y) = x^2 + y^2",
    )

    # ── FUNCTION 2 ────────────────────────────────────────────────────────
    run_function(
        fitness_func = function2,
        x_range      = (-2, 2),
        y_range      = (-1, 3),
        func_label   = "Function 2  Rosenbrock",
    )

    print(f"\nAll CSV files saved in: {CSV_DIR}/")