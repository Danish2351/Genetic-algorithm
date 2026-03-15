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

import numpy as np
import matplotlib.pyplot as plt
import csv

# ── SETTINGS ──────────────────────────────────────────────────────────────────
POP_SIZE     = 10     # number of individuals in population
N_GENS       = 40     # number of generations per run
N_RUNS       = 10     # number of independent runs to average over
MUT_PROB     = 0.5    # probability of mutating each gene
MUT_STEP     = 0.25   # mutation step size (plus/minus)
RANDOM_SEED  = 42     # for reproducibility

# Colors for each of the 6 combinations in combined plots
COMBO_COLORS = ["#D4550F", "#C428C4", "#8B8B00", "#2B5FC4", "#3DA63D", "#7B3DC4"]


# ==============================================================================
# 1. FITNESS FUNCTIONS  — we MAXIMIZE both
# ==============================================================================

def function1(x, y):
    """ f(x,y) = x^2 + y^2  |  range: -5 < x,y < 5  |  max at corners (~50) """
    return x**2 + y**2


def function2(x, y):
    """ Rosenbrock: f(x,y) = 100*(x^2 - y)^2 + (1-x)^2  |  range: -2<x<2, -1<y<3 """
    return 100 * (x**2 - y)**2 + (1 - x)**2


# ==============================================================================
# 2. POPULATION
# ==============================================================================

def init_population(pop_size, x_range, y_range):
    """ Initialize random (x, y) individuals within bounds. """
    x = np.random.uniform(x_range[0], x_range[1], pop_size)
    y = np.random.uniform(y_range[0], y_range[1], pop_size)
    return np.column_stack((x, y))


def evaluate(population, fitness_func):
    """ Compute fitness for all individuals. Higher = better (MAXIMIZE). """
    return np.array([fitness_func(ind[0], ind[1]) for ind in population])


# ==============================================================================
# 3. PARENT SELECTION  — higher fitness = more likely to be selected
# ==============================================================================

def fps(fitness, n):
    """
    Fitness Proportional Selection (Roulette Wheel).
    Each individual's selection chance is proportional to its fitness value.
    """
    shifted = fitness - fitness.min() + 1e-9    # shift so all values > 0
    probs   = shifted / shifted.sum()
    return np.random.choice(len(fitness), size=n, replace=True, p=probs)


def rbs(fitness, n):
    """
    Rank-Based Selection.
    Assign rank 1..N (N = best), select by rank probability.
    Avoids domination by very high fitness individuals (unlike FPS).
    """
    ranks = np.argsort(np.argsort(fitness)) + 1  # rank 1=worst, N=best
    probs = ranks / ranks.sum()
    return np.random.choice(len(fitness), size=n, replace=True, p=probs)


def binary_tournament(fitness, n):
    """
    Binary Tournament Selection.
    Pick 2 random individuals, higher fitness wins. Repeat n times.
    """
    winners = []
    for _ in range(n):
        i, j = np.random.choice(len(fitness), size=2, replace=False)
        winners.append(i if fitness[i] > fitness[j] else j)  # MAXIMIZE
    return np.array(winners)


# ==============================================================================
# 4. CROSSOVER + MUTATION
# ==============================================================================

def crossover(p1, p2):
    """ Arithmetic crossover: child = alpha*p1 + (1-alpha)*p2, alpha random in [0,1]. """
    a = np.random.uniform(0, 1)
    return a * p1 + (1 - a) * p2


def mutate(individual, x_range, y_range):
    """ Each gene mutated with MUT_PROB by +/- MUT_STEP, clamped to bounds. """
    child = individual.copy()
    if np.random.rand() < MUT_PROB:
        child[0] += np.random.uniform(-MUT_STEP, MUT_STEP)
        child[0]  = np.clip(child[0], x_range[0], x_range[1])
    if np.random.rand() < MUT_PROB:
        child[1] += np.random.uniform(-MUT_STEP, MUT_STEP)
        child[1]  = np.clip(child[1], y_range[0], y_range[1])
    return child


# ==============================================================================
# 5. SURVIVAL SELECTION  — pick best POP_SIZE from combined pool of 20
# ==============================================================================

def truncation_survival(pop, fit, n):
    """ Keep the n individuals with HIGHEST fitness (deterministic top-n). """
    best_idx = np.argsort(fit)[::-1][:n]   # descending sort, take top n
    return pop[best_idx], fit[best_idx]


def bt_survival(pop, fit, n):
    """ Binary tournament survival: run n tournaments on the combined pool. """
    idx = binary_tournament(fit, n)
    return pop[idx], fit[idx]


# ==============================================================================
# 6. ONE COMPLETE EA RUN (40 generations)
# ==============================================================================

def run_ea(fitness_func, x_range, y_range, parent_sel, survival_sel):
    """
    Runs the full EA for N_GENS generations.
    Returns:
        bsf_list : best-so-far fitness per generation  (never decreases)
        avg_list : average population fitness per generation
    """
    pop     = init_population(POP_SIZE, x_range, y_range)
    fitness = evaluate(pop, fitness_func)
    bsf     = fitness.max()
    bsf_list, avg_list = [], []

    for _ in range(N_GENS):

        # Step 1 — select parents
        p_idx = parent_sel(fitness, POP_SIZE)

        # Step 2 — create offspring via crossover + mutation
        offspring = []
        for i in range(0, POP_SIZE, 2):
            p1 = pop[p_idx[i]]
            p2 = pop[p_idx[(i + 1) % POP_SIZE]]
            offspring.append(mutate(crossover(p1, p2), x_range, y_range))
            offspring.append(mutate(crossover(p2, p1), x_range, y_range))
        offspring = np.array(offspring[:POP_SIZE])

        # Step 3 — combine parents + offspring (20 individuals total)
        combined_pop = np.vstack([pop, offspring])
        combined_fit = evaluate(combined_pop, fitness_func)

        # Step 4 — survival selection, keep best POP_SIZE
        pop, fitness = survival_sel(combined_pop, combined_fit, POP_SIZE)

        # Step 5 — record stats for this generation
        bsf = max(bsf, fitness.max())   # BSF never decreases
        bsf_list.append(bsf)
        avg_list.append(fitness.mean())

    return bsf_list, avg_list


# ==============================================================================
# 7. REPEAT N_RUNS TIMES AND AVERAGE
# ==============================================================================

def run_multiple(fitness_func, x_range, y_range, parent_sel, survival_sel):
    """
    Run the EA N_RUNS times (each with a fresh random population).
    Returns:
        avg_bsf : average best-so-far per generation across all runs
        avg_avg : average avg-fitness per generation across all runs
    """
    all_bsf, all_avg = [], []
    for _ in range(N_RUNS):
        bsf, avg = run_ea(fitness_func, x_range, y_range, parent_sel, survival_sel)
        all_bsf.append(bsf)
        all_avg.append(avg)
    all_bsf = np.array(all_bsf)  # shape: (N_RUNS, N_GENS)
    all_avg = np.array(all_avg)
    return all_bsf.mean(axis=0), all_avg.mean(axis=0)

# ==============================================================================
# 8. CSV GENERATION
# ==============================================================================


def save_csv(results, func_label):
    """
    Saves two CSV files per function:
      - one for avg best-so-far per generation
      - one for avg avg-fitness per generation
    """
    prefix = func_label.replace(" ", "_").replace(":", "").replace("/", "")

    # --- CSV 1: avg best-so-far ---
    with open(f"{prefix}_avg_best.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [r["label"] for r in results])
        for g in range(N_GENS):
            writer.writerow([g + 1] + [round(r["avg_bsf"][g], 4) for r in results])

    # --- CSV 2: avg average-fitness ---
    with open(f"{prefix}_avg_avg.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [r["label"] for r in results])
        for g in range(N_GENS):
            writer.writerow([g + 1] + [round(r["avg_avg"][g], 4) for r in results])

    print(f"  CSV saved: {prefix}_avg_best.csv")
    print(f"  CSV saved: {prefix}_avg_avg.csv")

# ==============================================================================
# 9. SELECTION LOOKUP + COMBO DEFINITIONS
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

# All 6 required combinations from the assignment
COMBINATIONS = [
    ("FPS", "Truncation"),
    ("RBS", "Truncation"),
    ("BT",  "Truncation"),
    ("FPS", "BT"),
    ("RBS", "BT"),
    ("BT",  "BT"),
]


# ==============================================================================
# 10. RUN ALL 6 COMBINATIONS FOR ONE FUNCTION
# ==============================================================================

def run_all(fitness_func, x_range, y_range, func_label):
    """
    Run all 6 combos and collect results.
    Returns list of dicts, one per combo.
    """
    results = []
    print(f"\n{'='*50}")
    print(f"  {func_label}")
    print(f"{'='*50}")

    for ps_name, ss_name in COMBINATIONS:
        label = f"{ps_name} + {ss_name}"
        print(f"  Running: {label} ...")
        avg_bsf, avg_avg = run_multiple(
            fitness_func, x_range, y_range,
            PARENT_SEL[ps_name],
            SURVIVAL_SEL[ss_name],
        )
        results.append({"label": label, "avg_bsf": avg_bsf, "avg_avg": avg_avg})

    print("  Done.\n")
    return results


# ==============================================================================
# 11. PLOTTING — live visualizations, nothing saved to disk
# ==============================================================================

def plot_single_combo(res, func_label, combo_num, total):
    """
    One separate window per combination.
    Shows avg best-so-far (blue) and avg avg-fitness (orange) on the same axes.
    The shaded region between the two lines shows the gap clearly.
    """
    gens = range(1, N_GENS + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"{func_label}  |  Combination {combo_num}/{total}\n"
        f"({N_RUNS} runs x {N_GENS} generations)",
        fontsize=12, fontweight="bold"
    )

    ax.plot(gens, res["avg_bsf"], color="#2B5FC4", linewidth=2, label="avg best")
    ax.plot(gens, res["avg_avg"], color="#E87722", linewidth=2, label="avg avg")
    # Shade the gap between best and average so the difference is visible
    ax.fill_between(gens, res["avg_avg"], res["avg_bsf"],
                    alpha=0.10, color="#2B5FC4")

    ax.set_title(res["label"], fontsize=12)
    ax.set_xlabel("generation", fontsize=10)
    ax.set_ylabel("fitness value", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, N_GENS)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


def plot_all_combos_separately(results, func_label):
    """
    Loop over all 6 results and open each one in its own separate window.
    Close one window to see the next.
    """
    for i, res in enumerate(results, start=1):
        plot_single_combo(res, func_label, combo_num=i, total=len(results))


def plot_combined(results, func_label):
    """
    One figure, two side-by-side subplots — all 6 combos on each.
    Left  : avg best-so-far for every combination
    Right : avg average-fitness for every combination
    Matches the 'all combinations' charts in the assignment template.
    """
    gens = range(1, N_GENS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{func_label} — All Combinations",
                 fontsize=13, fontweight="bold")

    for i, res in enumerate(results):
        color = COMBO_COLORS[i]
        ax1.plot(gens, res["avg_bsf"], color=color, linewidth=2, label=res["label"])
        ax2.plot(gens, res["avg_avg"], color=color, linewidth=2, label=res["label"])

    for ax, title in [
        (ax2, "Average fit of all combinations"),
        (ax1, "Best fit of all combinations"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("generation", fontsize=9)
        ax.set_ylabel("fitness values", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(1, N_GENS)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 12. MAIN
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # ── FUNCTION 1: f(x,y) = x^2 + y^2 ───────────────────────────────────
    results1 = run_all(
        fitness_func = function1,
        x_range      = (-5, 5),
        y_range      = (-5, 5),
        func_label   = "Function 1: f(x,y) = x^2 + y^2",
    )
    save_csv(results1, "Function 1 f(x,y) = x^2 + y^2")  
    plot_all_combos_separately(results1, "Function 1: f(x,y) = x^2 + y^2")  # 6 separate windows
    plot_combined(results1,             "Function 1: f(x,y) = x^2 + y^2")  # 1 combined window
    

    # ── FUNCTION 2: Rosenbrock ─────────────────────────────────────────────
    results2 = run_all(
        fitness_func = function2,
        x_range      = (-2, 2),
        y_range      = (-1, 3),
        func_label   = "Function 2: Rosenbrock",
    )
    save_csv(results2, "Function 2 Rosenbrock")   
    plot_all_combos_separately(results2, "Function 2: Rosenbrock")  # 6 separate windows
    plot_combined(results2,             "Function 2: Rosenbrock")  # 1 combined window
             