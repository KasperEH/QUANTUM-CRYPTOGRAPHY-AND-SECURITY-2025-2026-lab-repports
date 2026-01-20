import error_correction_lib as ec
import numpy as np
import matplotlib.pyplot as plt
from file_utils import codes_from_file

# --- Configuration ---
CODES_FILE = 'codes_1944.txt'
N_BLOCK = 1944
CODES = codes_from_file(CODES_FILE)
R_RANGE = sorted([code[0] for code in CODES])
N_TRIES = 5 # Number of tries per point (Keep low for speed, raise for smoothness)

def calc_f_prime(f, qber):
    if qber <= 0 or qber >= 0.5: return 0.0
    return (f - 1) * ec.h_b(qber)

def run_manual_test(qber, R, s_n, p_n, n_tries=N_TRIES):
    """
    Runs EC with explicitly defined R, s_n, p_n (Bypassing choose_sp).
    """
    code_params = CODES[(R, N_BLOCK)]
    m = (1-R)*N_BLOCK
    discl_n = int(round(N_BLOCK*(0.0280-0.02*R)))
    
    f_list = []
    
    for _ in range(n_tries):
        x = ec.generate_key(N_BLOCK - s_n - p_n)
        # Using Precise errors for cleaner plots
        y = ec.add_errors_prec(x, qber) 
        
        add_info, com_iters, x_dec, ver_check, _ = ec.perform_ec(
            x, y, 
            code_params['s_y_joins'], code_params['y_s_joins'], 
            qber, s_n, p_n, punct_list=code_params['punct_list'], 
            discl_n=discl_n, show=0
        )
        
        if ver_check:
            # Only count efficiency if it succeeded (or penalty applied)
            f_cur = float(m - p_n + add_info) / (N_BLOCK - p_n - s_n) / ec.h_b(qber)
            f_list.append(f_cur)
    
    if len(f_list) == 0: return None # All failed
    return np.mean(f_list)

# =============================================================================
# TASK 1: Varying Shortening and Puncturing
# =============================================================================
def task_1_vary_sp():
    print("\n--- Task 1: Varying Shortening and Puncturing ---")
    fixed_qber = 0.05
    fixed_R = 0.6667 # Choosing a middle rate
    
    # 1A: Fix Puncturing=0, Vary Shortening
    s_n_values = np.arange(0, 800, 100)
    eff_s = []
    effec_s = []
    print(f"Running Shortening sweep (QBER={fixed_qber}, R={fixed_R}, p_n=0)...")
    for s_n in s_n_values:
        print(f"  Testing s_n={s_n}...")
        f = run_manual_test(fixed_qber, fixed_R, s_n, 0)
        eff_s.append(f)
        effec_s.append(calc_f_prime(f, fixed_qber))
        
        
    # 1B: Fix Shortening=0, Vary Puncturing
    p_n_values = np.arange(0, 800, 100)
    eff_p = []
    effec_p = []
    print(f"Running Puncturing sweep (QBER={fixed_qber}, R={fixed_R}, s_n=0)...")
    for p_n in p_n_values:
        print(f"  Testing p_n={p_n}...")
        f = run_manual_test(fixed_qber, fixed_R, 0, p_n)
        eff_p.append(f)
        effec_p.append(calc_f_prime(f, fixed_qber))

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Create the twin X axis first
    ax2 = ax1.twiny() 
    # 2. Create the twin Y axis last
    ax3 = ax1.twinx() 

    # --- PRIMARY Y-AXIS: Efficiency (f) ---
    line1, = ax1.plot(s_n_values, eff_s, 'o-', label='Shortening ($f$)', color='tab:blue', zorder=3)
    line2, = ax2.plot(p_n_values, eff_p, 's-', label='Puncturing ($f$)', color='orange', zorder=3)
    
    ax1.set_ylabel('Efficiency ($f$)', fontsize=12)
    ax1.set_xlabel('Shortened Bits ($s_n$)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='x', labelcolor='tab:blue')
    ax1.set_ylim(1.0, 1.8)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- SECONDARY Y-AXIS: Effectiveness (f') ---
    # Plotting f' lines with markers and thicker dots to ensure visibility
    line1_prime, = ax3.plot(s_n_values, effec_s, 'o:', markersize=4, label='Shortening ($f\'$)', color='tab:blue', alpha=0.8, zorder=4)
    line2_prime, = ax3.plot(p_n_values, effec_p, 's:', markersize=4, label='Puncturing ($f\'$)', color='orange', alpha=0.8, zorder=4)
    
    ax3.set_ylabel('Effectiveness ($f\'$)', fontsize=12)

    # --- TOP X-AXIS ---
    ax2.set_xlabel('Punctured Bits ($p_n$)', color='orange', fontsize=12)
    ax2.tick_params(axis='x', labelcolor='orange')

    # Combined Title
    plt.title(f'Efficiency ($f$) and Effectiveness ($f\'$): Shortening vs. Puncturing\n(QBER={fixed_qber}, Base R={fixed_R:.2f})', pad=35)

    # Unified Legend
    lines = [line1, line1_prime, line2, line2_prime]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize='small', ncol=2, frameon=True).set_zorder(5)
    
    plt.tight_layout()
    plt.savefig('task1_sp_variation_fixed.png')

# =============================================================================
# TASK 2: Varying Base Rates
# =============================================================================
def task_2_vary_base_rate():
    print("\n--- Task 2: Varying Base Rate (No Shortening/Puncturing) ---")
    qber_values = np.arange(0.01, 0.15, 0.02)
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-axis for Efficiency
    color_eff = 'tab:blue'
    ax1.set_xlabel('QBER', fontsize=12)
    ax1.set_ylabel('Efficiency ($f$)', color=color_eff, fontsize=12)
    ax1.set_ylim(1.0, 2.0)
    ax1.tick_params(axis='y', labelcolor=color_eff)

    # Secondary Y-axis for Effectiveness
    ax2 = ax1.twinx()
    color_ext = 'tab:red'
    ax2.set_ylabel('Effectiveness ($f\'$)', color=color_ext, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_ext)

    for R in R_RANGE:
        print(f"Testing Base Rate R={R:.4f}...")
        efficiencies = []
        effectiveness = []
        valid_qbers = []
        
        for qber in qber_values:
            print(f"  QBER={qber:.4f}...")
            f = run_manual_test(qber, R, 0, 0, n_tries=3)
            if f is not None and f < 2.0:
                efficiencies.append(f)
                effectiveness.append(calc_f_prime(f, qber))
                valid_qbers.append(qber)
        
        if len(efficiencies) > 0:
            # Plot Efficiency on ax1 (Solid line)
            line1, = ax1.plot(valid_qbers, efficiencies, marker='.', linestyle='-', 
                              alpha=0.8, label=f'Eff ($f$) R={R:.2f}')
            # Plot Effectiveness on ax2 (Dashed line) using the same color as the efficiency line
            ax2.plot(valid_qbers, effectiveness, marker='x', linestyle='--', 
                    color=line1.get_color(), alpha=0.5, label=f'Effect. ($f\'$) R={R:.2f}')

    plt.title('Efficiency and Effectiveness vs QBER (No S/P)', pad=15)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Creating a combined legend for both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize='small', ncol=2)

    fig.tight_layout()
    plt.savefig('task2_base_rates_dual_axis.png')
    plt.show()

# =============================================================================
# TASK 3: LLR Evolution Tracking
# =============================================================================
def task_3_llr_tracking():
    print("\n--- Task 3: LLR Evolution Tracking ---")
    # Setup single simulation
    qber = 0.06
    R = 0.6667
    s_n, p_n = 0, 0
    
    code_params = CODES[(R, N_BLOCK)]
    m = (1-R)*N_BLOCK
    
    # Generate data
    x = ec.generate_key(N_BLOCK - s_n - p_n)
    y = ec.add_errors_prec(x, qber)
    
    # Run with tracking
    _, _, x_dec, _, llr_history = ec.perform_ec(
        x, y, 
        code_params['s_y_joins'], code_params['y_s_joins'], 
        qber, s_n, p_n, punct_list=code_params['punct_list'], 
        discl_n=20, show=0, track_llr=True
    )
    
    if llr_history is None:
        print("Decoding converged too fast or failed to track.")
        return

    # Process LLRs
    # We want to pick a few interesting bits:
    # 1. A bit that was an error (flipped) and got corrected.
    # 2. A bit that was correct and stayed correct.
    
    error_indices = np.where(x != y)[0]
    correct_indices = np.where(x == y)[0]
    
    # Safety check
    if len(error_indices) == 0: return

    # Pick 2 error bits and 2 correct bits to trace
    trace_indices = np.concatenate([error_indices[:2], correct_indices[:2]])
    
    llr_data = np.array(llr_history) # Shape: (Iterations, Block_Size)
    
    plt.figure(figsize=(10, 6))
    iterations = range(llr_data.shape[0])
    
    for idx in trace_indices:
        label = "Error Bit" if idx in error_indices else "Correct Bit"
        # We plot absolute LLR magnitude because sign indicates value (0 or 1)
        # Magnitude indicates confidence.
        plt.plot(iterations, np.abs(llr_data[:, idx]), label=f'{label} (idx {idx})')

    plt.title(f'LLR Magnitude Evolution (QBER={qber}, R={R:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('LLR Magnitude (|L|)')
    plt.legend()
    plt.grid(True)
    plt.savefig('task3_llr_evolution.png')
    print("Saved task3_llr_evolution.png")


# --- Main Execution ---
if __name__ == "__main__":
    task_1_vary_sp()
    task_2_vary_base_rate()
    task_3_llr_tracking()
    print("\nAll simulations complete.")