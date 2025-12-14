import sys
import os
import math
import numpy as np

# --- Helper Functions ---

def get_times_by_channel(filename, duration_s=None, verbose=True):
    """
    Parses the file and separates time tags by channel,
    optionally for a limited duration.
    """
    bob_times = []  # Channel 2
    alice_times = [] # Channel 3
    unit_time_ps = 80.955
    first_time_tag_ps = None
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    if 'Unit Time Tag:' in line:
                        try:
                            unit_time_ps = float(line.split(':')[1].strip().split('ps')[0])
                        except Exception:
                            pass
                    continue
            
            f.seek(0)
            
            for line in f:
                if line.strip().startswith('#'):
                    continue
                
                parts = line.split(';')
                if len(parts) == 2:
                    try:
                        time_tag = int(parts[0].strip())
                        channel = parts[1].strip()
                        time_in_ps = time_tag * unit_time_ps
                        
                        if first_time_tag_ps is None:
                            first_time_tag_ps = time_in_ps
                        
                        if duration_s is not None:
                            if (time_in_ps - first_time_tag_ps) > (duration_s * 1_000_000_000):
                                break
                        
                        if channel == '2':
                            bob_times.append(time_in_ps)
                        elif channel == '3':
                            alice_times.append(time_in_ps)
                    except Exception:
                        continue
                        
    except FileNotFoundError:
        if verbose: print(f"Error: File not found {filename}")
        return None, None
        
    return sorted(bob_times), sorted(alice_times)

def count_coincidences(filename, delay_ps, window_ps, duration_s=None, verbose=True):
    """
    Counts 1-to-1 coincidences between Alice (Ch 3) and Bob (Ch 2).
    """
    bob_times, alice_times = get_times_by_channel(filename, duration_s, verbose=verbose)
    
    if bob_times is None or not bob_times or not alice_times:
        if verbose: print(f"Error for {filename}: Could not find events for counting.")
        return 0

    coincidences = 0
    i = 0  # Pointer for Bob
    j = 0  # Pointer for Alice
    half_window = window_ps / 2.0
    
    while i < len(bob_times) and j < len(alice_times):
        bob_corrected_time = bob_times[i] + delay_ps
        diff = alice_times[j] - bob_corrected_time
        
        if abs(diff) <= half_window:
            coincidences += 1
            i += 1
            j += 1
        elif diff < -half_window:
            j += 1
        elif diff > half_window:
            i += 1
            
    if verbose:
        print(f"--- Coincidence Results for: {filename} ---")
        print(f"Parameters: Delay={delay_ps} ps, Window={window_ps} ps")
        if duration_s:
            print(f"(Processed for {duration_s}s duration)")
        print(f"Raw Ch 2 (Bob) Events:   {len(bob_times):,}")
        print(f"Raw Ch 3 (Alice) Events: {len(alice_times):,}")
        print(f"Coincidence Count:       {coincidences:,}")
        
    return coincidences

# --- ANALYSIS FUNCTION ---

def analyze_state(state_prefix, data_path, delay_ps, window_ps, duration_s=None):
    """
    Performs the full min-entropy analysis for a given state.
    """
    print(f"\n{'=' * 50}")
    print(f"  Analysing State: {state_prefix.upper()}")
    print(f"{'=' * 50}")
    
    projectors = ['H', 'V', 'D', 'A', 'R', 'L']
    counts = {}
    
    # 1. Get all 6 coincidence counts
    for proj in projectors:
        filename = os.path.join(data_path, f"{state_prefix}_measured_{proj}.txt")
        print(f"Processing: {filename} ...")
        
        c = count_coincidences(filename, delay_ps, window_ps, duration_s, verbose=False)
        if c == 0:
            print(f"Warning: Found 0 coincidences for {filename}. Check file or parameters.")
        counts[proj] = c
    
    print("\n--- Coincidence Counts ---")
    print(f"  H: {counts['H']:,}  |  V: {counts['V']:,}")
    print(f"  D: {counts['D']:,}  |  A: {counts['A']:,}")
    print(f"  R: {counts['R']:,}  |  L: {counts['L']:,}")
    
    # 2. Calculate Probabilities and Bloch Vector
    try:
        # Z-Basis
        C_total_z = counts['H'] + counts['V']
        P_H = counts['H'] / C_total_z if C_total_z > 0 else 0.5
        P_V = counts['V'] / C_total_z if C_total_z > 0 else 0.5
        r_z = P_H - P_V
        
        # X-Basis
        C_total_x = counts['D'] + counts['A']
        P_D = counts['D'] / C_total_x if C_total_x > 0 else 0.5
        P_A = counts['A'] / C_total_x if C_total_x > 0 else 0.5
        r_x = P_D - P_A
        
        # Y-Basis
        C_total_y = counts['R'] + counts['L']
        P_R = counts['R'] / C_total_y if C_total_y > 0 else 0.5
        P_L = counts['L'] / C_total_y if C_total_y > 0 else 0.5
        r_y = P_R - P_L
        
    except ZeroDivisionError:
        print("\nError: Total counts for a basis was zero. Cannot calculate probabilities.")
        return None

    r_vec = np.array([r_x, r_y, r_z])
    r_len = np.linalg.norm(r_vec)
    
    print("\n--- Bloch Vector Analysis ---")
    print(f"  <sigma_x>: {r_x:+.4f}")
    print(f"  <sigma_y>: {r_y:+.4f}")
    print(f"  <sigma_z>: {r_z:+.4f}")
    print(f"  Vector Length |r|: {r_len:.4f}")

    if r_len > 1:
        r_vec_phys = r_vec / r_len
        r_len_phys = 1.0
    else:
        r_vec_phys = r_vec
        r_len_phys = r_len

    # 3. Calculate Min-Entropy (H_min)
    
    # Protocol 1: Trusted (Z-Basis)
    P_max_trusted = max(P_H, P_V)
    if P_max_trusted == 0: P_max_trusted = 0.5
    h_min_trusted = -math.log2(P_max_trusted)

    # Protocol 2: SDI-EUP [Vallone et al.]
    term_inside_log = 0.5 + 0.5 * math.sqrt(max(0, 1 - r_vec_phys[0]**2))
    h_min_eup = -math.log2(term_inside_log)
    
    # Protocol 3: SDI-TOMO (Full Tomography)
    h_min_tomo = -math.log2( (1 + r_len_phys) / 2 )

    # Protocol 4: SDI - POVM
    total_povm_counts = counts['H'] + counts['V'] + counts['D'] + counts['A']
    if total_povm_counts > 0:
        prob_H_povm = counts['H'] / total_povm_counts
        prob_V_povm = counts['V'] / total_povm_counts
        prob_D_povm = counts['D'] / total_povm_counts
        prob_A_povm = counts['A'] / total_povm_counts
        max_p_povm = max(prob_H_povm, prob_V_povm, prob_D_povm, prob_A_povm)
        h_min_povm = -math.log2(max_p_povm)
    else:
        h_min_povm = 0.0

    print("\n--- Min-Entropy (H_min) Results ---")
    print(f"  Protocol 1 (Trusted):    {h_min_trusted:.4f} bits")
    print(f"  Protocol 2 (SDI-EUP):    {h_min_eup:.4f} bits")
    print(f"  Protocol 3 (SDI-TOMO):   {h_min_tomo:.4f} bits")
    print(f"  Protocol 4 (SDI-POVM):   {h_min_povm:.4f} bits")
    
    # We return SDI-TOMO (Protocol 3) as the baseline for security comparison
    return h_min_tomo 

# --- NEW COMBINED PLOTTING FUNCTION ---

def plot_combined_security(results):
    """
    Generates a single plot comparing the security of all analyzed states.
    results: list of tuples (state_prefix, h_min)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWarning: Matplotlib not found. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    n_values = np.linspace(100, 2000, 100) # Block lengths
    
    # Define styles for clarity
    styles = ['-', '--', '-.', ':']
    
    print("\n--- Generating Combined Security Plot ---")
    
    for idx, (state_prefix, h_min) in enumerate(results):
        if h_min <= 0.0001:
            print(f"  State '{state_prefix}': H_min ~ 0. Plotting as insecure (epsilon=1).")
            # Plot a flat line at epsilon = 1
            plt.plot(n_values, [1.0]*len(n_values), 
                     label=f"{state_prefix.upper()} ($H_{{min}} \\approx 0$)",
                     linestyle=styles[idx % len(styles)], linewidth=2)
        else:
            # Pick extraction rate R
            R = h_min * 0.8
            # Calculate epsilon
            exponent = - (n_values / 2.0) * (h_min - R)
            epsilon_values = np.power(2.0, exponent)
            
            print(f"  State '{state_prefix}': Plotting curve (H_min={h_min:.3f}, R={R:.3f})")
            plt.semilogy(n_values, epsilon_values, 
                         label=f"{state_prefix.upper()} ($H_{{min}}={h_min:.2f}$)",
                         linestyle=styles[idx % len(styles)], linewidth=2)

    plt.title("Security vs. Block Length (Comparison using SDI-TOMO)", fontsize=16)
    plt.xlabel("Block Length (n)", fontsize=12)
    plt.ylabel("Security Parameter ($\epsilon$) [Lower is Better]", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Set y-axis limits to make it readable (don't go too low or high)
    plt.ylim(1e-30, 1.5) 
    
    plot_filename = "security_plot_combined.png"
    plt.savefig(plot_filename)
    print(f"Success: Saved combined plot to {plot_filename}")
    plt.close()


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python analyze_report_v3.py <delay_ps> <window_ps> --path <data_path> --states <prefix1> [prefix2 ...]")
        print("       [--duration <seconds>] [--plot_security]")
        sys.exit(1)

    try:
        delay_ps = float(sys.argv[1])
        window_ps = float(sys.argv[2])
        
        args = sys.argv[3:]
        
        data_path = ""
        state_prefixes = []
        duration_s = None
        do_plot = False
        
        i = 0
        while i < len(args):
            if args[i] == '--path':
                data_path = args[i+1]
                i += 2
            elif args[i] == '--states':
                i += 1
                while i < len(args) and not args[i].startswith('--'):
                    state_prefixes.append(args[i])
                    i += 1
            elif args[i] == '--duration':
                duration_s = float(args[i+1])
                i += 2
            elif args[i] == '--plot_security':
                do_plot = True
                i += 1
            else:
                print(f"Unknown argument: {args[i]}")
                sys.exit(1)
        
        if not data_path or not state_prefixes:
            print("Error: --path and --states must be provided.")
            sys.exit(1)
        
        print(f"Starting analysis with: Delay={delay_ps} ps, Window={window_ps} ps")
        
        # Collect results here
        plot_results = []
        
        for prefix in state_prefixes:
            h_min_result = analyze_state(prefix, data_path, delay_ps, window_ps, duration_s)
            if h_min_result is not None:
                plot_results.append((prefix, h_min_result))
            
        # Plot combined at the end
        if do_plot and plot_results:
            plot_combined_security(plot_results)

    except ValueError:
        print("Error: Delay, window, and duration must be numbers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")