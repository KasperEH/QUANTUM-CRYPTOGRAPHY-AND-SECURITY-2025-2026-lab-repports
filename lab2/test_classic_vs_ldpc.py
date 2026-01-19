import numpy as np
import matplotlib.pyplot as plt
import classic_protocols as cp
import error_correction_lib as ec
from file_utils import codes_from_file
import time

def calc_efficiency(n_revealed, n_total, qber):
    """Calculates efficiency f. Returns None if QBER is invalid."""
    if qber <= 0 or qber >= 0.5: return None
    h = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
    if h == 0: return None
    return n_revealed / (n_total * h)

def calc_f_prime(f, qber):
    """Calculates f' (waste)."""
    if qber <= 0 or qber >= 0.5 or np.isnan(f): return np.nan
    return (f - 1) * ec.h_b(qber)

def test_protocols():
    # --- SETUP ---
    N = 1944  # Matches LDPC matrix size
    codes = codes_from_file('codes_1944.txt')
    R_range = [code[0] for code in codes]

    qber_range = np.arange(0.01, 0.25, 0.02)
    n_tries = 5 
    
    results = {
        'cascade': {'f': [], 'f_prime': [], 'falures': 0},
        'winnow':  {'f': [], 'f_prime': [], 'falures': 0},
        'ldpc':    {'f': [], 'f_prime': [], 'falures': 0}
    }


    for qber in qber_range:

        print("QBER: " + str(qber))
        f_sums = {'cascade': [], 'winnow': [], 'ldpc': []}

        # Progress indicator
        print(f"Proc QBER {qber:.2f} ", end='', flush=True)

        for _ in range(n_tries):
            alice = cp.generate_key(N)
            bob_raw = cp.add_errors(alice, qber) 
            
            # --- 1. CASCADE ---
            bob_c, rev_c, _ = cp.run_cascade(alice, bob_raw, qber)
            #if np.array_equal(alice, bob_c):
            f_sums['cascade'].append(calc_efficiency(rev_c, N, qber))
            if not np.array_equal(alice, bob_c):
                results['cascade']['falures'] += 1

            # --- 2. WINNOW ---
            bob_w, rev_w, _ = cp.run_winnow(alice, bob_raw, qber)
            #if np.array_equal(alice, bob_w):
            f_sums['winnow'].append(calc_efficiency(rev_w, N, qber))
            if not np.array_equal(alice, bob_w):
                results['winnow']['falures'] += 1


            # --- 3. LDPC ---
            R, s_n, p_n = ec.choose_sp(qber, 1.2, R_range, N)
            
            if R is None:
                # No suitable code found
                results['ldpc']['falures'] += 1

            if R is not None:
                k_ldpc = N - s_n - p_n
                alice_ldpc = alice[:k_ldpc]
                bob_ldpc = bob_raw[:k_ldpc]
                
                m = (1-R)*N
                discl_n = int(round(N*(0.0280-0.02*R)))
                code_params = codes[(R, N)]
                
                add_info, _, _, ver_check, _ = ec.perform_ec(
                    alice_ldpc, bob_ldpc, 
                    code_params['s_y_joins'], code_params['y_s_joins'], 
                    qber, s_n, p_n, punct_list=code_params['punct_list'], 
                    discl_n=discl_n, show=0
                )
                
                if ver_check:
                    curr_f = float(m - p_n + add_info)/(k_ldpc)/ec.h_b(qber)
                    f_sums['ldpc'].append(curr_f)
            
            print(".", end="", flush=True)

        print(" Done!") 
        
        # --- Store Results ---
        row_str = ""
        
        for proto in ['cascade', 'winnow', 'ldpc']:
            # >>> FIX: Handle both lists inside the check <<<
            mean_f = np.mean(f_sums[proto])
            results[proto]['f'].append(mean_f)
            results[proto]['f_prime'].append(calc_f_prime(mean_f, qber))
            
            # Format string for console
            fp_val = results[proto]['f_prime'][-1]
            row_str += f"{fp_val:<10.4f} | "
           

        print(f" -> Result: {row_str}")

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    protocols = [
        ('cascade', 'Cascade', 'green', '^'),
        ('winnow', 'Winnow', 'orange', 's'),
        ('ldpc', 'LDPC', 'blue', 'o')
    ]

    lines = []
    for key, name, color, marker in protocols:
        # Filter NaNs so lines connect properly
        valid_idx = np.isfinite(results[key]['f'])
        if np.any(valid_idx):
            x_vals = np.array(qber_range)[valid_idx]
            
            # 1. Primary Axis: Efficiency f (Solid)
            l_f, = ax1.plot(x_vals, np.array(results[key]['f'])[valid_idx], 
                            linestyle='-', marker=marker, color=color, label=f'{name} Eff ($f$)')
            lines.append(l_f)
            
            # 2. Secondary Axis: f' (Dotted)
            l_fp, = ax2.plot(x_vals, np.array(results[key]['f_prime'])[valid_idx], 
                             linestyle=':', marker=marker, color=color, alpha=0.6, label=f'{name} Effectiveness ($f\'$)')
            lines.append(l_fp)

    print("falures:")
    for key in ['cascade', 'winnow', 'ldpc']:
        print(f"  {key}: {results[key]['falures']}")

    # Axis 1 Settings
    ax1.set_title('Protocol Comparison: Efficiency ($f$) vs. Effectiveness ($f\'$)', fontsize=14)
    ax1.set_xlabel('QBER')
    ax1.set_ylabel('Efficiency Ratio $f$ (Ideal = 1.0)')
    ax1.set_ylim(1.0, 1.8) # Adjusted for typical f values
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Axis 2 Settings
    ax2.set_ylabel('Effectiveness $f\'$ (Bits per Symbol)')
    ax2.set_ylim(0, 0.2) # f' is usually small (0.0 to 0.15)

    # Legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=3, fontsize='small', frameon=True)

    plt.tight_layout()
    plt.savefig('protocol_comparison_f_prime.png')
    print("\nSaved plot to 'protocol_comparison_f_prime.png'")
    plt.show()

if __name__ == "__main__":
    test_protocols()