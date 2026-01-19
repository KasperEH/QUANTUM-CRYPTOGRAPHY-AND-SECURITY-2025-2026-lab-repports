import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
from os import path

def calc_f_prime(f, qber):
    """
    Calculates f' (f-prime).
    Formula: f' = (l/n) - h(Q) = f*h(Q) - h(Q) = (f-1)*h(Q)
    """
    if qber <= 0 or qber >= 0.5: return 0.0
    h_q = ec.h_b(qber)
    return (f - 1) * h_q

# Choose of the codes pool:
#codes = codes_from_file('codes_4000.txt'); n = 4000
codes = codes_from_file('codes_1944.txt')
n = 1944

# Computing the range of rates for given codes
R_range = []
for code in codes:
    R_range.append(code[0])
print(f"R range is: {np.sort(R_range)}")

fname = 'output.txt'  # file name for the output
f_start = 1.0  # initial efficiency of decoding
qber_start = 0.01
qber_end = 0.3
qber_step = 0.02  # range of QBERs
n_tries = 2  # number of keys proccessed for each QBER value
qber_values = np.arange(qber_start, qber_end + qber_step/10, qber_step)

# --- 1. PRINT THE SCHEMA (New Requirement) ---
print("Generating Parameter Schema...")
ec.print_parameters_schema(qber_values, R_range, n)

if path.exists(fname):
    file_output = open(fname, 'a')
else:
    file_output = open(fname, 'w')
    file_output.write(
        "code_n, n_tries, qber, f_mean, com_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER \n")

print(r"""
\begin{table}[htbp]
    \centering
    \label{tab:efficiency_comparison}
    \caption{Comparison of Effectiveness and Efficiency: Random (BSC) vs. Deterministic Errors}
    \begin{tabularx}{\textwidth}{c | cccc | cccc}
        \toprule
        & \multicolumn{4}{c|}{\textbf{Random Errors (BSC)}} & \multicolumn{4}{c}{\textbf{Deterministic Errors}} \\
        \textbf{QBER} & \textbf{$f$} & \textbf{$f'$} & \textbf{Iter} & \textbf{FER} & \textbf{$f$} & \textbf{$f'$} & \textbf{Iter} & \textbf{FER} \\
        \midrule""")

# --- RUN SIMULATION & PRINT ROWS ---
for qber in qber_values:
    # 1. Random (BSC) Run
    f_r, it_r, _, _, _, _, _, _, fer_r = ec.test_ec(
        qber, R_range, codes, n, n_tries, show=0, error_func=ec.add_errors)
    fp_r = calc_f_prime(f_r, qber)

    # 2. Deterministic (Fixed) Run
    f_d, it_d, _, _, _, _, _, _, fer_d = ec.test_ec(
        qber, R_range, codes, n, n_tries, show=0, error_func=ec.add_errors_prec)
    fp_d = calc_f_prime(f_d, qber)

    # Print LaTeX Row
    # Format: QBER & f & f' & It & FER & f & f' & It & FER \\
    print(f"{qber:.4f} & {f_r:.4f} & {fp_r:.4f} & {it_r:.1f} & {fer_r:.2f} & {f_d:.4f} & {fp_d:.4f} & {it_d:.1f} & {fer_d:.2f} \\\\")

# --- PRINT LATEX FOOTER ---
print(r"""        \bottomrule
    \end{tabularx}
\end{table}
""")
