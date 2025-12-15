import numpy as np
import matplotlib.pyplot as plt
import sift_cython 

# --- CONFIGURATION ---
BLOCK_SIZE = 240000 
PACKET_SIZE = 24000 # Size of chunks in bitpos.txt
PACKETS_PER_BLOCK = BLOCK_SIZE // PACKET_SIZE # Should be 10

MU_1 = 0.60  
MU_2 = 0.17 
MU_3 = 0.0001 

EPS_SEC = 1e-10
EPS_COR = 1e-15
f_EC = 1.16

def binary_entropy(x):
    if x <= 0 or x >= 1: return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def calculate_bounds(n, epsilon):
    """
    Implements Rusca Eq. (3) for finite-size statistical bounds.
    """
    if n == 0: return 0, 0
    fluctuation = np.sqrt((n / 2) * np.log(1 / epsilon))
    n_lower = n - fluctuation
    n_upper = n + fluctuation
    return (0 if n_lower < 0 else n_lower), n_upper

def calculate_skr(counts, duration):
    # Unpack
    n_Z = counts['n_Z'] # [Vac, Decoy, Signal]
    n_sent = counts['sent']
    
    # 1. Finite-Size Bounds (Eq 3 from your image)
    _, n_Z0_upper = calculate_bounds(n_Z[0], EPS_SEC)
    n_Z1_lower, _ = calculate_bounds(n_Z[1], EPS_SEC)
    n_Z2_lower, _ = calculate_bounds(n_Z[2], EPS_SEC)

    # 2. Pessimistic Yields
    # Denominator is Raw Sent because Alice knows exactly how many she sent.
    Y_0 = n_Z0_upper / n_sent[0] if n_sent[0] > 0 else 0
    Y_1 = n_Z1_lower / n_sent[1] if n_sent[1] > 0 else 0
    Y_2 = n_Z2_lower / n_sent[2] if n_sent[2] > 0 else 0

    # 3. Decoy State Calculation (Algebraic Solution)
    term_decoy = (MU_1**2) * np.exp(-MU_1) * (Y_1 - Y_0)
    term_signal = (MU_2**2) * np.exp(-MU_2) * (Y_2 - Y_0)
    denominator = MU_1 * MU_2 * (MU_1 - MU_2)
    
    y_1_lower = (term_decoy - term_signal) / denominator
    if y_1_lower < 0: y_1_lower = 0
    
    # Total secure single-photon events
    s_Z1 = n_sent[2] * (MU_1 * np.exp(-MU_1)) * y_1_lower

    # 4. Phase Error & Leakage
    E_Z_signal = counts['m_Z'][2] / counts['n_Z'][2] if counts['n_Z'][2] > 0 else 0.5
    E_X_signal = counts['m_X'][2] / counts['n_X'][2] if counts['n_X'][2] > 0 else 0.5
    
    phi_Z = E_X_signal + 0.01 
    n_Z_tot = sum(n_Z)
    
    leak_EC = f_EC * n_Z_tot * binary_entropy(E_Z_signal)
    penalty = np.log2(1 / EPS_COR) 

    l = s_Z1 * (1 - binary_entropy(phi_Z)) - leak_EC - penalty
    
    rate = l / duration if duration > 0 else 0
    return rate, E_Z_signal, E_X_signal

def parse_time_data(filepath, limit_blocks=None):
    """
    Reads bitpos.txt to find the start time of every packet.
    Returns a list of start times (in seconds).
    """
    print(f"Scanning {filepath} for timestamps...")
    packet_starts = []
    
    # We look for lines containing "-1". The NEXT line is a start position.
    # Since reading line-by-line in Python can be slow for 50MB, we look for -1 carefully.
    
    with open(filepath, 'r') as f:
        # Check first line for -1
        line = f.readline()
        while line:
            if line.strip() == '-1':
                # The next line is the Absolute Position
                pos_line = f.readline()
                if not pos_line: break
                
                abs_pos = int(pos_line.strip())
                # Convert 50MHz clock cycles to seconds
                t_sec = abs_pos / 50e6 
                packet_starts.append(t_sec)
                
                # Optimization: We know the next ~24000 lines are diffs.
                # We could skip them, but standard readline is safe enough.
            
            line = f.readline()
            
            # Stop if we have enough data for our blocks to save time
            if limit_blocks and len(packet_starts) > limit_blocks * PACKETS_PER_BLOCK + 20:
                break
                
    return packet_starts

# --- MAIN EXECUTION ---
def main():
    # 1. Load Data
    print("Loading binary files...")
    with open('states.txt', 'rb') as f:
        states = np.frombuffer(f.read(), dtype=np.uint8).copy()
    with open('statesRCV.txt', 'rb') as f:
        rcv = np.frombuffer(f.read(), dtype=np.uint8).copy()
    with open('decoy.txt', 'rb') as f:
        decoy = np.frombuffer(f.read(), dtype=np.uint8).copy()
        
    total_len = len(states)
    num_blocks = total_len // BLOCK_SIZE
    print(f"Total events: {total_len} ({num_blocks} blocks)")

    # 2. Get Time Info
    # We expect roughly 10 packets per block.
    # We need timestamps for packets 0, 10, 20...
    packet_times = parse_time_data('bitpos.txt', limit_blocks=num_blocks)
    
    # 3. Analyze Blocks
    print("Analyzing...")
    results_time = []
    results_skr = []
    results_qber_z = []
    results_qber_x = []

    print(f"{'Time (s)':<10} | {'QBER_Z':<8} | {'QBER_X':<8} | {'SKR (bps)':<10}")
    print("-" * 50)
    
    for i in range(num_blocks):
        start_idx = i * BLOCK_SIZE
        end_idx = start_idx + BLOCK_SIZE
        
        # Determine Duration from bitpos data
        # Block i starts at packet i*10
        p_start_idx = i * PACKETS_PER_BLOCK
        p_end_idx = (i + 1) * PACKETS_PER_BLOCK
        
        if p_end_idx < len(packet_times):
            t_start = packet_times[p_start_idx]
            t_end = packet_times[p_end_idx] # Start of next block ~ End of current
            duration = t_end - t_start
        else:
            # Fallback for last block if data incomplete
            duration = (BLOCK_SIZE / 50e6) * 20 # Rough estimate
            t_start = results_time[-1] + duration if results_time else 0

        # Slicing & Cython
        b_states = states[start_idx:end_idx]
        b_rcv = rcv[start_idx:end_idx]
        b_decoy = decoy[start_idx:end_idx]
        
        counts = sift_cython.get_counts(b_states, b_rcv, b_decoy)
        skr, qber_z, qber_x = calculate_skr(counts, duration)
        
        if skr < 0: skr = 0
        
        # Store for plotting
        results_time.append(t_start)
        results_skr.append(skr)
        results_qber_z.append(qber_z)
        results_qber_x.append(qber_x)
        
        print(f"{t_start:<10.2f} | {qber_z:.4f}   | {qber_x:.4f}   | {skr:.2e}")

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    results_time = results_time[:-1]
    results_skr = results_skr[:-1]
    results_qber_z = results_qber_z[:-1]
    results_qber_x = results_qber_x[:-1]

    # Plot SKR on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Secret Key Rate (bps)', color=color)
    ax1.plot(results_time, results_skr, color=color, label='SKR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot QBER on right axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('QBER', color=color)  
    ax2.plot(results_time, results_qber_z, color=color, linestyle='--', label='QBER Z')
    ax2.plot(results_time, results_qber_x, color='tab:orange', linestyle=':', label='QBER X')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 0.10) # Zoom in on valid QBER range

    plt.title('QKD Performance: SKR and QBER vs Time')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()