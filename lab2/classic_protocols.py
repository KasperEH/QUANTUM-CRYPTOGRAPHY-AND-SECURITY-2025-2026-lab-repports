import numpy as np
import random
from math import ceil, log2, floor

def generate_key(length):
    return np.random.randint(0, 2, length)

def add_errors(key, qber):
    """Simulates a BSC channel."""
    err_key = key.copy()
    num_errors = int(len(key) * qber)
    # Using deterministic count for stability in comparison, 
    # but positions are random (as per your request for 'Deterministic')
    indices = random.sample(range(len(key)), num_errors)
    for idx in indices:
        err_key[idx] = 1 - err_key[idx]
    return err_key

# ==============================================================================
# 1. CLASSIC HAMMING (7,4) LOGIC
# ==============================================================================
# H matrix for Hamming(7,4)
H_74 = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1]
])

def syndrome_hamming74(block):
    """Calculates syndrome for a 7-bit block."""
    if len(block) != 7: return None
    # s = H * v^T  (mod 2)
    return np.dot(H_74, block) % 2

def correct_hamming74(block):
    """
    Corrects a single error in a 7-bit block.
    Returns: (corrected_block, bits_revealed)
    bits_revealed = 3 (The syndrome length)
    """
    syn = syndrome_hamming74(block)
    syn_val = syn[0]*4 + syn[1]*2 + syn[2]*1 # Convert binary [s1, s2, s3] to int
    
    if syn_val != 0:
        # Error position is syn_val - 1 (1-based index in standard Hamming)
        # Note: The H matrix above assumes columns 1..7. 
        # For standard H sorted by binary value, position is strictly syn_val-1.
        error_pos = int(syn_val) - 1
        block[error_pos] = 1 - block[error_pos]
        
    return block, 3 # 3 bits (syndrome) revealed

# ==============================================================================
# 2. WINNOW PROTOCOL
# ==============================================================================
def run_winnow(alice, bob, qber):
    """
    Implements the Winnow protocol (Hamming based).
    Simplified version based on Slide 31.
    """
    n = len(alice)
    alice_corr = alice.copy()
    bob_corr = bob.copy()
    
    bits_revealed = 0
    iterations = 0
    
    # Winnow typically uses Block Size = 8 for Hamming(7,4) compatibility
    # (1 parity bit + 7 code bits)
    block_size = 8 
    
    # --- Pass 1: Parity Check & Hamming Correction ---
    # Split into blocks
    num_blocks = n // block_size
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        
        a_block = alice_corr[start:end]
        b_block = bob_corr[start:end]
        
        # 1. Public Parity Check
        p_a = np.sum(a_block) % 2
        p_b = np.sum(b_block) % 2
        bits_revealed += 1
        
        if p_a != p_b:
            iterations += 1
            # Mismatch detected.
            # Slide 31: "Remove one bit (to get 7), do Hamming correction"
            # We treat the first bit as the 'removed' bit or just apply Hamming to last 7
            
            # Extract last 7 bits for Hamming(7,4)
            # (In real Winnow, we might discard a bit to maintain privacy, 
            # here we focus on correction mechanics)
            a_sub = a_block[1:] 
            b_sub = b_block[1:]
            
            # Alice sends Syndrome of her 7 bits
            s_a = syndrome_hamming74(a_sub)
            s_b = syndrome_hamming74(b_sub)
            bits_revealed += 3
            
            diff_syndrome = (s_a + s_b) % 2
            diff_val = diff_syndrome[0]*4 + diff_syndrome[1]*2 + diff_syndrome[2]*1
            
            if diff_val != 0:
                # Error is in the 7 bits
                err_idx = int(diff_val) - 1
                b_block[1 + err_idx] = 1 - b_block[1 + err_idx] # Correct Bob
            else:
                # If parity mismatch BUT syndrome match -> Error is in the 1st bit (the one we skipped)
                b_block[0] = 1 - b_block[0]

            # Update master array
            bob_corr[start:end] = b_block

    return bob_corr, bits_revealed, iterations

def run_winnow_iterative(alice, bob, qber):
    # Maximum iterations to prevent infinite loops
    MAX_ITER = 5 
    
    # Working copies
    alice_curr = alice.copy()
    bob_curr = bob.copy()
    
    total_revealed = 0
    
    # Generate a permutation index list
    n = len(alice)
    perm = np.arange(n)
    
    for i in range(MAX_ITER):
        # 1. Run Single Pass Winnow
        bob_curr, revealed, _ = run_winnow(alice_curr, bob_curr, qber)
        total_revealed += revealed
        
        # 2. Check if we are done (optional optimization: Check Parity of whole key)
        if np.array_equal(alice_curr, bob_curr):
            print(f"Converged in {i+1} iterations")
            return bob_curr, total_revealed
            
        # 3. SHUFFLE (Permute)
        # Alice and Bob must permute exactly the same way
        np.random.shuffle(perm)
        alice_curr = alice_curr[perm]
        bob_curr = bob_curr[perm]
        
    # Un-shuffle at the end if you need the original order (usually QKD just uses the final random string)
    # If un-shuffle is needed, you need to track the inverse permutation.
    
    return bob_curr, total_revealed

# ==============================================================================
# 3. CASCADE PROTOCOL
# ==============================================================================
def get_parity(block):
    return np.sum(block) % 2

def binary_search_correction(alice_block, bob_block):
    """
    Recursively finds the error in a block known to have odd parity diff.
    Returns: (corrected_bob_block, bits_revealed)
    """
    n = len(alice_block)
    revealed = 0
    
    # Base case
    if n == 1:
        bob_block[0] = 1 - bob_block[0] # Flip it
        return bob_block, 0
    
    mid = n // 2
    
    # Alice sends parity of left half
    p_a_left = get_parity(alice_block[:mid])
    p_b_left = get_parity(bob_block[:mid])
    revealed += 1
    
    if p_a_left != p_b_left:
        # Error is in left half
        new_left, r = binary_search_correction(alice_block[:mid], bob_block[:mid])
        bob_block[:mid] = new_left
        revealed += r
    else:
        # Error is in right half
        new_right, r = binary_search_correction(alice_block[mid:], bob_block[mid:])
        bob_block[mid:] = new_right
        revealed += r
        
    return bob_block, revealed

def run_cascade(alice, bob, qber):
    """
    Implements a simplified multi-pass Cascade protocol.
    """
    n = len(alice)
    bob_corr = bob.copy()
    bits_revealed = 0
    iterations = 0 # Counting 'passes' as iterations
    
    # Heuristic block size calculation (from literature/slides implies 1/QBER or similar)
    if qber > 0:
        current_block_size = int(0.73 / qber)
    else:
        current_block_size = n
        
    if current_block_size < 4: current_block_size = 4
    
    # We do 4 passes (standard Cascade is usually 4)
    for pass_idx in range(4):
        iterations += 1
        
        # In passes > 0, we permute. 
        # For simplicity in this simulation, we just shuffle indices differently every pass
        # unless it's pass 0.
        if pass_idx == 0:
            perm_indices = np.arange(n)
        else:
            perm_indices = np.random.permutation(n)
            current_block_size = current_block_size * 2 # Double block size
            
        inv_perm = np.argsort(perm_indices)
        
        # Apply permutation
        a_perm = alice[perm_indices]
        b_perm = bob_corr[perm_indices]
        
        # Process blocks
        num_blocks = ceil(n / current_block_size)
        
        for i in range(num_blocks):
            start = i * current_block_size
            end = min(start + current_block_size, n)
            if start >= n: break
            
            # Check Parity
            p_a = get_parity(a_perm[start:end])
            p_b = get_parity(b_perm[start:end])
            bits_revealed += 1
            
            if p_a != p_b:
                # Perform Binary Search to fix 1 error
                b_perm[start:end], r = binary_search_correction(a_perm[start:end], b_perm[start:end])
                bits_revealed += r
        
        # Inverse Permutation (put bits back)
        bob_corr = b_perm[inv_perm]

    return bob_corr, bits_revealed, iterations