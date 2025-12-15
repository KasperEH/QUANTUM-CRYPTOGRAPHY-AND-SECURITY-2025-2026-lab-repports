# sift_cython.pyx
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, int64_t

def get_counts(uint8_t[:] tx_states, uint8_t[:] rx_states, uint8_t[:] decoy_levels):
    """
    Sifts the data and counts events/errors.
    Also counts the TOTAL SENT pulses for each intensity to calculate Yields.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = tx_states.shape[0]
    
    # Storage for Detections: [Basis 0=Z, 1=X][Intensity 0=Vac, 1=Dec, 2=Sig]
    cdef int64_t n_counts[2][3] 
    cdef int64_t m_errors[2][3] 
    
    # Storage for TOTAL SENT pulses: [Intensity 0=Vac, 1=Dec, 2=Sig]
    cdef int64_t sent_counts[3]

    # Zero out arrays
    cdef int b, d
    for d in range(3):
        sent_counts[d] = 0
        for b in range(2):
            n_counts[b][d] = 0
            m_errors[b][d] = 0

    cdef uint8_t tx, rx, dec, tx_basis, rx_basis, is_error
    
    # --- MAIN LOOP ---
    for i in range(N):
        tx = tx_states[i]
        rx = rx_states[i]
        dec = decoy_levels[i]
        
        # Safety check for invalid decoy values
        if dec > 2: continue

        # 1. Count Total Sent Pulses (Crucial for Yield calculation)
        sent_counts[dec] += 1

        # 2. Determine Bases (0=Z, 1=X)
        if tx <= 1: tx_basis = 0
        else:       tx_basis = 1
            
        if rx <= 1: rx_basis = 0
        else:       rx_basis = 1
        
        # 3. Sifting (Only count if bases match)
        if tx_basis == rx_basis:
            n_counts[tx_basis][dec] += 1
            
            # Check for Errors
            is_error = 0
            if tx_basis == 0: # Z Basis
                if tx != rx: is_error = 1
            else: # X Basis
                # 3-state logic: Alice D(2) vs Bob A(3) is error
                if tx != rx: is_error = 1
            
            if is_error == 1:
                m_errors[tx_basis][dec] += 1
                
    # --- RETURN ---
    return {
        "n_Z": [int(n_counts[0][0]), int(n_counts[0][1]), int(n_counts[0][2])],
        "m_Z": [int(m_errors[0][0]), int(m_errors[0][1]), int(m_errors[0][2])],
        "n_X": [int(n_counts[1][0]), int(n_counts[1][1]), int(n_counts[1][2])],
        "m_X": [int(m_errors[1][0]), int(m_errors[1][1]), int(m_errors[1][2])],
        "sent": [int(sent_counts[0]), int(sent_counts[1]), int(sent_counts[2])]
    }