# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import scipy.sparse as sp # For handling sparse H matrix
import random
import math
import time
import sys
import io
import traceback # For detailed error printing
import json # For history store
import os # Added to make file path relative
from flask import Response # <<< Import Flask Response for the ping endpoint >>>

# --- Parameters for WiFi N=648, R=5/6 ---
EXPECTED_N = 648; EXPECTED_K = 540; EXPECTED_M = 108; EXPECTED_RATE = 5/6
IMAGE_WIDTH = 27; IMAGE_HEIGHT = 20
# Check dimensions at module level - if this fails, the app shouldn't start.
if IMAGE_WIDTH * IMAGE_HEIGHT != EXPECTED_K:
    print(f"FATAL Error: Image dimensions ({IMAGE_WIDTH}x{IMAGE_HEIGHT}={IMAGE_WIDTH*IMAGE_HEIGHT}) do not match Expected K ({EXPECTED_K}).")
    sys.exit(1) # Exit immediately if fundamental parameters are wrong

MAX_DECODER_ITER_ALG = 150; MAX_SLIDER_ITER = 50
ALIST_FILENAME = "wifi_648_r083.alist"

# --- Global variable ---
LDPC_PARAMS_GLOBAL = None

# --- ALIST Parser Function ---
def parse_alist_file(filename):
    ldpc_params = {}
    # Construct absolute path relative to this script file
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, filename)
    print(f"Attempting to read ALIST file from: {abs_file_path}")

    try:
        # Ensure the file exists before trying to open
        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"ALIST file not found at {abs_file_path}")

        with open(abs_file_path, 'r') as f: lines = f.readlines()
        n, m = map(int, lines[0].strip().split())
        max_col_weight_hdr, max_row_weight_hdr = map(int, lines[1].strip().split())
        vnode_deg_list = list(map(int, lines[2].strip().split()))
        cnode_deg_list = list(map(int, lines[3].strip().split()))
        if len(vnode_deg_list) != n or len(cnode_deg_list) != m: raise ValueError("Degree list lengths mismatch")
        rows_H = []; cols_H = []; line_offset = 4
        if len(lines) < line_offset + n: raise ValueError("ALIST file too short")
        var_node_conn_tmp = [[] for _ in range(n)]; chk_node_conn_tmp = [[] for _ in range(m)]
        for col_idx in range(n):
            line_num = line_offset + col_idx; parts = lines[line_num].strip().split()
            if not parts: continue
            check_nodes = list(map(int, parts)); check_nodes_0based = [cn - 1 for cn in check_nodes if cn > 0]
            var_node_conn_tmp[col_idx] = check_nodes_0based
            rows_H.extend(check_nodes_0based); cols_H.extend([col_idx] * len(check_nodes_0based))
            for cn_idx in check_nodes_0based:
                 if cn_idx < m: chk_node_conn_tmp[cn_idx].append(col_idx)
        data = np.ones(len(rows_H), dtype=int); H_coo = sp.coo_matrix((data, (rows_H, cols_H)), shape=(m, n)); H_csc = H_coo.tocsc()
        print(f"Parsed ALIST '{filename}': n={n}, m={m}. H shape={H_csc.shape}, nnz={H_csc.nnz}")
        if n != EXPECTED_N or m != EXPECTED_M: print(f"!!! WARNING: ALIST dimensions ({n}x{m}) != expected ({EXPECTED_N}x{EXPECTED_M}) !!!")
        var_neighbors, chk_neighbors = get_neighbors(H_csc) # Pre-calculate neighbors
        ldpc_params = { 'n': n, 'm': m, 'k': n - m, 'H': H_csc, 'var_neighbors': var_neighbors, 'chk_neighbors': chk_neighbors,
                        'max_vnode_deg': max_col_weight_hdr, 'max_cnode_deg': max_row_weight_hdr,
                        'vnode_deg_list': np.array(vnode_deg_list, dtype=int), 'cnode_deg_list': np.array(cnode_deg_list, dtype=int) }
        return ldpc_params
    except FileNotFoundError as fnf_err:
        print(f"Error: ALIST file '{filename}' not found. Details: {fnf_err}")
        return None
    except Exception as e:
        print(f"Error parsing ALIST file '{filename}': {e}")
        traceback.print_exc()
        return None

# --- Get Neighbors Function ---
def get_neighbors(H_sparse):
    if H_sparse is None: return [], [] # Handle case where H is None
    if not sp.issparse(H_sparse): H_sparse = sp.csc_matrix(H_sparse)
    m, n = H_sparse.shape; H_coo = H_sparse.tocoo()
    check_indices = H_coo.row; var_indices = H_coo.col
    var_neighbors = [[] for _ in range(n)]; chk_neighbors = [[] for _ in range(m)]
    for r, c in zip(check_indices, var_indices): var_neighbors[c].append(r); chk_neighbors[r].append(c)
    return var_neighbors, chk_neighbors

# --- Image Pattern Generator ---
def create_image_pattern(pattern_name, width, height):
    img = np.zeros((height, width), dtype=int)
    if pattern_name == 'square': pad_h, pad_w = height//4, width//4; h_start, h_end = pad_h, height-pad_h; w_start, w_end = pad_w, width-pad_w; img[h_start:h_end, w_start:w_end]=1 if h_start<h_end and w_start<w_end else 0; img[height//2,width//2]=1 if h_start>=h_end or w_start>=w_end else img[height//2,width//2]
    elif pattern_name == 'cross': center_h, center_w = height//2, width//2; thick = max(1, min(height,width)//5); h_s=max(0,center_h-thick//2); h_e=min(height,center_h+(thick+1)//2); w_s=max(0,center_w-thick//2); w_e=min(width,center_w+(thick+1)//2); img[h_s:h_e,:]=1; img[:,w_s:w_e]=1
    elif pattern_name == 'checkerboard': img = np.fromfunction(lambda r, c: (r+c)%2==0, (height, width), dtype=int)
    elif pattern_name == 'random_msg': img = np.random.randint(0, 2, size=(height, width))
    return img

# --- Systematic Encoder (Robust Version) ---
def encode_systematic_robust(message_bits_k, H):
    if H is None: print("Encode Error: H matrix is None."); return None
    try:
        if sp.issparse(H): H_dense = H.toarray()
        else: H_dense = H
    except MemoryError: print("Encode Error: H matrix too large to convert to dense array."); return None
    except Exception as e: print(f"Encode Error: Failed to prepare H matrix: {e}"); return None

    m, n = H_dense.shape; k = n - m
    if len(message_bits_k) != k:
        print(f"Encode Error: Message length {len(message_bits_k)} != k ({k})"); return None

    m_vec = np.array(message_bits_k, dtype=int); H_m = H_dense[:, :k]; H_p = H_dense[:, k:]
    if H_p.shape != (m,m): print(f"Encode Error: H_p shape {H_p.shape} != ({m}x{m})"); return None
    s = H_m @ m_vec.T % 2; Aug = np.hstack((H_p.astype(int), s.reshape(-1, 1)))
    pivot_row = 0
    try: # Forward elimination
        for col in range(m):
            if pivot_row >= m: break
            pivot = pivot_row
            while pivot < m and Aug[pivot, col] == 0:
                pivot += 1
            if pivot == m: continue # Skip column if no pivot found (might indicate singularity)
            Aug[[pivot_row, pivot], :] = Aug[[pivot, pivot_row], :] # Swap rows
            pivot_val = Aug[pivot_row, col] # Should be 1 after swap
            if pivot_val == 0: continue # Should not happen if pivot < m, but safety check
            # Use broadcasting for row operations
            rows_to_update = np.where(Aug[pivot_row + 1:, col] == 1)[0] + pivot_row + 1
            if rows_to_update.size > 0:
                 Aug[rows_to_update, :] = (Aug[rows_to_update, :] + Aug[pivot_row, :]) % 2
            pivot_row += 1

        rank = pivot_row # Check Rank
        if rank < m: print(f"Encode Error: H_p appears singular (rank={rank} < m={m}). Gaussian elimination failed."); return None

        # Back substitution
        for i in range(m - 1, -1, -1):
            # Find pivot column for row i (first '1' from the left)
            pivot_col_candidates = np.where(Aug[i, :m] == 1)[0]
            if pivot_col_candidates.size == 0:
                 # This case means the row is all zeros, should have been caught by rank check, but double-check.
                 if Aug[i, m] == 1: # Check the augmented part (syndrome)
                     print(f"Encode Error: Inconsistent system during back-substitution (Row {i})."); return None
                 else:
                     continue # Row of zeros is fine if augmented part is also 0

            pivot_col = pivot_col_candidates[0] # Use the first '1' as pivot

            # Eliminate '1's above the pivot
            rows_above_to_update = np.where(Aug[:i, pivot_col] == 1)[0]
            if rows_above_to_update.size > 0:
                # Add row i to rows above where pivot column is 1
                Aug[rows_above_to_update, :] = (Aug[rows_above_to_update, :] + Aug[i, :]) % 2

        # After back-substitution, the last column should contain the parity bits
        # This assumes H_p was reduced to identity, if not, need to solve for p
        # Check if the left part (H_p) became identity
        if not np.all(np.equal(Aug[:m, :m], np.identity(m))):
             print(f"Encode Warning: Gaussian elimination did not result in Identity matrix. Trying to solve.")
             # Fallback: Try solving H_p * p = s (less robust if H_p singular)
             try:
                 from scipy.linalg import solve_triangular # Use if H_p is triangular after forward elim
                 # Assuming forward elim made it upper triangular (check Aug[:m,:m])
                 # This part needs careful implementation based on actual GE result
                 print("Encode Warning: Fallback solver not fully implemented.")
                 parity_bits_p = Aug[:, m] # Placeholder if GE worked as expected
             except ImportError:
                 print("Encode Error: Need SciPy for fallback solver.")
                 return None
             except Exception as solve_e:
                 print(f"Encode Error: Fallback solver failed: {solve_e}"); return None
        else:
            parity_bits_p = Aug[:, m] # Parity bits are in the last column

        codeword_c = np.concatenate((m_vec, parity_bits_p)).astype(int)

        # Verification step
        final_syndrome = calculate_syndrome(H, codeword_c)
        if final_syndrome is None or np.any(final_syndrome != 0):
            print(f"Encode Verify Failed! Syndrome: {final_syndrome}"); return None
        return codeword_c
    except Exception as e:
        print(f"Gaussian elim error in encoder: {e}")
        traceback.print_exc()
        return None

# --- Other Helpers ---
def binary_symmetric_channel(codeword, p):
    noisy_codeword = codeword.copy(); flips = 0; indices_flipped = []
    error_pattern = np.random.rand(len(noisy_codeword)) < p
    flips = np.sum(error_pattern)
    indices_flipped = np.where(error_pattern)[0].tolist()
    noisy_codeword[error_pattern] = 1 - noisy_codeword[error_pattern]
    return noisy_codeword, flips, indices_flipped

def calculate_syndrome(H_input, codeword):
    if H_input is None or codeword is None: return None
    if H_input.shape[1] != len(codeword): return np.array([-1], dtype=int) # Indicate error
    codeword_int = np.array(codeword, dtype=int)
    try:
        if sp.issparse(H_input): syndrome = H_input.dot(codeword_int) % 2
        else: syndrome = H_input @ codeword_int % 2
        return syndrome
    except Exception as e:
        print(f"Error calculating syndrome: {e}")
        return None

def calculate_bsc_llrs(noisy_codeword, p):
    # Ensure p is within a safe range to avoid log(0) or division by zero
    p_stable = np.clip(p, 1e-9, 1.0 - 1e-9)
    # Calculate LLR for P(x=0|y) / P(x=1|y)
    # If y=0 (received 0), LLR = log( P(x=0|y=0) / P(x=1|y=0) ) = log( (1-p)/p )
    # If y=1 (received 1), LLR = log( P(x=0|y=1) / P(x=1|y=1) ) = log( p/(1-p) ) = -log( (1-p)/p )
    L0 = math.log( (1.0 - p_stable) / p_stable )
    # Efficiently apply based on received bit value
    llrs = np.where(noisy_codeword == 0, L0, -L0)
    # Clip LLRs to prevent extreme values which can cause numerical instability
    MAX_LLR_VALUE = 50
    llrs = np.clip(llrs, -MAX_LLR_VALUE, MAX_LLR_VALUE)
    return llrs

def safe_atanh(x):
    # Clip input to prevent infinity/NaN from values exactly at +/- 1
    clipped_x = np.clip(x, -0.99999999, 0.99999999)
    return np.arctanh(clipped_x)

# --- Custom LLR Belief Propagation Decoder ---
def decode_llrbp_for_dash(noisy_codeword, H, var_neighbors, chk_neighbors, max_iter, channel_prob):
    history = { 'iteration': [], 'total_llrs': [], 'decoded_bits': [], 'syndrome': [], 'syndrome_weight': [], 'status': "Starting"}
    if H is None or not var_neighbors or not chk_neighbors:
        history['status'] = "Error: Decoder called with invalid LDPC parameters."
        return history, noisy_codeword.copy(), 0

    n = len(noisy_codeword)
    m = H.shape[0]

    if H.shape[1] != n:
        history['status'] = "Error: H matrix columns != codeword length."
        return history, noisy_codeword.copy(), 0

    # --- Initialization ---
    intrinsic_llrs = calculate_bsc_llrs(noisy_codeword, channel_prob)
    msg_c2v = np.zeros((m, n)) # Messages from Check Nodes to Variable Nodes
    total_llrs = intrinsic_llrs.copy()
    current_decoded_codeword = (total_llrs < 0).astype(int) # Initial decision based on channel LLRs

    # Record initial state (Iteration 0)
    history['iteration'].append(0)
    history['total_llrs'].append(total_llrs.tolist())
    history['decoded_bits'].append(current_decoded_codeword.tolist())
    initial_syndrome = calculate_syndrome(H, current_decoded_codeword)
    if initial_syndrome is None:
        history['status'] = "Error calculating initial syndrome."
        return history, noisy_codeword.copy(), 0
    history['syndrome'].append(initial_syndrome.tolist())
    history['syndrome_weight'].append(np.sum(initial_syndrome))

    # Check if received word is already a codeword
    noisy_syndrome = calculate_syndrome(H, noisy_codeword)
    if noisy_syndrome is not None and np.sum(noisy_syndrome) == 0:
        history['status'] = "Converged (Input is Codeword)"
        history['iteration'].append(0) # Re-record state 0 as final
        history['total_llrs'][-1] = intrinsic_llrs.tolist() # Use intrinsic as total
        history['decoded_bits'][-1] = noisy_codeword.tolist()
        history['syndrome'][-1] = noisy_syndrome.tolist()
        history['syndrome_weight'][-1] = 0
        return history, noisy_codeword.copy(), 0

    converged = False
    stalled = False
    iters_done = 0

    # --- Iteration Loop ---
    for iteration in range(1, max_iter + 1):
        iters_done = iteration
        history['status'] = f"Running Iter {iters_done}"
        prev_decoded_codeword = current_decoded_codeword.copy()

        # --- Variable Node Update (V2C messages) ---
        msg_v2c = np.zeros((m, n))
        extrinsic_llrs = np.zeros(n)
        for v in range(n):
            connected_checks = var_neighbors[v]
            # Sum incoming C2V messages for this variable node
            incoming_c2v_sum = np.sum(msg_c2v[connected_checks, v])
            # Calculate the message from v to each check c
            for c in connected_checks:
                 # Message is intrinsic + sum of all *other* C2V messages
                 msg_v2c[c, v] = intrinsic_llrs[v] + incoming_c2v_sum - msg_c2v[c, v]

        # --- Check Node Update (C2V messages) ---
        # Clip tanh inputs to avoid instability near +/- 1
        tanh_half_v2c = np.tanh(np.clip(msg_v2c / 2.0, -20, 20))
        new_msg_c2v = np.zeros((m, n))

        for c in range(m):
            connected_vars = chk_neighbors[c]
            if not connected_vars: continue # Skip if check node has no connections

            # Get the tanh messages for variables connected to this check
            tanh_messages_for_c = tanh_half_v2c[c, connected_vars]

            # Calculate product of tanh messages efficiently
            # Using log-sum-exp trick for products can be more stable but more complex.
            # Direct product is often fine if inputs are clipped.
            # Handle potential zeros in product:
            log_tanh_abs = np.log(np.abs(tanh_messages_for_c) + 1e-20) # Add epsilon to avoid log(0)
            sum_log_tanh = np.sum(log_tanh_abs)
            prod_signs = np.prod(np.sign(tanh_messages_for_c))

            for v_idx, v in enumerate(connected_vars):
                 # Product of all *other* tanh messages
                 # Avoid division by zero if tanh_messages_for_c[v_idx] is near zero
                 if abs(tanh_messages_for_c[v_idx]) < 1e-9:
                      # If one message is zero, the product excluding it is needed. Calculate manually.
                      other_tanh_prod = np.prod(np.delete(tanh_messages_for_c, v_idx))
                 else:
                      # More stable way: total_prod / current_tanh
                      # Using logs: exp( sum_log_tanh - log_tanh_abs[v_idx] )
                      prod_others_abs = np.exp(sum_log_tanh - log_tanh_abs[v_idx])
                      # Determine sign: overall sign / current sign
                      sign_others = prod_signs * np.sign(tanh_messages_for_c[v_idx])
                      other_tanh_prod = sign_others * prod_others_abs

                 # Apply safe_atanh after clipping the product
                 new_msg_c2v[c, v] = 2 * safe_atanh(other_tanh_prod)

        msg_c2v = np.clip(new_msg_c2v, -MAX_LLR_VALUE, MAX_LLR_VALUE) # Clip outgoing C2V messages

        # --- Update Total LLRs and Decode ---
        total_llrs = intrinsic_llrs.copy()
        for v in range(n):
            total_llrs[v] += np.sum(msg_c2v[var_neighbors[v], v])

        # Clip final LLRs before making decision
        total_llrs = np.clip(total_llrs, -MAX_LLR_VALUE, MAX_LLR_VALUE)
        current_decoded_codeword = (total_llrs < 0).astype(int) # Hard decision

        # --- Check Syndrome and Convergence ---
        syndrome = calculate_syndrome(H, current_decoded_codeword)
        if syndrome is None:
            history['status'] = f"Error calculating syndrome at iter {iters_done}."
            # Return the last known good state or the noisy codeword
            last_good_cw = history['decoded_bits'][-1] if history['decoded_bits'] else noisy_codeword
            return history, np.array(last_good_cw), iters_done -1

        syndrome_weight = np.sum(syndrome)

        # --- Record History ---
        history['iteration'].append(iters_done)
        history['total_llrs'].append(total_llrs.tolist())
        history['decoded_bits'].append(current_decoded_codeword.tolist())
        history['syndrome'].append(syndrome.tolist())
        history['syndrome_weight'].append(syndrome_weight)

        # --- Check Termination Conditions ---
        if syndrome_weight == 0:
            history['status'] = f"Converged iter {iters_done}."
            converged = True
            break # Converged

        if iteration > 1 and np.array_equal(current_decoded_codeword, prev_decoded_codeword):
             # Check if syndrome also stalled? Sometimes codeword stalls but syndrome changes.
             # More robust stall check might involve LLR stability.
             # For simplicity, we stick to codeword stalling.
             history['status'] = f"Stalled iter {iters_done}."
             stalled = True
             break # Stalled

    # --- After Loop ---
    if not converged and not stalled and iters_done >= max_iter:
        final_syndrome_weight = history['syndrome_weight'][-1] if history['syndrome_weight'] else -1
        history['status'] = f"Max iters ({max_iter}) reached. Final synd weight: {final_syndrome_weight}."

    # Return the last computed codeword and the full history
    return history, current_decoded_codeword, iters_done

# --- Image Figure Helper ---
def create_image_fig(image_array, title="Image"):
    if image_array is None or image_array.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (No Data)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            template='plotly_white',
             margin=dict(l=5, r=5, t=30, b=5) # Adjust title margin
            )
        return fig

    img_numeric = image_array.astype(float)
    fig = px.imshow(img_numeric, binary_string=False, color_continuous_scale='gray_r', aspect='equal')
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        coloraxis_showscale=False,
        margin=dict(l=5, r=5, t=30, b=5) # Adjust title margin
    )
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>")
    return fig

# --- H Matrix Plot Helper ---
def create_h_matrix_fig(H_sparse):
    if H_sparse is None or not sp.issparse(H_sparse) or H_sparse.nnz == 0:
        fig = go.Figure()
        fig.update_layout(
            title="H Matrix (No Data / Load Error)",
            xaxis_visible=False,
            yaxis_visible=False,
            template='plotly_white',
             margin=dict(l=10, r=10, t=30, b=10)
            )
        return fig
    try:
        # Check size before attempting to plot directly
        m, n = H_sparse.shape
        nnz = H_sparse.nnz
        # Heuristic: Avoid plotting extremely large/dense matrices directly
        # Adjust these limits based on typical browser/Plotly performance
        MAX_PLOT_ELEMENTS = 5_000_000
        MAX_PLOT_NNZ = 500_000
        if m * n > MAX_PLOT_ELEMENTS or nnz > MAX_PLOT_NNZ:
             fig = go.Figure()
             fig.update_layout(
                 title=f"H Matrix ({m}x{n}, {nnz} non-zeros) - Too large to plot directly",
                 xaxis_visible=False, yaxis_visible=False, template='plotly_white',
                 margin=dict(l=10, r=10, t=30, b=10)
             )
             return fig

        # Use scatter plot for sparse matrix visualization if small enough
        H_coo = H_sparse.tocoo()
        fig = go.Figure(data=go.Scattergl(
            x=H_coo.col,
            y=H_coo.row,
            mode='markers',
            marker=dict(color='black', size=max(1, 8 - np.log10(nnz+1)) ), # Adjust size based on nnz
            hoverinfo='skip' # Skip hover for individual points if too many
        ))
        fig.update_layout(
            title=f"H Matrix ({m}x{n}, {nnz} non-zeros)",
            xaxis_title="Variable Nodes (Columns)",
            yaxis_title="Check Nodes (Rows)",
            xaxis=dict(range=[-0.5, n-0.5], showgrid=False, zeroline=False, tickvals=[], ticktext=[]),
            yaxis=dict(range=[m-0.5, -0.5], autorange=False, showgrid=False, zeroline=False, tickvals=[], ticktext=[]), # Invert y-axis to match imshow
            margin=dict(l=10, r=10, t=30, b=10),
            template='plotly_white',
            hovermode='closest' # Enable hover for axes/general area
        )
        # Update axes to show node indices on hover if desired (can be slow)
        # fig.update_xaxes(side="top")
        # fig.update_traces(hovertemplate="Var Node (x): %{x}<br>Check Node (y): %{y}<extra></extra>")

    except MemoryError:
        fig = go.Figure().update_layout(title="H Matrix (Memory Error during Plotting)", xaxis_visible=False, yaxis_visible=False)
    except Exception as e:
        fig = go.Figure().update_layout(title=f"H Matrix Plot Error: {e}", xaxis_visible=False, yaxis_visible=False)

    return fig


# ==============================================================================
# <<< GLOBAL CODE EXECUTION on Import/Startup >>>
# ==============================================================================

print("-" * 50)
print("Script starting / Loading module...")

# --- Attempt to parse ALIST file globally ---
print(f"Attempting to parse ALIST file '{ALIST_FILENAME}' globally...")
try:
    LDPC_PARAMS_GLOBAL = parse_alist_file(ALIST_FILENAME)
    if LDPC_PARAMS_GLOBAL is None:
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"WARNING: Failed to load or parse '{ALIST_FILENAME}' during startup.")
         print("The application might not function correctly.")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         # Keep LDPC_PARAMS_GLOBAL as None
    else:
         print(f"Successfully loaded LDPC parameters from '{ALIST_FILENAME}'.")
         # Optional: Perform checks on loaded params if needed
         if LDPC_PARAMS_GLOBAL['k'] != EXPECTED_K:
              print(f"WARNING: Loaded K ({LDPC_PARAMS_GLOBAL['k']}) != Expected K ({EXPECTED_K})")

except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL ERROR during initial ALIST parsing: {e}")
    traceback.print_exc()
    print("LDPC parameters are NOT loaded. App will likely fail.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    LDPC_PARAMS_GLOBAL = None # Ensure it's None on critical error

# --- Initialize Dash App ---
# Pass `meta_tags` to ensure responsiveness on different devices
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server # Expose server variable for Gunicorn

# --- >>> ADD FLASK PING ENDPOINT <<< ---
@server.route('/ping')
def ping():
    """A simple endpoint to keep the Render server alive."""
    print("Received /ping request") # Optional: Log ping requests
    return Response("pong", status=200)
# --- >>> END PING ENDPOINT <<< ---

# --- Create Initial Figures (Handle potential None LDPC_PARAMS_GLOBAL) ---
print("Creating initial figures for layout...")
initial_h_fig = create_h_matrix_fig(LDPC_PARAMS_GLOBAL['H'] if LDPC_PARAMS_GLOBAL else None)
no_data_fig_layout = create_image_fig(None, "No Data")
no_plot_fig_layout = go.Figure().update_layout(title="(No data)", xaxis_visible=False, yaxis_visible=False)

# --- App Layout Definition ---
# Uses the initial figures created above
print("Defining Dash app layout...")
app.layout = dbc.Container([
    dcc.Store(id='history-store'),
    dcc.Store(id='orig-image-store'),
    dbc.Row(dbc.Col(html.H1("LDPC Decoder Demo (WiFi 802.11n)"), width=12)),
     dbc.Row([
         # --- Controls Column ---
         dbc.Col([
             dbc.Card([ dbc.CardHeader("Controls"), dbc.CardBody([
                    html.Label("Image Pattern:", className="fw-bold"),
                    dcc.Dropdown( id='pattern-dropdown', options=[ {'label': 'Square', 'value': 'square'}, {'label': 'Cross', 'value': 'cross'}, {'label': 'Checkerboard', 'value': 'checkerboard'}, {'label': 'Random Message', 'value': 'random_msg'}, ], value='checkerboard', clearable=False ), html.Br(),

                    html.Label("Channel Noise Probability (p):", className="fw-bold"),
                    dcc.Slider(id='noise-slider', min=0.0, max=0.05, step=0.001, value=0.01, marks={i/100: f'{i/100:.3f}' for i in range(0, 6, 1)}, tooltip={"placement": "bottom", "always_visible": True}), html.Br(), # Increase precision in marks

                    html.Label("Max Decoder Iterations:", className="fw-bold"),
                    dcc.Slider(id='max-iter-slider', min=10, max=MAX_DECODER_ITER_ALG + 50, step=10, value=MAX_DECODER_ITER_ALG, marks={i: str(i) for i in range(0, MAX_DECODER_ITER_ALG + 51, 50)}, tooltip={"placement": "bottom", "always_visible": True}), html.Br(), # Adjust marks

                    html.Label("Random Seed (optional):", className="fw-bold"),
                    dbc.Input(id='seed-input', type='number', placeholder="Leave blank for random", step=1, min=0), html.Br(), # Add min=0

                    dbc.Button("Run Simulation", id="run-button", n_clicks=0, color="primary", className="mt-3 w-100"), # Make button full width
                ])]),
             dbc.Card([dbc.CardHeader("Run Info"), dbc.CardBody(dbc.Spinner(html.Pre(id="run-summary", style={'maxHeight': '300px', 'overflowY': 'scroll', 'fontSize': 'small'})))], className="mt-3"),
        ], md=4, className="mb-3"), # Add bottom margin to column

        # --- Displays Column ---
        dbc.Col([
             dbc.Tabs([
                 dbc.Tab(label="H Matrix", children=[
                     # Use the pre-generated figure here
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='h-matrix-plot', figure=initial_h_fig))), className="mt-2")
                 ]),
                 dbc.Tab(label="Syndrome Weight", children=[
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='syndrome-plot', figure=no_plot_fig_layout))), className="mt-2")
                 ]),
                 dbc.Tab(label="LLR Evolution", children=[
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='llr-plot', figure=no_plot_fig_layout))), className="mt-2")
                 ]),
             ]),
             html.Hr(),
             dbc.Row([
                 dbc.Col(dbc.Card([dbc.CardHeader(f"Original Image ({IMAGE_WIDTH}x{IMAGE_HEIGHT})"), dbc.CardBody(dbc.Spinner(dcc.Graph(id='img-original', figure=no_data_fig_layout)))]), md=4, className="mb-2"), # Add bottom margin
                 dbc.Col(dbc.Card([dbc.CardHeader("Received Noisy"),
                                   dbc.CardBody([dbc.Spinner(dcc.Graph(id='img-noisy', figure=no_data_fig_layout)),
                                                 html.Pre(id='noisy-info-text', style={'fontSize':'small','textAlign':'center', 'marginTop':'5px'})
                                                ])]), md=4, className="mb-2"), # Add bottom margin
                 dbc.Col(dbc.Card([dbc.CardHeader("LDPC Decoded"),
                                   dbc.CardBody([dbc.Spinner(dcc.Graph(id='img-decoded', figure=no_data_fig_layout)),
                                                 html.Pre(id='decoded-info-text', style={'fontSize':'small','textAlign':'center', 'marginTop':'5px'})
                                                ])]), md=4, className="mb-2"), # Add bottom margin
            ]),
            html.Hr(),
            dbc.Row([
                 dbc.Col([
                     html.Label("View Decoder Iteration:", style={'fontWeight':'bold'}),
                     dcc.Slider( id='iteration-slider', min=0, max=MAX_SLIDER_ITER, step=1, value=0, marks={i: str(i) for i in range(0, MAX_SLIDER_ITER + 1, 5)}, disabled=True, tooltip={"placement": "bottom", "always_visible": True} ),
                     html.P(id='iteration-display-label', style={'textAlign':'center', 'marginTop':'5px', 'fontSize':'small'})
                 ])
            ]),
        ], md=8)
    ]),
], fluid=True) # Use fluid container for better responsiveness
print("Dash app layout defined.")

# ==============================================================================
# <<< DASH CALLBACKS >>>
# ==============================================================================
print("Defining Dash callbacks...")

# --- Main Simulation Callback ---
@app.callback(
    [Output('img-original', 'figure'),
     Output('img-noisy', 'figure'),
     Output('h-matrix-plot', 'figure'), # Output H plot (might be updated if params change, though unlikely here)
     Output('syndrome-plot', 'figure'),
     Output('llr-plot', 'figure'),
     Output('run-summary', 'children'),
     Output('history-store', 'data'),
     Output('iteration-slider', 'value'),
     Output('iteration-slider', 'marks'),
     Output('iteration-slider', 'max'), # Update slider max based on iterations
     Output('iteration-slider', 'disabled'),
     Output('orig-image-store', 'data'),
     Output('noisy-info-text', 'children') ],
    [Input('run-button', 'n_clicks')],
    [State('pattern-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('max-iter-slider', 'value'),
     State('seed-input', 'value')],
    prevent_initial_call=True # Prevent callback firing on page load before button click
)
def update_simulation(n_clicks, pattern, noise_prob, max_iter_alg, seed_val):
    global LDPC_PARAMS_GLOBAL

    start_time_sim = time.time()
    print(f"\n--- Run Button Clicked (n_clicks={n_clicks}) ---")
    print(f"Params: pattern='{pattern}', p={noise_prob}, max_iter={max_iter_alg}, seed={seed_val}")

    # --- Initial State Handling & Parameter Check ---
    default_slider_marks = {i: str(i) for i in range(0, MAX_SLIDER_ITER + 1, 5)}
    default_noisy_info = "Syndrome W: N/A | Img Errors: N/A"
    no_data_fig = create_image_fig(None, "No Data")
    no_plot_fig = go.Figure().update_layout(title="(No data)", xaxis_visible=False, yaxis_visible=False)

    # Crucial Check: Ensure LDPC params were loaded on startup
    if LDPC_PARAMS_GLOBAL is None:
         print("ERROR in callback: LDPC_PARAMS_GLOBAL is None. Cannot run simulation.")
         error_msg = "FATAL ERROR:\nLDPC parameters failed to load on application startup.\nPlease check server logs and the ALIST file.\nApplication cannot function."
         h_fig_error = create_h_matrix_fig(None) # Show error state for H plot too
         return (no_data_fig, no_data_fig, h_fig_error, no_plot_fig, no_plot_fig,
                 error_msg, None, 0, default_slider_marks, MAX_SLIDER_ITER, True, None, default_noisy_info)

    # Use the globally loaded parameters
    ldpc_params = LDPC_PARAMS_GLOBAL
    n_val = ldpc_params['n']
    k_val = ldpc_params['k']
    m_val = ldpc_params['m']
    H_csc = ldpc_params['H']
    var_neighbors = ldpc_params['var_neighbors']
    chk_neighbors = ldpc_params['chk_neighbors']
    current_h_fig = create_h_matrix_fig(H_csc) # Generate current H fig (should match initial)

    # --- Simulation Setup ---
    summary_lines = [f"--- Run {n_clicks} ---"]

    # Seed
    seed_to_use = None
    if seed_val is not None:
        try:
            seed_to_use = int(seed_val)
            random.seed(seed_to_use)
            np.random.seed(seed_to_use)
            summary_lines.append(f"Using Seed: {seed_to_use}")
        except ValueError:
            summary_lines.append("Using Seed: Random (Invalid input)")
            random.seed() # Use system time
            np.random.seed()
    else:
        summary_lines.append("Using Seed: Random")
        random.seed()
        np.random.seed()

    actual_rate = k_val / n_val if n_val > 0 else 0
    summary_lines.append(f"Code: N={n_val}, K={k_val}, M={m_val}, R={actual_rate:.3f}")
    summary_lines.append(f"Source: '{ALIST_FILENAME}'")
    summary_lines.append(f"Image: '{pattern}' ({IMAGE_WIDTH}x{IMAGE_HEIGHT})")

    # --- Generate Message & Encode ---
    print("Generating image pattern and encoding...")
    img_orig_array = create_image_pattern(pattern, IMAGE_WIDTH, IMAGE_HEIGHT)
    message_bits_k = img_orig_array.flatten()
    fig_orig = create_image_fig(img_orig_array, "Original")
    orig_image_list = img_orig_array.tolist() # Store original image

    summary_lines.append(f"\nEncoding K={k_val} message bits...")
    encode_start_time = time.time()
    codeword_c = encode_systematic_robust(message_bits_k, H_csc)
    encode_time = time.time() - encode_start_time
    summary_lines.append(f"   Encode time: {encode_time:.4f}s")

    if codeword_c is None:
        summary_lines.append(f"\nFATAL ERROR: Systematic encoding failed.")
        summary_lines.append("   (Check H matrix structure, especially H_p part)")
        err_fig=create_image_fig(None, "Encoding Error")
        return (fig_orig, err_fig, current_h_fig, no_plot_fig, no_plot_fig,
                "\n".join(summary_lines), None, 0, default_slider_marks, MAX_SLIDER_ITER, True, orig_image_list, default_noisy_info)
    summary_lines.append(f"   Encoding successful (N={len(codeword_c)} bits).")

    # --- Noise & Decoding ---
    summary_lines.append(f"\nApplying Noise & Decoding...")
    summary_lines.append(f"   Channel p = {noise_prob:.4f}")
    summary_lines.append(f"   Max Alg Iter = {max_iter_alg}")
    noisy_cw, flips, flipped_idx = binary_symmetric_channel(codeword_c, noise_prob)
    summary_lines.append(f"   Channel flips introduced: {flips}")
    if 0 < flips <= 20: summary_lines.append(f"   Flip indices (0-based): {flipped_idx}")
    elif flips > 20: summary_lines.append(f"   (Too many flips to list indices)")

    # Extract noisy image part
    noisy_msg_bits = noisy_cw[:k_val];
    img_noisy_arr = noisy_msg_bits.reshape((IMAGE_HEIGHT, IMAGE_WIDTH));
    fig_noisy = create_image_fig(img_noisy_arr, "Received Noisy")
    err_noisy_img_before_decode = np.sum(img_orig_array != img_noisy_arr);
    noisy_syndrome = calculate_syndrome(H_csc, noisy_cw)
    noisy_syndrome_weight = np.sum(noisy_syndrome) if noisy_syndrome is not None else -1
    noisy_info_str = f"Syndr. W: {noisy_syndrome_weight} | Img Errors: {err_noisy_img_before_decode}"
    summary_lines.append(f"   Image errors BEFORE decoding: {err_noisy_img_before_decode}")
    summary_lines.append(f"   Syndrome weight BEFORE decoding: {noisy_syndrome_weight}")

    # Call Decoder
    print("Running LLR-BP decoder...")
    decode_start_time = time.time()
    history, final_decoded_cw, iters_done = decode_llrbp_for_dash(noisy_cw, H_csc, var_neighbors, chk_neighbors, max_iter_alg, noise_prob)
    decode_time = time.time() - decode_start_time
    summary_lines.append(f"\nDecoder Run:")
    summary_lines.append(f"   Decoder time: {decode_time:.4f}s")
    summary_lines.append(f"   Iterations performed: {iters_done}")
    summary_lines.append(f"   Final Status: {history['status']}")
    final_syndrome_weight = history['syndrome_weight'][-1] if history['syndrome_weight'] else -1
    summary_lines.append(f"   Final syndrome weight: {final_syndrome_weight}")

    # Prepare history for storage
    history_for_store = {'iteration': history.get('iteration', [0]),
                         'decoded_bits': history.get('decoded_bits', [codeword_c.tolist()]),
                         'syndrome_weight': history.get('syndrome_weight', [0]),
                         'total_llrs': history.get('total_llrs', [[]])}

    # --- Final Stats & Results ---
    print("Calculating final statistics...")
    if final_decoded_cw is None or len(final_decoded_cw) != n_val:
        summary_lines.append("\nERROR: Decoder returned invalid codeword. Using noisy codeword for stats.")
        final_decoded_cw = noisy_cw.copy() # Fallback

    decoded_msg_bits_final = final_decoded_cw[:k_val]
    if len(decoded_msg_bits_final) != k_val:
        summary_lines.append(f"\nFATAL ERROR: Final decoded message length {len(decoded_msg_bits_final)} != k {k_val}")
        err_fig = create_image_fig(None, "Decode Error")
        return (fig_orig, fig_noisy, err_fig, current_h_fig, no_plot_fig, no_plot_fig,
                "\n".join(summary_lines), None, 0, default_slider_marks, MAX_SLIDER_ITER, True, orig_image_list, noisy_info_str)

    img_decoded_arr_final = decoded_msg_bits_final.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    err_remain_img = np.sum(img_orig_array != img_decoded_arr_final);
    cw_err_remain = np.sum(codeword_c != final_decoded_cw);
    summary_lines.append(f"\nResults:")
    summary_lines.append(f"   Image errors AFTER decoding: {err_remain_img}")
    summary_lines.append(f"   Codeword errors AFTER decoding: {cw_err_remain}")

    # Result message
    if final_syndrome_weight == 0 and cw_err_remain == 0:
        result_msg = "Result: SUCCESS! (Converged to original codeword)"
    elif final_syndrome_weight == 0 and cw_err_remain > 0:
        result_msg = "Result: Partial Success (Converged to wrong codeword)"
    elif final_syndrome_weight != 0 and err_remain_img < err_noisy_img_before_decode:
         result_msg = "Result: Partial Correction (Failed convergence but reduced errors)"
    elif final_syndrome_weight != 0:
         result_msg = "Result: FAILURE (Failed convergence)"
    else:
         result_msg = "Result: Unknown State" # Should not happen
    summary_lines.append(result_msg)
    elapsed_time_sim = time.time() - start_time_sim
    summary_lines.append(f"\nTotal Simulation Time: {elapsed_time_sim:.3f} seconds")

    # --- Generate Diagnostic Plots ---
    print("Generating diagnostic plots...")
    # Syndrome Plot
    fig_syndrome = go.Figure()
    iterations_synd = history.get('iteration', [])
    syndrome_weights = history.get('syndrome_weight', [])
    if iterations_synd and len(iterations_synd) == len(syndrome_weights):
        fig_syndrome.add_trace(go.Scattergl(x=iterations_synd, y=syndrome_weights, mode='lines+markers', name='Syndrome Weight'))
        fig_syndrome.update_layout(title="Syndrome Weight vs Iteration", xaxis_title="Decoder Iteration", yaxis_title="Syndrome Weight", yaxis=dict(range=[-0.5, max(10, m_val * 0.1)]), uirevision=n_clicks) # Adjust y range dynamically
    else: fig_syndrome.update_layout(title="Syndrome Weight Plot (No data)")

    # LLR Plot
    fig_llr = go.Figure()
    iterations_llr = history.get('iteration', [])
    llrs_history_list = history.get('total_llrs', [])
    if iterations_llr and llrs_history_list and len(iterations_llr) == len(llrs_history_list):
         try:
             llrs_history_array = np.array(llrs_history_list)
             if llrs_history_array.ndim == 2 and llrs_history_array.shape[1] == n_val:
                 # Plot average LLR for bits that end up 0 vs 1
                 final_bits = (llrs_history_array[-1, :] < 0).astype(int)
                 indices_0 = np.where(final_bits == 0)[0]
                 indices_1 = np.where(final_bits == 1)[0]

                 avg_llr_0 = np.mean(llrs_history_array[:, indices_0], axis=1) if len(indices_0) > 0 else np.full(len(iterations_llr), np.nan)
                 avg_llr_1 = np.mean(llrs_history_array[:, indices_1], axis=1) if len(indices_1) > 0 else np.full(len(iterations_llr), np.nan)

                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_llr_0, mode='lines', name=f'Avg LLR (Final=0, {len(indices_0)} bits)', line=dict(color='blue', width=2)))
                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_llr_1, mode='lines', name=f'Avg LLR (Final=1, {len(indices_1)} bits)', line=dict(color='red', width=2)))

                 # Optionally plot a few individual LLRs (e.g., ones with smallest final magnitude)
                 final_llr_magnitudes = np.abs(llrs_history_array[-1, :])
                 num_to_plot = 5 # Plot a small number of individual LLRs
                 indices_to_plot_individually = np.argsort(final_llr_magnitudes)[:num_to_plot]

                 for bit_index in indices_to_plot_individually:
                      final_val = final_bits[bit_index]
                      color = 'lightblue' if final_val == 0 else 'salmon'
                      fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=llrs_history_array[:, bit_index], mode='lines', name=f'Bit {bit_index} (Final={final_val})', line=dict(width=1, dash='dot', color=color), opacity=0.8))

                 plot_title = f"LLR Evolution (Averages & {num_to_plot} Weakest)"
                 fig_llr.update_layout(title=plot_title, xaxis_title="Decoder Iteration", yaxis_title="Total LLR", uirevision=n_clicks)
             else:
                 print("LLR history array shape mismatch.")
                 fig_llr.update_layout(title="LLR Plot (Data Shape Error)", xaxis_visible=False, yaxis_visible=False)
         except Exception as e:
             print(f"Error creating LLR plot: {e}")
             traceback.print_exc()
             fig_llr.update_layout(title="LLR Plot (Error)", xaxis_visible=False, yaxis_visible=False)
    else:
        fig_llr.update_layout(title="LLR Plot (No Data)", xaxis_visible=False, yaxis_visible=False)

    # --- Prepare Iteration Slider Outputs ---
    # The actual iterations run by the algorithm
    actual_iters_run = iters_done
    # The maximum value the slider should show (don't exceed MAX_SLIDER_ITER visually)
    slider_max_display = min(actual_iters_run, MAX_SLIDER_ITER)
    # The default value to set the slider to (usually the last iteration shown)
    slider_value = slider_max_display
    # Create marks for the slider
    mark_step = max(1, (slider_max_display + 1) // 10, 5) # Dynamic step for marks
    if slider_max_display > 0:
         slider_marks = {i: str(i) for i in range(0, slider_max_display + 1, mark_step)}
         if slider_max_display not in slider_marks: # Ensure last iteration is a mark
             slider_marks[slider_max_display] = str(slider_max_display)
         slider_marks = dict(sorted(slider_marks.items()))
         slider_disabled = False
    else: # If 0 iterations, disable slider
        slider_marks = {0: '0'}
        slider_disabled = True

    print("Callback finished.")
    return (fig_orig, fig_noisy, current_h_fig, fig_syndrome, fig_llr,
            "\n".join(summary_lines), history_for_store,
            slider_value, slider_marks, slider_max_display, slider_disabled,
            orig_image_list, noisy_info_str)


# --- Callback for Iteration Slider ---
@app.callback(
    [Output('img-decoded', 'figure'),
     Output('iteration-display-label', 'children'),
     Output('decoded-info-text', 'children')],
    [Input('iteration-slider', 'value')], # Use 'value' directly, drag_value can be laggy
    [State('history-store', 'data'),
     State('orig-image-store', 'data')],
    prevent_initial_call=True # Prevent initial call
)
def update_displayed_iteration(selected_iter_val, history_data, orig_image_list_data):
    # triggered = callback_context.triggered_id # Find what triggered the callback
    # print(f"Slider Callback Triggered by: {triggered}, Value: {selected_iter_val}")

    default_info_text = "Syndrome W: N/A | Img Errors: N/A"
    no_data_label = f"Iteration: {selected_iter_val} (No History Data)"
    no_data_fig = create_image_fig(None, "Decoded (No Data)")

    if (selected_iter_val is None or history_data is None or orig_image_list_data is None
            or not history_data.get('iteration') # Check if lists exist and are not empty
            or not history_data.get('decoded_bits')
            or not history_data.get('syndrome_weight')):
        print("Slider callback: Missing data.")
        return no_data_fig, no_data_label, default_info_text

    iterations_list = history_data.get('iteration', [])
    decoded_bits_history = history_data.get('decoded_bits', [])
    syndrome_weight_history = history_data.get('syndrome_weight', [])

    # Map slider value to the index in the history arrays
    # Slider value corresponds directly to the desired iteration number
    # Find the closest index <= selected_iter_val
    possible_indices = [i for i, iter_num in enumerate(iterations_list) if iter_num <= selected_iter_val]
    if not possible_indices:
         iter_index = 0 # Default to the first entry (iter 0) if slider is before first iter
    else:
         iter_index = max(possible_indices) # Get the index for the latest iteration <= slider value


    # Safety check for index bounds
    if iter_index < 0 or iter_index >= len(decoded_bits_history):
        print(f"Slider callback: Index {iter_index} out of bounds for history length {len(decoded_bits_history)}.")
        return no_data_fig, f"Iteration: {selected_iter_val} (History Index Error)", default_info_text

    selected_iter_actual = iterations_list[iter_index]
    label_text = f"Showing Iteration: {selected_iter_actual} / {iterations_list[-1]}" # Show current / total iters

    decoded_bits_list = decoded_bits_history[iter_index]
    syndrome_weight_iter = syndrome_weight_history[iter_index]

    decoded_cw_iter = np.array(decoded_bits_list, dtype=int)
    # Use EXPECTED_K directly as k_val should be consistent
    k_val = EXPECTED_K
    decoded_message_bits_iter = decoded_cw_iter[:k_val]

    if len(decoded_message_bits_iter) != IMAGE_WIDTH * IMAGE_HEIGHT:
        print(f"Slider callback: Decoded message length error at index {iter_index}.")
        return create_image_fig(None, "Decoded (Dim Error)"), label_text, default_info_text

    # Reshape and create figure
    img_decoded_array_iter = decoded_message_bits_iter.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    fig_decoded_iter = create_image_fig(img_decoded_array_iter, f"Decoded (Iter {selected_iter_actual})")

    # Calculate errors compared to original
    try:
        img_orig_array = np.array(orig_image_list_data, dtype=int)
        errors_iter = np.sum(img_orig_array != img_decoded_array_iter)
        info_text = f"Syndr. W: {syndrome_weight_iter} | Img Errors: {errors_iter}"
    except Exception as e:
        print(f"Slider callback: Error comparing images: {e}")
        info_text = f"Syndr. W: {syndrome_weight_iter} | Img Errors: Error"

    return fig_decoded_iter, label_text, info_text


print("Dash callbacks defined.")
print("-" * 50)
# ==============================================================================
# <<< MAIN EXECUTION BLOCK (for Local Development ONLY) >>>
# ==============================================================================
if __name__ == '__main__':
    print("Running script directly (__name__ == '__main__')")
    # The ALIST parsing and global setup happens *above* this block when the script is loaded.
    # We just need to check if the parsing was successful before trying to run locally.
    if LDPC_PARAMS_GLOBAL is None:
        print("\nFATAL ERROR: LDPC parameters failed to load during script startup.")
        print("Cannot start local development server.")
        print("Please check the ALIST file path and parsing logic.")
        sys.exit(1) # Exit if failed for local run

    print(f"\nStarting Dash development server locally...")
    print(f"Access at: http://127.0.0.1:8050/")
    print("-" * 50)
    # Set debug=False for a production-like test, or True for development features (like hot-reloading)
    # Use host='0.0.0.0' to make it accessible on your local network if needed
    app.run(debug=False, port=8050, host='127.0.0.1')