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

# --- Parameters for WiFi N=648, R=5/6 ---
EXPECTED_N = 648; EXPECTED_K = 540; EXPECTED_M = 108; EXPECTED_RATE = 5/6
IMAGE_WIDTH = 27; IMAGE_HEIGHT = 20
if IMAGE_WIDTH * IMAGE_HEIGHT != EXPECTED_K: sys.exit(f"FATAL Error: Image dimensions.")
MAX_DECODER_ITER_ALG = 150; MAX_SLIDER_ITER = 50
ALIST_FILENAME = "wifi_648_r083.alist"

# --- Global variable ---
LDPC_PARAMS_GLOBAL = None

# --- ALIST Parser Function ---
def parse_alist_file(filename):
    ldpc_params = {}
    # Construct absolute path relative to this script file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, filename)

        print(f"--- DEBUG: Inside parse_alist_file ---", flush=True) # flush=True helps on some platforms
        print(f"--- DEBUG: Script directory (__file__): {__file__}", flush=True)
        print(f"--- DEBUG: Absolute script directory: {script_dir}", flush=True)
        print(f"--- DEBUG: Trying to open ALIST file at: {abs_file_path}", flush=True)
        print(f"--- DEBUG: Current working directory: {os.getcwd()}", flush=True)
        try:
            print(f"--- DEBUG: Files in script directory ({script_dir}): {os.listdir(script_dir)}", flush=True)
        except Exception as list_err:
            print(f"--- DEBUG: Error listing script directory: {list_err}", flush=True)

    except Exception as path_err:
         print(f"--- DEBUG: CRITICAL ERROR constructing path: {path_err}", flush=True)
         traceback.print_exc()
         return None # Cannot proceed if path construction fails

    try:
        # Ensure the file exists before trying to open
        print(f"--- DEBUG: Checking existence of: {abs_file_path}", flush=True)
        if not os.path.exists(abs_file_path):
            # Explicitly raise FileNotFoundError if os.path.exists fails
            print(f"--- DEBUG: os.path.exists returned False for {abs_file_path}", flush=True)
            raise FileNotFoundError(f"ALIST file not found at {abs_file_path} (checked with os.path.exists)")
        else:
             print(f"--- DEBUG: os.path.exists returned True.", flush=True)


        with open(abs_file_path, 'r') as f: lines = f.readlines()
        print(f"--- DEBUG: Successfully opened and read {abs_file_path}", flush=True) # Add success print

        # --- Rest of your original parsing logic ---
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
        print(f"Parsed ALIST '{filename}': n={n}, m={m}. H shape={H_csc.shape}, nnz={H_csc.nnz}", flush=True) # Keep existing print
        var_neighbors, chk_neighbors = get_neighbors(H_csc) # Pre-calculate neighbors
        ldpc_params = { 'n': n, 'm': m, 'k': n - m, 'H': H_csc, 'var_neighbors': var_neighbors, 'chk_neighbors': chk_neighbors,
                        'max_vnode_deg': max_col_weight_hdr, 'max_cnode_deg': max_row_weight_hdr,
                        'vnode_deg_list': np.array(vnode_deg_list, dtype=int), 'cnode_deg_list': np.array(cnode_deg_list, dtype=int) }
        # --- End of original parsing logic ---
        print(f"--- DEBUG: Successfully parsed. Returning parameters.", flush=True)
        return ldpc_params # Return statement

    except FileNotFoundError as fnf_err:
        print(f"--- DEBUG: Caught FileNotFoundError ---", flush=True) # Explicit log for this exception
        print(f"Error: ALIST file '{filename}' not found. Details: {fnf_err}", flush=True)
        traceback.print_exc() # Print full traceback for file not found
        return None
    except Exception as e:
        print(f"--- DEBUG: Caught other Exception during parsing ---", flush=True) # Explicit log for other exceptions
        print(f"Error parsing ALIST file '{filename}': {e}", flush=True)
        traceback.print_exc()
        return None
    finally:
         # This block might not be reached if return happens earlier, but good practice
         print(f"--- DEBUG: Exiting parse_alist_file ---", flush=True)

# --- Get Neighbors Function ---
def get_neighbors(H_sparse):
    # ... (Unchanged) ...
    if not sp.issparse(H_sparse): H_sparse = sp.csc_matrix(H_sparse)
    m, n = H_sparse.shape; H_coo = H_sparse.tocoo()
    check_indices = H_coo.row; var_indices = H_coo.col
    var_neighbors = [[] for _ in range(n)]; chk_neighbors = [[] for _ in range(m)]
    for r, c in zip(check_indices, var_indices): var_neighbors[c].append(r); chk_neighbors[r].append(c)
    return var_neighbors, chk_neighbors

# --- Image Pattern Generator ---
def create_image_pattern(pattern_name, width, height):
    # ... (Unchanged) ...
    img = np.zeros((height, width), dtype=int)
    if pattern_name == 'square': pad_h, pad_w = height//4, width//4; h_start, h_end = pad_h, height-pad_h; w_start, w_end = pad_w, width-pad_w; img[h_start:h_end, w_start:w_end]=1 if h_start<h_end and w_start<w_end else 0; img[height//2,width//2]=1 if h_start>=h_end or w_start>=w_end else img[height//2,width//2]
    elif pattern_name == 'cross': center_h, center_w = height//2, width//2; thick = max(1, min(height,width)//5); h_s=max(0,center_h-thick//2); h_e=min(height,center_h+(thick+1)//2); w_s=max(0,center_w-thick//2); w_e=min(width,center_w+(thick+1)//2); img[h_s:h_e,:]=1; img[:,w_s:w_e]=1
    elif pattern_name == 'checkerboard': img = np.fromfunction(lambda r, c: (r+c)%2==0, (height, width), dtype=int)
    elif pattern_name == 'random_msg': img = np.random.randint(0, 2, size=(height, width))
    return img

# --- Systematic Encoder (Robust Version) ---
def encode_systematic_robust(message_bits_k, H):
    # ... (Unchanged) ...
    if sp.issparse(H): H_dense = H.toarray()
    else: H_dense = H
    m, n = H_dense.shape; k = n - m
    if len(message_bits_k) != k: return None
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
            if pivot == m: continue
            Aug[[pivot_row, pivot], :] = Aug[[pivot, pivot_row], :]
            for i in range(pivot_row + 1, m):
                if Aug[i, col] == 1: Aug[i, :] = (Aug[i, :] + Aug[pivot_row, :]) % 2
            pivot_row += 1
        rank = pivot_row # Check Rank
        if rank < m: print(f"Encode Error: H_p singular (rank={rank} < m={m})."); return None
        for i in range(m - 1, -1, -1): # Back substitution
            pivot_col = -1;
            for c in range(m):
                 if Aug[i, c] == 1: pivot_col = c; break
            if pivot_col == -1: print(f"Encode Error: No pivot row {i}"); return None
            for row_above in range(i):
                if Aug[row_above, pivot_col] == 1: Aug[row_above, :] = (Aug[row_above, :] + Aug[i, :]) % 2
        parity_bits_p = Aug[:, -1]; codeword_c = np.concatenate((m_vec, parity_bits_p)).astype(int)
        if np.any(calculate_syndrome(H, codeword_c) != 0): print("Encode verify failed!"); return None
        return codeword_c
    except Exception as e: print(f"Gaussian elim error in encoder: {e}"); traceback.print_exc(); return None

# --- Other Helpers ---
def binary_symmetric_channel(codeword, p):
    # ... (Unchanged) ...
    noisy_codeword = codeword.copy(); flips = 0; indices_flipped = []
    for i in range(len(noisy_codeword)):
        if random.random() < p: noisy_codeword[i] = 1 - noisy_codeword[i]; flips += 1; indices_flipped.append(i)
    return noisy_codeword, flips, indices_flipped
def calculate_syndrome(H_input, codeword):
    # ... (Unchanged) ...
    if H_input is None or codeword is None or H_input.shape[1] != len(codeword): return np.array([-1], dtype=int)
    codeword_int = np.array(codeword, dtype=int)
    if sp.issparse(H_input): syndrome = H_input.dot(codeword_int) % 2
    else: syndrome = H_input @ codeword_int % 2
    return syndrome
def calculate_bsc_llrs(noisy_codeword, p):
    # ... (Unchanged) ...
    p_stable = max(1e-9, min(p, 1.0 - 1e-9)); L0 = math.log((1.0 - p_stable) / p_stable)
    llrs = (1 - 2 * noisy_codeword) * L0; MAX_LLR_VALUE = 50; llrs = np.clip(llrs, -MAX_LLR_VALUE, MAX_LLR_VALUE)
    return llrs
def safe_atanh(x):
    # ... (Unchanged) ...
    clipped_x = np.clip(x, -0.9999999, 0.9999999); return np.arctanh(clipped_x)

# --- Custom LLR Belief Propagation Decoder ---
def decode_llrbp_for_dash(noisy_codeword, H, var_neighbors, chk_neighbors, max_iter, channel_prob):
    # ... (Unchanged) ...
    history = { 'iteration': [], 'total_llrs': [], 'decoded_bits': [], 'syndrome': [], 'syndrome_weight': [], 'status': "Starting"}
    n = len(noisy_codeword)
    if H.size == 0 or n == 0: history['status'] = "Error: Invalid H/codeword dims."; return history, noisy_codeword.copy(), 0
    m = H.shape[0]; MAX_LLR_INIT = 20
    stable_p = max(1e-9, min(channel_prob, 0.5 - 1e-9))
    if stable_p < 1e-8: L0 = MAX_LLR_INIT; history['status'] = "Note: p≈0"
    elif abs(stable_p - 0.5) < 1e-9: L0 = 0; history['status'] = "Note: p≈0.5"
    else: L0 = math.log((1.0 - stable_p) / stable_p); L0 = max(-MAX_LLR_INIT, min(L0, MAX_LLR_INIT))
    intrinsic_llrs = np.array([(1 - 2 * bit) * L0 for bit in noisy_codeword])
    msg_c2v = np.zeros((m, n)); current_decoded_codeword = (intrinsic_llrs < 0).astype(int)
    total_llrs = intrinsic_llrs.copy()
    history['iteration'].append(0); history['total_llrs'].append(total_llrs.copy().tolist()); history['decoded_bits'].append(current_decoded_codeword.copy().tolist())
    initial_syndrome = calculate_syndrome(H, current_decoded_codeword); history['syndrome'].append(initial_syndrome.tolist()); history['syndrome_weight'].append(np.sum(initial_syndrome))
    converged = False; stalled = False; iters_done = 0
    noisy_syndrome_weight = np.sum(calculate_syndrome(H, noisy_codeword))
    if noisy_syndrome_weight == 0: history['status'] = "No errors detected (Syndrome 0)."; converged = True; current_decoded_codeword = noisy_codeword.copy(); history['decoded_bits'][-1] = current_decoded_codeword.tolist(); history['syndrome'][-1] = np.zeros(m, dtype=int).tolist(); history['syndrome_weight'][-1] = 0
    for iteration in range(max_iter if not converged else 0):
        iters_done = iteration + 1; history['status'] = f"Running Iter {iters_done}"
        prev_decoded_codeword = current_decoded_codeword.copy()
        msg_v2c = np.zeros((m, n)) # V2C
        for v in range(n): conn_c = var_neighbors[v]; in_llr = sum(msg_c2v[c_p, v] for c_p in conn_c); [msg_v2c.__setitem__((c, v), intrinsic_llrs[v] + in_llr - msg_c2v[c, v]) for c in conn_c]
        new_msg_c2v = np.zeros((m, n)) # C2V
        for c in range(m): conn_v = chk_neighbors[c]; [new_msg_c2v.__setitem__((c, v), 2 * safe_atanh(np.prod([np.tanh(np.clip(msg_v2c[c, v_p], -30, 30)/2.0) for v_p in conn_v if v_p != v]))) for v in conn_v]
        msg_c2v = new_msg_c2v
        total_llrs = intrinsic_llrs.copy(); [total_llrs.__setitem__(v, total_llrs[v] + sum(msg_c2v[c, v] for c in var_neighbors[v])) for v in range(n)] # Update LLRs
        current_decoded_codeword = (total_llrs < 0).astype(int) # Decide
        syndrome = calculate_syndrome(H, current_decoded_codeword); syndrome_weight = np.sum(syndrome) # Check Syndrome
        history['iteration'].append(iters_done); history['total_llrs'].append(total_llrs.copy().tolist()); history['decoded_bits'].append(current_decoded_codeword.copy().tolist()); history['syndrome'].append(syndrome.tolist()); history['syndrome_weight'].append(syndrome_weight) # Record
        if syndrome_weight == 0: history['status'] = f"Converged iter {iters_done}."; converged = True; break # Converged?
        if iteration > 0 and np.array_equal(current_decoded_codeword, prev_decoded_codeword): history['status'] = f"Stalled iter {iters_done}."; stalled = True; break # Stalled?
    if not converged and not stalled and iters_done >= max_iter: # Max iters?
        final_syndrome_weight = history['syndrome_weight'][-1] if history['syndrome_weight'] else -1
        history['status'] = f"Max iters ({max_iter}) reached. Final synd weight: {final_syndrome_weight}."
    return history, current_decoded_codeword, iters_done

# --- Image Figure Helper ---
def create_image_fig(image_array, title="Image"):
    # ... (Unchanged) ...
    if image_array is None or image_array.size == 0: fig = go.Figure().update_layout(title=title + " (No Data)", xaxis_visible=False, yaxis_visible=False, template='plotly_white'); return fig
    img_numeric = image_array.astype(float); fig = px.imshow(img_numeric, binary_string=False, color_continuous_scale='gray_r', aspect='equal')
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), coloraxis_showscale=False, margin=dict(l=5, r=5, t=5, b=5));
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>color: %{z}<extra></extra>")
    return fig

# --- H Matrix Plot Helper ---
def create_h_matrix_fig(H_sparse):
    # ... (Unchanged) ...
    if H_sparse is None or not sp.issparse(H_sparse) or H_sparse.nnz == 0: fig = go.Figure().update_layout(title="H Matrix (No Data)", xaxis_visible=False, yaxis_visible=False); return fig
    try: H_dense = H_sparse.toarray()
    except MemoryError: fig = go.Figure().update_layout(title="H Matrix (Too Large to Plot)"); return fig
    except Exception as e: fig = go.Figure().update_layout(title=f"H Matrix Plot Error: {e}"); return fig
    fig = px.imshow(H_dense, color_continuous_scale='gray_r', aspect='auto')
    fig.update_layout(title=f"H Matrix ({H_sparse.shape[0]}x{H_sparse.shape[1]})", xaxis_title="Variable Nodes", yaxis_title="Check Nodes", coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(side="top", tickvals=[], ticktext=[]); fig.update_yaxes(tickvals=[], ticktext=[]);
    fig.update_traces(hovertemplate="Var Node (x): %{x}<br>Check Node (y): %{y}<br>Value: %{z}<extra></extra>")
    return fig

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# --- App Layout (Tabs added) ---
app.layout = dbc.Container([
    dcc.Store(id='history-store'),
    dcc.Store(id='orig-image-store'),
    dbc.Row(dbc.Col(html.H1("LDPC Demo"), width=12)),
     dbc.Row([ dbc.Col([ # Controls Column
             dbc.Card([ dbc.CardHeader("Controls"), dbc.CardBody([
                    html.Label("Image Pattern:"),
                    dcc.Dropdown( id='pattern-dropdown', options=[ {'label': 'Square', 'value': 'square'}, {'label': 'Cross', 'value': 'cross'}, {'label': 'Checkerboard', 'value': 'checkerboard'}, {'label': 'Random Message', 'value': 'random_msg'}, ], value='checkerboard', clearable=False ), html.Br(),
                    html.Label("Channel Noise Probability (p):"),
                    dcc.Slider(id='noise-slider', min=0.0, max=0.05, step=0.001, value=0.01, marks={i/100: f'{i/100:.2f}' for i in range(0, 6, 1)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label("Max Decoder Iterations:"),
                    dcc.Slider(id='max-iter-slider', min=10, max=MAX_DECODER_ITER_ALG + 50, step=10, value=MAX_DECODER_ITER_ALG, marks={i: str(i) for i in range(0, MAX_DECODER_ITER_ALG + 51, 25)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label("Random Seed (optional):"),
                    dbc.Input(id='seed-input', type='number', placeholder="Leave blank for random", step=1), html.Br(),
                    dbc.Button("Run Simulation", id="run-button", n_clicks=0, color="primary", className="mt-3"),
                ])]), dbc.Card([dbc.CardHeader("Run Info"), dbc.CardBody(dbc.Spinner(html.Pre(id="run-summary", style={'maxHeight': '250px', 'overflowY': 'scroll'})))], className="mt-3"),
        ], md=4), dbc.Col([ # Displays Column
             # *** Use Tabs for H Matrix and Diagnostic Plots ***
             dbc.Tabs([
                 dbc.Tab(label="H Matrix", children=[
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='h-matrix-plot'))), className="mt-2") # Added margin top
                 ]),
                 dbc.Tab(label="Syndrome Weight", children=[
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='syndrome-plot'))), className="mt-2")
                 ]),
                 dbc.Tab(label="LLR Evolution", children=[
                     dbc.Card(dbc.CardBody(dbc.Spinner(dcc.Graph(id='llr-plot'))), className="mt-2")
                 ]),
             ]),
             html.Hr(),
             # (Rest of display layout unchanged)
            dbc.Row([
                 dbc.Col(dbc.Card([dbc.CardHeader(f"Original Image ({IMAGE_WIDTH}x{IMAGE_HEIGHT})"), dbc.CardBody(dbc.Spinner(dcc.Graph(id='img-original')))]), md=4),
                 dbc.Col(dbc.Card([dbc.CardHeader("Received Noisy"),
                                   dbc.CardBody([dbc.Spinner(dcc.Graph(id='img-noisy')),
                                                 html.Pre(id='noisy-info-text', style={'fontSize':'small','textAlign':'center', 'marginTop':'5px'})
                                                ])]), md=4),
                 dbc.Col(dbc.Card([dbc.CardHeader("LDPC Decoded"),
                                   dbc.CardBody([dbc.Spinner(dcc.Graph(id='img-decoded')),
                                                 html.Pre(id='decoded-info-text', style={'fontSize':'small','textAlign':'center', 'marginTop':'5px'})
                                                ])]), md=4),
            ]),
            html.Hr(),
            dbc.Row([
                 dbc.Col([
                     html.Label("View Decoder Iteration:", style={'fontWeight':'bold'}),
                     dcc.Slider( id='iteration-slider', min=0, max=MAX_SLIDER_ITER, step=1, value=0, marks={i: str(i) for i in range(0, MAX_SLIDER_ITER + 1, 5)}, disabled=True, tooltip={"placement": "bottom", "always_visible": True} ),
                     html.P(id='iteration-display-label', style={'textAlign':'center', 'marginTop':'5px'})
                 ])
            ]),
        ], md=8)
    ]),
], fluid=True)


# --- Main Simulation Callback (Outputs new diagnostic plots) ---
@app.callback(
    [Output('img-original', 'figure'), Output('img-noisy', 'figure'),
     Output('h-matrix-plot', 'figure'), # Keep H plot
     Output('syndrome-plot', 'figure'), # Add Syndrome plot output
     Output('llr-plot', 'figure'),      # Add LLR plot output
     Output('run-summary', 'children'), Output('history-store', 'data'),
     Output('iteration-slider', 'value'), Output('iteration-slider', 'marks'),
     Output('iteration-slider', 'disabled'), Output('orig-image-store', 'data'),
     Output('noisy-info-text', 'children') ],
    [Input('run-button', 'n_clicks')],
    [State('pattern-dropdown', 'value'), State('noise-slider', 'value'),
     State('max-iter-slider', 'value'), State('seed-input', 'value')]
)
def update_simulation(n_clicks, pattern, noise_prob, max_iter, seed_val):
    global LDPC_PARAMS_GLOBAL

    # --- Initial State Handling ---
    initial_marks = {i: str(i) for i in range(0, MAX_SLIDER_ITER + 1, 5)}
    initial_noisy_info = "Syndrome W: N/A | Img Errors: N/A"
    no_data_fig=create_image_fig(None, "No Data");
    no_plot_fig = go.Figure().update_layout(title="(No data)", xaxis_visible=False, yaxis_visible=False)
    h_fig = create_h_matrix_fig(LDPC_PARAMS_GLOBAL['H'] if LDPC_PARAMS_GLOBAL else None)

    if n_clicks == 0:
        err_msg = "Click 'Run Simulation'."
        return (no_data_fig, no_data_fig, h_fig, no_plot_fig, no_plot_fig,
                err_msg, None, 0, initial_marks, True, None, initial_noisy_info)

    if LDPC_PARAMS_GLOBAL is None:
         error_msg = f"FATAL ERROR: LDPC parameters not loaded on startup."
         summary_lines = [error_msg]; err_fig = create_image_fig(None, "Error");
         return (err_fig, err_fig, h_fig, no_plot_fig, no_plot_fig,
                 "\n".join(summary_lines), None, 0, initial_marks, True, None, initial_noisy_info)

    # --- Simulation Setup ---
    start_time = time.time();
    ldpc_params = LDPC_PARAMS_GLOBAL
    n_val = ldpc_params['n']; k_val = ldpc_params['k']; m_val = ldpc_params['m']
    H_csc = ldpc_params['H']
    var_neighbors = ldpc_params['var_neighbors']
    chk_neighbors = ldpc_params['chk_neighbors']

    # Seed & Summary Init
    if seed_val is not None:
        try: seed=int(seed_val); random.seed(seed); np.random.seed(seed)
        except ValueError: seed=None
    else: seed=None
    seed_msg = f"Seed: {seed}" if seed is not None else "Seed: Random"
    summary_lines = [f"--- Run {n_clicks} ---", seed_msg]
    actual_rate = k_val / n_val if n_val > 0 else 0
    summary_lines.append(f"Using WiFi N={n_val}, K={k_val}, R={actual_rate:.3f} from '{ALIST_FILENAME}'")
    summary_lines.append(f"Image Pattern: '{pattern}' ({IMAGE_WIDTH}x{IMAGE_HEIGHT})")

    # Create H matrix plot
    fig_h_matrix = create_h_matrix_fig(H_csc)

    # --- Generate Message & Encode ---
    img_orig_array = create_image_pattern(pattern, IMAGE_WIDTH, IMAGE_HEIGHT)
    message_bits_k = img_orig_array.flatten()
    fig_orig = create_image_fig(img_orig_array, "Original")
    orig_image_list = img_orig_array.tolist() # Store original image

    summary_lines.append(f"Encoding message...")
    codeword_c = encode_systematic_robust(message_bits_k, H_csc)

    if codeword_c is None:
        summary_lines.append(f"\nFATAL ERROR: Systematic encoding failed (H_p singular?).")
        err_fig=create_image_fig(None, "Encoding Error")
        return (fig_orig, err_fig, fig_h_matrix, no_plot_fig, no_plot_fig,
                "\n".join(summary_lines), None, 0, initial_marks, True, orig_image_list, initial_noisy_info)

    summary_lines.append(f"Encoding successful.")

    # --- Noise & Decoding ---
    summary_lines.append(f"\n--- Noise & Decoding ---"); summary_lines.append(f"Noise p={noise_prob:.4f}, Max Alg Iter={max_iter}")
    noisy_cw, flips, flipped_idx = binary_symmetric_channel(codeword_c, noise_prob)
    summary_lines.append(f"   Total flips: {flips}")
    if flips <= 15 and flips > 0: summary_lines.append(f"   Flip indices: {flipped_idx}")

    noisy_msg_bits = noisy_cw[:k_val]; img_noisy_arr = noisy_msg_bits.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)); fig_noisy = create_image_fig(img_noisy_arr, "Received Noisy")
    err_noisy_img_before_decode = np.sum(img_orig_array != img_noisy_arr);
    noisy_syndrome = calculate_syndrome(H_csc, noisy_cw)
    noisy_syndrome_weight = np.sum(noisy_syndrome)
    noisy_info_str = f"Syndrome W: {noisy_syndrome_weight} | Img Errors: {err_noisy_img_before_decode}"
    summary_lines.append(f"   Errors in image part before decoding: {err_noisy_img_before_decode}")
    summary_lines.append(f"   Initial syndrome weight: {noisy_syndrome_weight}")

    # Call OUR decoder
    history, final_decoded_cw, iters_done = decode_llrbp_for_dash(noisy_cw, H_csc, var_neighbors, chk_neighbors, max_iter, noise_prob)
    summary_lines.append(f"Decoder iters performed: {iters_done}")
    summary_lines.append(f"Decoder status: {history['status']}")
    final_syndrome_weight = history['syndrome_weight'][-1] if history['syndrome_weight'] else -1
    summary_lines.append(f"Final syndrome weight: {final_syndrome_weight}")

    # Prepare history for storage
    history_for_store = {'iteration': history.get('iteration', [0]),
                         'decoded_bits': history.get('decoded_bits', [codeword_c.tolist()]),
                         'syndrome_weight': history.get('syndrome_weight', [0]),
                         'total_llrs': history.get('total_llrs', [[]])} # Store LLRs too

    # --- Final Stats ---
    if final_decoded_cw is None: final_decoded_cw = noisy_cw.copy(); summary_lines.append("Warning: Using noisy CW.")
    decoded_msg_bits_final = final_decoded_cw[:k_val]
    if len(decoded_msg_bits_final) != k_val:
        summary_lines.append(f"Error: Final decoded message length != k"); err_fig = create_image_fig(None, "Decode Error")
        return (fig_orig, fig_noisy, err_fig, fig_h_matrix, no_plot_fig, no_plot_fig,
                "\n".join(summary_lines), None, 0, initial_marks, True, orig_image_list, noisy_info_str)

    img_decoded_arr_final = decoded_msg_bits_final.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    err_remain = np.sum(img_orig_array != img_decoded_arr_final); summary_lines.append(f"\nErrors remaining in image part after FINAL decoding: {err_remain}")
    cw_err_remain = np.sum(codeword_c != final_decoded_cw); summary_lines.append(f"Errors remaining in full codeword after FINAL decoding: {cw_err_remain}")
    # Result message
    if final_syndrome_weight==0 and cw_err_remain==0: result_msg="Result: SUCCESS!"
    elif final_syndrome_weight==0 and cw_err_remain>0: result_msg="Result: Partial Success (Converged to wrong codeword)."
    elif final_syndrome_weight!=0 and err_remain < err_noisy_img_before_decode: result_msg="Result: Partial Correction (Failed convergence)."
    elif final_syndrome_weight!=0: result_msg="Result: FAILURE (Failed convergence)."
    else: result_msg="Result: Unknown."
    summary_lines.append(result_msg); elapsed_time = time.time() - start_time; summary_lines.append(f"Running time: {elapsed_time:.3f} seconds")

    # --- Generate Syndrome and LLR Plots ---
    fig_syndrome = go.Figure()
    iterations_synd = history.get('iteration', [])
    syndrome_weights = history.get('syndrome_weight', [])
    if iterations_synd and len(iterations_synd) == len(syndrome_weights):
        fig_syndrome.add_trace(go.Scattergl(x=iterations_synd, y=syndrome_weights, mode='lines+markers', name='Syndrome Weight'))
        fig_syndrome.update_layout(title="Syndrome Weight vs Iteration", xaxis_title="Decoder Iteration", yaxis_title="Syndrome Weight", yaxis=dict(range=[-0.5, m_val + 0.5]), uirevision=n_clicks)
    else: fig_syndrome.update_layout(title="Syndrome Weight Plot (No data)")

    # --- Generate Focused LLR Plot ---
    fig_llr = go.Figure()
    iterations_llr = history.get('iteration', [])
    llrs_history_list = history.get('total_llrs', [])
    if iterations_llr and llrs_history_list and len(iterations_llr) == len(llrs_history_list):
         try:
             llrs_history_array = np.array(llrs_history_list) # Shape (iters+1, n)
             if llrs_history_array.ndim == 2 and llrs_history_array.shape[1] == n_val:
                 # Identify bit indices
                 signs = np.sign(llrs_history_array)
                 # Oscillating: Sign changes at any point (compare sign array columns)
                 oscillating_indices = np.where(np.any(signs != signs[0,:], axis=0))[0]
                 # Always Positive: All signs are >= 0 (or zero)
                 always_pos_indices = np.where(np.all(signs >= 0, axis=0))[0]
                 # Always Negative: All signs are <= 0 (or zero)
                 always_neg_indices = np.where(np.all(signs <= 0, axis=0))[0]

                 # Calculate averages (handle division by zero if no bits in category)
                 avg_pos_llrs = np.mean(llrs_history_array[:, always_pos_indices], axis=1) if len(always_pos_indices) > 0 else np.full(len(iterations_llr), np.nan)
                 avg_neg_llrs = np.mean(llrs_history_array[:, always_neg_indices], axis=1) if len(always_neg_indices) > 0 else np.full(len(iterations_llr), np.nan)

                 # Add traces
                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_pos_llrs, mode='lines', name='Avg Always Pos LLR', line=dict(color='blue', width=3)))
                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_neg_llrs, mode='lines', name='Avg Always Neg LLR', line=dict(color='red', width=3)))

                 # Plot individual oscillating LLRs (limit if too many)
                 max_oscillating_to_plot = 30
                 num_oscillating = len(oscillating_indices)
                 indices_to_plot = oscillating_indices
                 if num_oscillating > max_oscillating_to_plot:
                     indices_to_plot = np.random.choice(oscillating_indices, size=max_oscillating_to_plot, replace=False)
                     plot_title = f"LLR Evolution"
                 else:
                      plot_title = f"LLR Evolution"

                 for bit_index in indices_to_plot:
                     fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=llrs_history_array[:, bit_index], mode='lines', name=f'Bit {bit_index}', line=dict(width=1, dash='dot'), opacity=0.7))

                 fig_llr.update_layout(title=plot_title, xaxis_title="Decoder Iteration", yaxis_title="Total LLR", uirevision=n_clicks)
             else:
                 raise ValueError("LLR history array shape mismatch")
         except Exception as e:
             print(f"Error creating LLR plot: {e}")
             traceback.print_exc()
             fig_llr.update_layout(title="LLR Plot Error")
    else:
        fig_llr.update_layout(title="LLR Plot (No data)")
    # --- End Focused LLR Plot ---

    # --- Prepare Slider Outputs ---
    actual_iters_run = iters_done
    slider_value = min(actual_iters_run, MAX_SLIDER_ITER)
    max_slider_index = min(actual_iters_run, MAX_SLIDER_ITER); mark_step = max(1, MAX_SLIDER_ITER // 10)
    if mark_step == 0: mark_step = 1
    elif mark_step < 5 and MAX_SLIDER_ITER > 10: mark_step = 5
    slider_marks = {i: str(i) for i in range(0, MAX_SLIDER_ITER + 1, mark_step)}
    if max_slider_index not in slider_marks and max_slider_index <= MAX_SLIDER_ITER: slider_marks[max_slider_index] = str(max_slider_index)
    slider_marks = dict(sorted(slider_marks.items())); slider_disabled = False if actual_iters_run > 0 else True

    return (fig_orig, fig_noisy, fig_h_matrix, fig_syndrome, fig_llr, # Return all plots
            "\n".join(summary_lines), history_for_store,
            slider_value, slider_marks, slider_disabled,
            orig_image_list, noisy_info_str)


# --- Callback for Iteration Slider ---
@app.callback(
    [Output('img-decoded', 'figure'), Output('iteration-display-label', 'children'),
     Output('decoded-info-text', 'children')],
    [Input('iteration-slider', 'drag_value'), Input('iteration-slider', 'value')],
    [State('history-store', 'data'), State('orig-image-store', 'data')]
)
def update_displayed_iteration(drag_value, value, history_data, orig_image_list_data):
    # ... (Unchanged) ...
    selected_iter = drag_value if drag_value is not None else value
    default_info_text = "Syndrome W: N/A | Img Errors: N/A"
    if (selected_iter is None or history_data is None or orig_image_list_data is None
            or not history_data.get('decoded_bits') or not history_data.get('syndrome_weight')):
        return create_image_fig(None, "Decoded (No Data)"), "Iteration: N/A", default_info_text
    iterations_list = history_data.get('iteration', []); decoded_bits_history = history_data.get('decoded_bits', []); syndrome_weight_history = history_data.get('syndrome_weight', [])
    max_hist_index = len(decoded_bits_history) - 1; iter_index = min(selected_iter, max_hist_index)
    if iter_index < 0: return create_image_fig(None, "Decoded (Error)"), f"Iteration: Error", default_info_text
    selected_iter_actual = iterations_list[iter_index] if iter_index < len(iterations_list) else '?'
    label_text = f"Showing Iteration: {selected_iter_actual} (Slider: {selected_iter})"
    decoded_bits_list = decoded_bits_history[iter_index]; syndrome_weight_iter = syndrome_weight_history[iter_index] if iter_index < len(syndrome_weight_history) else '?'
    decoded_cw_iter = np.array(decoded_bits_list, dtype=int); k_val = EXPECTED_K; decoded_message_bits_iter = decoded_cw_iter[:k_val]
    if len(decoded_message_bits_iter) != IMAGE_WIDTH * IMAGE_HEIGHT: return create_image_fig(None, "Decoded (Dim Error)"), label_text, default_info_text
    img_decoded_array_iter = decoded_message_bits_iter.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)); fig_decoded_iter = create_image_fig(img_decoded_array_iter, f"Decoded (Iter {selected_iter_actual})")
    img_orig_array = np.array(orig_image_list_data, dtype=int); errors_iter = np.sum(img_orig_array != img_decoded_array_iter)
    info_text = f"Syndrome W: {syndrome_weight_iter} | Img Errors: {errors_iter}"
    return fig_decoded_iter, label_text, info_text

# --- Run the App ---
if __name__ == '__main__':
    # Parse ALIST on startup
    print(f"Attempting to parse ALIST file '{ALIST_FILENAME}' on startup...")
    LDPC_PARAMS_GLOBAL = parse_alist_file(ALIST_FILENAME)

    if LDPC_PARAMS_GLOBAL is None:
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"ERROR: Failed to load or parse '{ALIST_FILENAME}' on startup.")
         print("Exiting.")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         sys.exit(1)

    print(f"Successfully loaded LDPC parameters from '{ALIST_FILENAME}'.")
    print(f"Starting Dash server on http://127.0.0.1:8050/ ...")
    # app.run(debug=True, port=8050) # Use standard port 8050