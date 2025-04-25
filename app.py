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
if IMAGE_WIDTH * IMAGE_HEIGHT != EXPECTED_K: sys.exit(f"FATAL Error: Image dimensions mismatch.")
MAX_DECODER_ITER_ALG = 150; MAX_SLIDER_ITER = 50
# ALIST_FILENAME = "wifi_648_r083.alist" # Removed - ALIST data is now embedded

# --- Embedded ALIST Data ---
# Instructions: Replace "PASTE HERE THE ALIST" (including the quotes)
# with the full content of your wifi_648_r083.alist file,
# enclosed in triple quotes like this: """<content>"""
wifi_648_r083_alist_content = """
648 108
4 22
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 
11 52 60 102 
12 53 61 103 
13 54 62 104 
14 28 63 105 
15 29 64 106 
16 30 65 107 
17 31 66 108 
18 32 67 82 
19 33 68 83 
20 34 69 84 
21 35 70 85 
22 36 71 86 
23 37 72 87 
24 38 73 88 
25 39 74 89 
26 40 75 90 
27 41 76 91 
1 42 77 92 
2 43 78 93 
3 44 79 94 
4 45 80 95 
5 46 81 96 
6 47 55 97 
7 48 56 98 
8 49 57 99 
9 50 58 100 
10 51 59 101 
15 43 66 102 
16 44 67 103 
17 45 68 104 
18 46 69 105 
19 47 70 106 
20 48 71 107 
21 49 72 108 
22 50 73 82 
23 51 74 83 
24 52 75 84 
25 53 76 85 
26 54 77 86 
27 28 78 87 
1 29 79 88 
2 30 80 89 
3 31 81 90 
4 32 55 91 
5 33 56 92 
6 34 57 93 
7 35 58 94 
8 36 59 95 
9 37 60 96 
10 38 61 97 
11 39 62 98 
12 40 63 99 
13 41 64 100 
14 42 65 101 
20 44 78 95 
21 45 79 96 
22 46 80 97 
23 47 81 98 
24 48 55 99 
25 49 56 100 
26 50 57 101 
27 51 58 102 
1 52 59 103 
2 53 60 104 
3 54 61 105 
4 28 62 106 
5 29 63 107 
6 30 64 108 
7 31 65 82 
8 32 66 83 
9 33 67 84 
10 34 68 85 
11 35 69 86 
12 36 70 87 
13 37 71 88 
14 38 72 89 
15 39 73 90 
16 40 74 91 
17 41 75 92 
18 42 76 93 
19 43 77 94 
7 41 79 95 
8 42 80 96 
9 43 81 97 
10 44 55 98 
11 45 56 99 
12 46 57 100 
13 47 58 101 
14 48 59 102 
15 49 60 103 
16 50 61 104 
17 51 62 105 
18 52 63 106 
19 53 64 107 
20 54 65 108 
21 28 66 82 
22 29 67 83 
23 30 68 84 
24 31 69 85 
25 32 70 86 
26 33 71 87 
27 34 72 88 
1 35 73 89 
2 36 74 90 
3 37 75 91 
4 38 76 92 
5 39 77 93 
6 40 78 94 
19 44 72 105 
20 45 73 106 
21 46 74 107 
22 47 75 108 
23 48 76 82 
24 49 77 83 
25 50 78 84 
26 51 79 85 
27 52 80 86 
1 53 81 87 
2 54 55 88 
3 28 56 89 
4 29 57 90 
5 30 58 91 
6 31 59 92 
7 32 60 93 
8 33 61 94 
9 34 62 95 
10 35 63 96 
11 36 64 97 
12 37 65 98 
13 38 66 99 
14 39 67 100 
15 40 68 101 
16 41 69 102 
17 42 70 103 
18 43 71 104 
25 30 61 93 
26 31 62 94 
27 32 63 95 
1 33 64 96 
2 34 65 97 
3 35 66 98 
4 36 67 99 
5 37 68 100 
6 38 69 101 
7 39 70 102 
8 40 71 103 
9 41 72 104 
10 42 73 105 
11 43 74 106 
12 44 75 107 
13 45 76 108 
14 46 77 82 
15 47 78 83 
16 48 79 84 
17 49 80 85 
18 50 81 86 
19 51 55 87 
20 52 56 88 
21 53 57 89 
22 54 58 90 
23 28 59 91 
24 29 60 92 
10 50 70 93 
11 51 71 94 
12 52 72 95 
13 53 73 96 
14 54 74 97 
15 28 75 98 
16 29 76 99 
17 30 77 100 
18 31 78 101 
19 32 79 102 
20 33 80 103 
21 34 81 104 
22 35 55 105 
23 36 56 106 
24 37 57 107 
25 38 58 108 
26 39 59 82 
27 40 60 83 
1 41 61 84 
2 42 62 85 
3 43 63 86 
4 44 64 87 
5 45 65 88 
6 46 66 89 
7 47 67 90 
8 48 68 91 
9 49 69 92 
16 37 77 85 
17 38 78 86 
18 39 79 87 
19 40 80 88 
20 41 81 89 
21 42 55 90 
22 43 56 91 
23 44 57 92 
24 45 58 93 
25 46 59 94 
26 47 60 95 
27 48 61 96 
1 49 62 97 
2 50 63 98 
3 51 64 99 
4 52 65 100 
5 53 66 101 
6 54 67 102 
7 28 68 103 
8 29 69 104 
9 30 70 105 
10 31 71 106 
11 32 72 107 
12 33 73 108 
13 34 74 82 
14 35 75 83 
15 36 76 84 
18 28 61 85 
19 29 62 86 
20 30 63 87 
21 31 64 88 
22 32 65 89 
23 33 66 90 
24 34 67 91 
25 35 68 92 
26 36 69 93 
27 37 70 94 
1 38 71 95 
2 39 72 96 
3 40 73 97 
4 41 74 98 
5 42 75 99 
6 43 76 100 
7 44 77 101 
8 45 78 102 
9 46 79 103 
10 47 80 104 
11 48 81 105 
12 49 55 106 
13 50 56 107 
14 51 57 108 
15 52 58 82 
16 53 59 83 
17 54 60 84 
1 46 68 99 
2 47 69 100 
3 48 70 101 
4 49 71 102 
5 50 72 103 
6 51 73 104 
7 52 74 105 
8 53 75 106 
9 54 76 107 
10 28 77 108 
11 29 78 82 
12 30 79 83 
13 31 80 84 
14 32 81 85 
15 33 55 86 
16 34 56 87 
17 35 57 88 
18 36 58 89 
19 37 59 90 
20 38 60 91 
21 39 61 92 
22 40 62 93 
23 41 63 94 
24 42 64 95 
25 43 65 96 
26 44 66 97 
27 45 67 98 
24 53 63 108 
25 54 64 82 
26 28 65 83 
27 29 66 84 
1 30 67 85 
2 31 68 86 
3 32 69 87 
4 33 70 88 
5 34 71 89 
6 35 72 90 
7 36 73 91 
8 37 74 92 
9 38 75 93 
10 39 76 94 
11 40 77 95 
12 41 78 96 
13 42 79 97 
14 43 80 98 
15 44 81 99 
16 45 55 100 
17 46 56 101 
18 47 57 102 
19 48 58 103 
20 49 59 104 
21 50 60 105 
22 51 61 106 
23 52 62 107 
13 29 77 102 
14 30 78 103 
15 31 79 104 
16 32 80 105 
17 33 81 106 
18 34 55 107 
19 35 56 108 
20 36 57 82 
21 37 58 83 
22 38 59 84 
23 39 60 85 
24 40 61 86 
25 41 62 87 
26 42 63 88 
27 43 64 89 
1 44 65 90 
2 45 66 91 
3 46 67 92 
4 47 68 93 
5 48 69 94 
6 49 70 95 
7 50 71 96 
8 51 72 97 
9 52 73 98 
10 53 74 99 
11 54 75 100 
12 28 76 101 
9 29 94 0 
10 30 95 0 
11 31 96 0 
12 32 97 0 
13 33 98 0 
14 34 99 0 
15 35 100 0 
16 36 101 0 
17 37 102 0 
18 38 103 0 
19 39 104 0 
20 40 105 0 
21 41 106 0 
22 42 107 0 
23 43 108 0 
24 44 82 0 
25 45 83 0 
26 46 84 0 
27 47 85 0 
1 48 86 0 
2 49 87 0 
3 50 88 0 
4 51 89 0 
5 52 90 0 
6 53 91 0 
7 54 92 0 
8 28 93 0 
26 45 74 103 
27 46 75 104 
1 47 76 105 
2 48 77 106 
3 49 78 107 
4 50 79 108 
5 51 80 82 
6 52 81 83 
7 53 55 84 
8 54 56 85 
9 28 57 86 
10 29 58 87 
11 30 59 88 
12 31 60 89 
13 32 61 90 
14 33 62 91 
15 34 63 92 
16 35 64 93 
17 36 65 94 
18 37 66 95 
19 38 67 96 
20 39 68 97 
21 40 69 98 
22 41 70 99 
23 42 71 100 
24 43 72 101 
25 44 73 102 
23 31 77 99 
24 32 78 100 
25 33 79 101 
26 34 80 102 
27 35 81 103 
1 36 55 104 
2 37 56 105 
3 38 57 106 
4 39 58 107 
5 40 59 108 
6 41 60 82 
7 42 61 83 
8 43 62 84 
9 44 63 85 
10 45 64 86 
11 46 65 87 
12 47 66 88 
13 48 67 89 
14 49 68 90 
15 50 69 91 
16 51 70 92 
17 52 71 93 
18 53 72 94 
19 54 73 95 
20 28 74 96 
21 29 75 97 
22 30 76 98 
18 48 64 83 
19 49 65 84 
20 50 66 85 
21 51 67 86 
22 52 68 87 
23 53 69 88 
24 54 70 89 
25 28 71 90 
26 29 72 91 
27 30 73 92 
1 31 74 93 
2 32 75 94 
3 33 76 95 
4 34 77 96 
5 35 78 97 
6 36 79 98 
7 37 80 99 
8 38 81 100 
9 39 55 101 
10 40 56 102 
11 41 57 103 
12 42 58 104 
13 43 59 105 
14 44 60 106 
15 45 61 107 
16 46 62 108 
17 47 63 82 
2 41 71 101 
3 42 72 102 
4 43 73 103 
5 44 74 104 
6 45 75 105 
7 46 76 106 
8 47 77 107 
9 48 78 108 
10 49 79 82 
11 50 80 83 
12 51 81 84 
13 52 55 85 
14 53 56 86 
15 54 57 87 
16 28 58 88 
17 29 59 89 
18 30 60 90 
19 31 61 91 
20 32 62 92 
21 33 63 93 
22 34 64 94 
23 35 65 95 
24 36 66 96 
25 37 67 97 
26 38 68 98 
27 39 69 99 
1 40 70 100 
9 35 77 91 
10 36 78 92 
11 37 79 93 
12 38 80 94 
13 39 81 95 
14 40 55 96 
15 41 56 97 
16 42 57 98 
17 43 58 99 
18 44 59 100 
19 45 60 101 
20 46 61 102 
21 47 62 103 
22 48 63 104 
23 49 64 105 
24 50 65 106 
25 51 66 107 
26 52 67 108 
27 53 68 82 
1 54 69 83 
2 28 70 84 
3 29 71 85 
4 30 72 86 
5 31 73 87 
6 32 74 88 
7 33 75 89 
8 34 76 90 
15 51 77 88 
16 52 78 89 
17 53 79 90 
18 54 80 91 
19 28 81 92 
20 29 55 93 
21 30 56 94 
22 31 57 95 
23 32 58 96 
24 33 59 97 
25 34 60 98 
26 35 61 99 
27 36 62 100 
1 37 63 101 
2 38 64 102 
3 39 65 103 
4 40 66 104 
5 41 67 105 
6 42 68 106 
7 43 69 107 
8 44 70 108 
9 45 71 82 
10 46 72 83 
11 47 73 84 
12 48 74 85 
13 49 75 86 
14 50 76 87 
15 53 67 95 
16 54 68 96 
17 28 69 97 
18 29 70 98 
19 30 71 99 
20 31 72 100 
21 32 73 101 
22 33 74 102 
23 34 75 103 
24 35 76 104 
25 36 77 105 
26 37 78 106 
27 38 79 107 
1 39 80 108 
2 40 81 82 
3 41 55 83 
4 42 56 84 
5 43 57 85 
6 44 58 86 
7 45 59 87 
8 46 60 88 
9 47 61 89 
10 48 62 90 
11 49 63 91 
12 50 64 92 
13 51 65 93 
14 52 66 94 
27 55 108 0 
1 56 82 0 
2 57 83 0 
3 58 84 0 
4 59 85 0 
5 60 86 0 
6 61 87 0 
7 62 88 0 
8 63 89 0 
9 64 90 0 
10 65 91 0 
11 66 92 0 
12 67 93 0 
13 68 94 0 
14 69 95 0 
15 70 96 0 
16 71 97 0 
17 72 98 0 
18 73 99 0 
19 74 100 0 
20 75 101 0 
21 76 102 0 
22 77 103 0 
23 78 104 0 
24 79 105 0 
25 80 106 0 
26 81 107 0 
1 28 0 0 
2 29 0 0 
3 30 0 0 
4 31 0 0 
5 32 0 0 
6 33 0 0 
7 34 0 0 
8 35 0 0 
9 36 0 0 
10 37 0 0 
11 38 0 0 
12 39 0 0 
13 40 0 0 
14 41 0 0 
15 42 0 0 
16 43 0 0 
17 44 0 0 
18 45 0 0 
19 46 0 0 
20 47 0 0 
21 48 0 0 
22 49 0 0 
23 50 0 0 
24 51 0 0 
25 52 0 0 
26 53 0 0 
27 54 0 0 
28 55 0 0 
29 56 0 0 
30 57 0 0 
31 58 0 0 
32 59 0 0 
33 60 0 0 
34 61 0 0 
35 62 0 0 
36 63 0 0 
37 64 0 0 
38 65 0 0 
39 66 0 0 
40 67 0 0 
41 68 0 0 
42 69 0 0 
43 70 0 0 
44 71 0 0 
45 72 0 0 
46 73 0 0 
47 74 0 0 
48 75 0 0 
49 76 0 0 
50 77 0 0 
51 78 0 0 
52 79 0 0 
53 80 0 0 
54 81 0 0 
55 82 0 0 
56 83 0 0 
57 84 0 0 
58 85 0 0 
59 86 0 0 
60 87 0 0 
61 88 0 0 
62 89 0 0 
63 90 0 0 
64 91 0 0 
65 92 0 0 
66 93 0 0 
67 94 0 0 
68 95 0 0 
69 96 0 0 
70 97 0 0 
71 98 0 0 
72 99 0 0 
73 100 0 0 
74 101 0 0 
75 102 0 0 
76 103 0 0 
77 104 0 0 
78 105 0 0 
79 106 0 0 
80 107 0 0 
81 108 0 0 
18 41 63 103 118 139 181 202 227 244 275 313 344 354 384 416 459 479 500 527 542 568 
19 42 64 104 119 140 182 203 228 245 276 314 345 355 385 417 433 480 501 528 543 569 
20 43 65 105 120 141 183 204 229 246 277 315 346 356 386 418 434 481 502 529 544 570 
21 44 66 106 121 142 184 205 230 247 278 316 347 357 387 419 435 482 503 530 545 571 
22 45 67 107 122 143 185 206 231 248 279 317 348 358 388 420 436 483 504 531 546 572 
23 46 68 108 123 144 186 207 232 249 280 318 349 359 389 421 437 484 505 532 547 573 
24 47 69 82 124 145 187 208 233 250 281 319 350 360 390 422 438 485 506 533 548 574 
25 48 70 83 125 146 188 209 234 251 282 320 351 361 391 423 439 486 507 534 549 575 
26 49 71 84 126 147 189 210 235 252 283 321 325 362 392 424 440 460 508 535 550 576 
27 50 72 85 127 148 163 211 236 253 284 322 326 363 393 425 441 461 509 536 551 577 
1 51 73 86 128 149 164 212 237 254 285 323 327 364 394 426 442 462 510 537 552 578 
2 52 74 87 129 150 165 213 238 255 286 324 328 365 395 427 443 463 511 538 553 579 
3 53 75 88 130 151 166 214 239 256 287 298 329 366 396 428 444 464 512 539 554 580 
4 54 76 89 131 152 167 215 240 257 288 299 330 367 397 429 445 465 513 540 555 581 
5 28 77 90 132 153 168 216 241 258 289 300 331 368 398 430 446 466 487 514 556 582 
6 29 78 91 133 154 169 190 242 259 290 301 332 369 399 431 447 467 488 515 557 583 
7 30 79 92 134 155 170 191 243 260 291 302 333 370 400 432 448 468 489 516 558 584 
8 31 80 93 135 156 171 192 217 261 292 303 334 371 401 406 449 469 490 517 559 585 
9 32 81 94 109 157 172 193 218 262 293 304 335 372 402 407 450 470 491 518 560 586 
10 33 55 95 110 158 173 194 219 263 294 305 336 373 403 408 451 471 492 519 561 587 
11 34 56 96 111 159 174 195 220 264 295 306 337 374 404 409 452 472 493 520 562 588 
12 35 57 97 112 160 175 196 221 265 296 307 338 375 405 410 453 473 494 521 563 589 
13 36 58 98 113 161 176 197 222 266 297 308 339 376 379 411 454 474 495 522 564 590 
14 37 59 99 114 162 177 198 223 267 271 309 340 377 380 412 455 475 496 523 565 591 
15 38 60 100 115 136 178 199 224 268 272 310 341 378 381 413 456 476 497 524 566 592 
16 39 61 101 116 137 179 200 225 269 273 311 342 352 382 414 457 477 498 525 567 593 
17 40 62 102 117 138 180 201 226 270 274 312 343 353 383 415 458 478 499 526 541 594 
4 40 66 96 120 161 168 208 217 253 273 324 351 362 403 413 447 480 491 516 568 595 
5 41 67 97 121 162 169 209 218 254 274 298 325 363 404 414 448 481 492 517 569 596 
6 42 68 98 122 136 170 210 219 255 275 299 326 364 405 415 449 482 493 518 570 597 
7 43 69 99 123 137 171 211 220 256 276 300 327 365 379 416 450 483 494 519 571 598 
8 44 70 100 124 138 172 212 221 257 277 301 328 366 380 417 451 484 495 520 572 599 
9 45 71 101 125 139 173 213 222 258 278 302 329 367 381 418 452 485 496 521 573 600 
10 46 72 102 126 140 174 214 223 259 279 303 330 368 382 419 453 486 497 522 574 601 
11 47 73 103 127 141 175 215 224 260 280 304 331 369 383 420 454 460 498 523 575 602 
12 48 74 104 128 142 176 216 225 261 281 305 332 370 384 421 455 461 499 524 576 603 
13 49 75 105 129 143 177 190 226 262 282 306 333 371 385 422 456 462 500 525 577 604 
14 50 76 106 130 144 178 191 227 263 283 307 334 372 386 423 457 463 501 526 578 605 
15 51 77 107 131 145 179 192 228 264 284 308 335 373 387 424 458 464 502 527 579 606 
16 52 78 108 132 146 180 193 229 265 285 309 336 374 388 425 459 465 503 528 580 607 
17 53 79 82 133 147 181 194 230 266 286 310 337 375 389 426 433 466 504 529 581 608 
18 54 80 83 134 148 182 195 231 267 287 311 338 376 390 427 434 467 505 530 582 609 
19 28 81 84 135 149 183 196 232 268 288 312 339 377 391 428 435 468 506 531 583 610 
20 29 55 85 109 150 184 197 233 269 289 313 340 378 392 429 436 469 507 532 584 611 
21 30 56 86 110 151 185 198 234 270 290 314 341 352 393 430 437 470 508 533 585 612 
22 31 57 87 111 152 186 199 235 244 291 315 342 353 394 431 438 471 509 534 586 613 
23 32 58 88 112 153 187 200 236 245 292 316 343 354 395 432 439 472 510 535 587 614 
24 33 59 89 113 154 188 201 237 246 293 317 344 355 396 406 440 473 511 536 588 615 
25 34 60 90 114 155 189 202 238 247 294 318 345 356 397 407 441 474 512 537 589 616 
26 35 61 91 115 156 163 203 239 248 295 319 346 357 398 408 442 475 513 538 590 617 
27 36 62 92 116 157 164 204 240 249 296 320 347 358 399 409 443 476 487 539 591 618 
1 37 63 93 117 158 165 205 241 250 297 321 348 359 400 410 444 477 488 540 592 619 
2 38 64 94 118 159 166 206 242 251 271 322 349 360 401 411 445 478 489 514 593 620 
3 39 65 95 119 160 167 207 243 252 272 323 350 361 402 412 446 479 490 515 594 621 
23 44 59 85 119 157 175 195 238 258 290 303 360 384 424 444 465 492 529 541 595 622 
24 45 60 86 120 158 176 196 239 259 291 304 361 385 425 445 466 493 530 542 596 623 
25 46 61 87 121 159 177 197 240 260 292 305 362 386 426 446 467 494 531 543 597 624 
26 47 62 88 122 160 178 198 241 261 293 306 363 387 427 447 468 495 532 544 598 625 
27 48 63 89 123 161 179 199 242 262 294 307 364 388 428 448 469 496 533 545 599 626 
1 49 64 90 124 162 180 200 243 263 295 308 365 389 429 449 470 497 534 546 600 627 
2 50 65 91 125 136 181 201 217 264 296 309 366 390 430 450 471 498 535 547 601 628 
3 51 66 92 126 137 182 202 218 265 297 310 367 391 431 451 472 499 536 548 602 629 
4 52 67 93 127 138 183 203 219 266 271 311 368 392 432 452 473 500 537 549 603 630 
5 53 68 94 128 139 184 204 220 267 272 312 369 393 406 453 474 501 538 550 604 631 
6 54 69 95 129 140 185 205 221 268 273 313 370 394 407 454 475 502 539 551 605 632 
7 28 70 96 130 141 186 206 222 269 274 314 371 395 408 455 476 503 540 552 606 633 
8 29 71 97 131 142 187 207 223 270 275 315 372 396 409 456 477 504 514 553 607 634 
9 30 72 98 132 143 188 208 224 244 276 316 373 397 410 457 478 505 515 554 608 635 
10 31 73 99 133 144 189 209 225 245 277 317 374 398 411 458 479 506 516 555 609 636 
11 32 74 100 134 145 163 210 226 246 278 318 375 399 412 459 480 507 517 556 610 637 
12 33 75 101 135 146 164 211 227 247 279 319 376 400 413 433 481 508 518 557 611 638 
13 34 76 102 109 147 165 212 228 248 280 320 377 401 414 434 482 509 519 558 612 639 
14 35 77 103 110 148 166 213 229 249 281 321 378 402 415 435 483 510 520 559 613 640 
15 36 78 104 111 149 167 214 230 250 282 322 352 403 416 436 484 511 521 560 614 641 
16 37 79 105 112 150 168 215 231 251 283 323 353 404 417 437 485 512 522 561 615 642 
17 38 80 106 113 151 169 216 232 252 284 324 354 405 418 438 486 513 523 562 616 643 
18 39 81 107 114 152 170 190 233 253 285 298 355 379 419 439 460 487 524 563 617 644 
19 40 55 108 115 153 171 191 234 254 286 299 356 380 420 440 461 488 525 564 618 645 
20 41 56 82 116 154 172 192 235 255 287 300 357 381 421 441 462 489 526 565 619 646 
21 42 57 83 117 155 173 193 236 256 288 301 358 382 422 442 463 490 527 566 620 647 
22 43 58 84 118 156 174 194 237 257 289 302 359 383 423 443 464 491 528 567 621 648 
8 35 69 96 113 152 179 214 241 254 272 305 340 358 389 432 441 478 508 528 542 622 
9 36 70 97 114 153 180 215 242 255 273 306 341 359 390 406 442 479 509 529 543 623 
10 37 71 98 115 154 181 216 243 256 274 307 342 360 391 407 443 480 510 530 544 624 
11 38 72 99 116 155 182 190 217 257 275 308 343 361 392 408 444 481 511 531 545 625 
12 39 73 100 117 156 183 191 218 258 276 309 344 362 393 409 445 482 512 532 546 626 
13 40 74 101 118 157 184 192 219 259 277 310 345 363 394 410 446 483 513 533 547 627 
14 41 75 102 119 158 185 193 220 260 278 311 346 364 395 411 447 484 487 534 548 628 
15 42 76 103 120 159 186 194 221 261 279 312 347 365 396 412 448 485 488 535 549 629 
16 43 77 104 121 160 187 195 222 262 280 313 348 366 397 413 449 486 489 536 550 630 
17 44 78 105 122 161 188 196 223 263 281 314 349 367 398 414 450 460 490 537 551 631 
18 45 79 106 123 162 189 197 224 264 282 315 350 368 399 415 451 461 491 538 552 632 
19 46 80 107 124 136 163 198 225 265 283 316 351 369 400 416 452 462 492 539 553 633 
20 47 81 108 125 137 164 199 226 266 284 317 325 370 401 417 453 463 493 540 554 634 
21 48 55 82 126 138 165 200 227 267 285 318 326 371 402 418 454 464 494 514 555 635 
22 49 56 83 127 139 166 201 228 268 286 319 327 372 403 419 455 465 495 515 556 636 
23 50 57 84 128 140 167 202 229 269 287 320 328 373 404 420 456 466 496 516 557 637 
24 51 58 85 129 141 168 203 230 270 288 321 329 374 405 421 457 467 497 517 558 638 
25 52 59 86 130 142 169 204 231 244 289 322 330 375 379 422 458 468 498 518 559 639 
26 53 60 87 131 143 170 205 232 245 290 323 331 376 380 423 459 469 499 519 560 640 
27 54 61 88 132 144 171 206 233 246 291 324 332 377 381 424 433 470 500 520 561 641 
1 28 62 89 133 145 172 207 234 247 292 298 333 378 382 425 434 471 501 521 562 642 
2 29 63 90 134 146 173 208 235 248 293 299 334 352 383 426 435 472 502 522 563 643 
3 30 64 91 135 147 174 209 236 249 294 300 335 353 384 427 436 473 503 523 564 644 
4 31 65 92 109 148 175 210 237 250 295 301 336 354 385 428 437 474 504 524 565 645 
5 32 66 93 110 149 176 211 238 251 296 302 337 355 386 429 438 475 505 525 566 646 
6 33 67 94 111 150 177 212 239 252 297 303 338 356 387 430 439 476 506 526 567 647 
7 34 68 95 112 151 178 213 240 253 271 304 339 357 388 431 440 477 507 527 541 648 

"""

# --- Global variable ---
LDPC_PARAMS_GLOBAL = None

# --- ALIST Parser Function (Modified to parse string) ---
def parse_alist_string(alist_string):
    """Parses LDPC parameters from an ALIST string."""
    ldpc_params = {}
    if not alist_string or alist_string == "PASTE HERE THE ALIST":
        print("Error: ALIST string is empty or still contains the placeholder.")
        return None
    try:
        lines = alist_string.strip().splitlines() # Split string into lines
        if len(lines) < 4: raise ValueError("ALIST string too short (minimum 4 lines needed)")

        n, m = map(int, lines[0].strip().split())
        max_col_weight_hdr, max_row_weight_hdr = map(int, lines[1].strip().split())
        vnode_deg_list = list(map(int, lines[2].strip().split()))
        cnode_deg_list = list(map(int, lines[3].strip().split()))

        if len(vnode_deg_list) != n or len(cnode_deg_list) != m:
            raise ValueError(f"Degree list lengths mismatch: Got {len(vnode_deg_list)} (exp {n}) vars, {len(cnode_deg_list)} (exp {m}) checks")

        rows_H = []; cols_H = []; line_offset = 4
        if len(lines) < line_offset + n: # Check if enough lines exist for variable nodes
            raise ValueError(f"ALIST string too short: Expected {line_offset + n} lines, got {len(lines)}")

        var_node_conn_tmp = [[] for _ in range(n)]; chk_node_conn_tmp = [[] for _ in range(m)]
        for col_idx in range(n):
            line_num = line_offset + col_idx
            parts = lines[line_num].strip().split()
            if not parts: continue # Allow empty lines if degree is 0? Check ALIST standard. Usually not expected.
            check_nodes = list(map(int, parts))
            check_nodes_0based = [cn - 1 for cn in check_nodes if cn > 0] # ALIST is 1-based
            var_node_conn_tmp[col_idx] = check_nodes_0based

            # Build COO sparse matrix data
            rows_H.extend(check_nodes_0based)
            cols_H.extend([col_idx] * len(check_nodes_0based))

            # Build check node connections (reverse lookup)
            for cn_idx in check_nodes_0based:
                 if cn_idx < m:
                     chk_node_conn_tmp[cn_idx].append(col_idx)
                 else:
                     print(f"Warning: Check node index {cn_idx+1} out of bounds (m={m}) referenced by var node {col_idx+1}.")


        data = np.ones(len(rows_H), dtype=int)
        H_coo = sp.coo_matrix((data, (rows_H, cols_H)), shape=(m, n))
        H_csc = H_coo.tocsc() # Convert to CSC for efficient column operations (syndrome)

        print(f"Parsed ALIST string: n={n}, m={m}. H shape={H_csc.shape}, nnz={H_csc.nnz}")
        if H_csc.nnz != len(rows_H):
            print(f"Warning: Possible duplicate entries in ALIST? NNZ mismatch ({H_csc.nnz} vs {len(rows_H)})")

        if n != EXPECTED_N or m != EXPECTED_M:
            print(f"!!! WARNING: ALIST dimensions ({n}x{m}) != expected ({EXPECTED_N}x{EXPECTED_M}) !!!")

        # Verify degree lists match constructed H matrix
        actual_vnode_degrees = np.diff(H_csc.indptr)
        if not np.array_equal(actual_vnode_degrees, vnode_deg_list):
             print(f"Warning: Variable node degree list mismatch. Header: {vnode_deg_list[:10]}..., Actual: {actual_vnode_degrees[:10]}...")
        actual_cnode_degrees = np.diff(H_csc.tocsr().indptr) # Need CSR for row degrees
        if not np.array_equal(actual_cnode_degrees, cnode_deg_list):
             print(f"Warning: Check node degree list mismatch. Header: {cnode_deg_list[:10]}..., Actual: {actual_cnode_degrees[:10]}...")


        var_neighbors, chk_neighbors = get_neighbors(H_csc) # Pre-calculate neighbors

        ldpc_params = {
            'n': n, 'm': m, 'k': n - m,
            'H': H_csc,
            'var_neighbors': var_neighbors, 'chk_neighbors': chk_neighbors,
            'max_vnode_deg': max_col_weight_hdr, 'max_cnode_deg': max_row_weight_hdr,
            'vnode_deg_list': np.array(vnode_deg_list, dtype=int),
            'cnode_deg_list': np.array(cnode_deg_list, dtype=int)
        }
        return ldpc_params

    # Removed FileNotFoundError
    except Exception as e:
        print(f"Error parsing ALIST string: {e}")
        traceback.print_exc()
        return None

# --- Get Neighbors Function ---
def get_neighbors(H_sparse):
    # ... (Unchanged - safe to keep) ...
    if not sp.issparse(H_sparse): H_sparse = sp.csc_matrix(H_sparse)
    m, n = H_sparse.shape; H_coo = H_sparse.tocoo()
    check_indices = H_coo.row; var_indices = H_coo.col
    var_neighbors = [[] for _ in range(n)]; chk_neighbors = [[] for _ in range(m)]
    for r, c in zip(check_indices, var_indices): var_neighbors[c].append(r); chk_neighbors[r].append(c)
    return var_neighbors, chk_neighbors

# --- Image Pattern Generator ---
def create_image_pattern(pattern_name, width, height):
    # ... (Unchanged - safe to keep) ...
    img = np.zeros((height, width), dtype=int)
    if pattern_name == 'square': pad_h, pad_w = height//4, width//4; h_start, h_end = pad_h, height-pad_h; w_start, w_end = pad_w, width-pad_w; img[h_start:h_end, w_start:w_end]=1 if h_start<h_end and w_start<w_end else 0; img[height//2,width//2]=1 if h_start>=h_end or w_start>=w_end else img[height//2,width//2]
    elif pattern_name == 'cross': center_h, center_w = height//2, width//2; thick = max(1, min(height,width)//5); h_s=max(0,center_h-thick//2); h_e=min(height,center_h+(thick+1)//2); w_s=max(0,center_w-thick//2); w_e=min(width,center_w+(thick+1)//2); img[h_s:h_e,:]=1; img[:,w_s:w_e]=1
    elif pattern_name == 'checkerboard': img = np.fromfunction(lambda r, c: (r+c)%2==0, (height, width), dtype=int)
    elif pattern_name == 'random_msg': img = np.random.randint(0, 2, size=(height, width))
    return img

# --- Systematic Encoder (Robust Version) ---
def encode_systematic_robust(message_bits_k, H):
    # ... (Unchanged - safe to keep) ...
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
    # ... (Unchanged - safe to keep) ...
    noisy_codeword = codeword.copy(); flips = 0; indices_flipped = []
    for i in range(len(noisy_codeword)):
        if random.random() < p: noisy_codeword[i] = 1 - noisy_codeword[i]; flips += 1; indices_flipped.append(i)
    return noisy_codeword, flips, indices_flipped
def calculate_syndrome(H_input, codeword):
    # ... (Unchanged - safe to keep) ...
    if H_input is None or codeword is None or H_input.shape[1] != len(codeword): return np.array([-1], dtype=int)
    codeword_int = np.array(codeword, dtype=int)
    if sp.issparse(H_input): syndrome = H_input.dot(codeword_int) % 2
    else: syndrome = H_input @ codeword_int % 2
    return syndrome
def calculate_bsc_llrs(noisy_codeword, p):
    # ... (Unchanged - safe to keep) ...
    p_stable = max(1e-9, min(p, 1.0 - 1e-9)); L0 = math.log((1.0 - p_stable) / p_stable)
    llrs = (1 - 2 * noisy_codeword) * L0; MAX_LLR_VALUE = 50; llrs = np.clip(llrs, -MAX_LLR_VALUE, MAX_LLR_VALUE)
    return llrs
def safe_atanh(x):
    # ... (Unchanged - safe to keep) ...
    clipped_x = np.clip(x, -0.9999999, 0.9999999); return np.arctanh(clipped_x)

# --- Custom LLR Belief Propagation Decoder ---
def decode_llrbp_for_dash(noisy_codeword, H, var_neighbors, chk_neighbors, max_iter, channel_prob):
    # ... (Unchanged - safe to keep) ...
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
    # ... (Unchanged - safe to keep) ...
    if image_array is None or image_array.size == 0: fig = go.Figure().update_layout(title=title + " (No Data)", xaxis_visible=False, yaxis_visible=False, template='plotly_white'); return fig
    img_numeric = image_array.astype(float); fig = px.imshow(img_numeric, binary_string=False, color_continuous_scale='gray_r', aspect='equal')
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), coloraxis_showscale=False, margin=dict(l=5, r=5, t=5, b=5));
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>color: %{z}<extra></extra>")
    return fig

# --- H Matrix Plot Helper ---
def create_h_matrix_fig(H_sparse):
    # ... (Unchanged - safe to keep) ...
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
# Assumes filename is app.py for deployment
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server # Expose server variable for platforms like Render (using Gunicorn)

# --- App Layout (Tabs added) ---
# ... (Layout is unchanged - safe to keep) ...
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
    # Generate H plot even initially if params are loaded
    h_fig = create_h_matrix_fig(LDPC_PARAMS_GLOBAL['H'] if LDPC_PARAMS_GLOBAL else None)

    if n_clicks == 0:
        err_msg = "Click 'Run Simulation' to start."
        # Display H matrix plot even before first run if available
        return (no_data_fig, no_data_fig, h_fig, no_plot_fig, no_plot_fig,
                err_msg, None, 0, initial_marks, True, None, initial_noisy_info)

    if LDPC_PARAMS_GLOBAL is None:
         # This should ideally not happen if startup parsing worked
         error_msg = f"FATAL ERROR: LDPC parameters not loaded."
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
    # Update summary to reflect embedded data source
    summary_lines.append(f"Using WiFi N={n_val}, K={k_val}, R={actual_rate:.3f} (from embedded ALIST)")
    summary_lines.append(f"Image Pattern: '{pattern}' ({IMAGE_WIDTH}x{IMAGE_HEIGHT})")

    # Create H matrix plot (already created above, just ensure it's passed)
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
        # Return H matrix plot even on error
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

    # Call decoder
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
    if final_decoded_cw is None: final_decoded_cw = noisy_cw.copy(); summary_lines.append("Warning: Using noisy CW due to decoder error.")
    decoded_msg_bits_final = final_decoded_cw[:k_val]
    if len(decoded_msg_bits_final) != k_val:
        summary_lines.append(f"Error: Final decoded message length ({len(decoded_msg_bits_final)}) != k ({k_val})");
        err_fig = create_image_fig(None, "Decode Length Error")
        # Still return plots generated so far
        return (fig_orig, fig_noisy, fig_h_matrix, no_plot_fig, no_plot_fig,
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

    # --- Generate Syndrome Plot ---
    # ... (Syndrome plot logic unchanged - safe to keep) ...
    fig_syndrome = go.Figure()
    iterations_synd = history.get('iteration', [])
    syndrome_weights = history.get('syndrome_weight', [])
    if iterations_synd and len(iterations_synd) == len(syndrome_weights):
        fig_syndrome.add_trace(go.Scattergl(x=iterations_synd, y=syndrome_weights, mode='lines+markers', name='Syndrome Weight'))
        fig_syndrome.update_layout(title="Syndrome Weight vs Iteration", xaxis_title="Decoder Iteration", yaxis_title="Syndrome Weight", yaxis=dict(range=[-0.5, m_val + 0.5]), uirevision=n_clicks)
    else: fig_syndrome.update_layout(title="Syndrome Weight Plot (No data)")

    # --- Generate Focused LLR Plot ---
    # ... (LLR plot logic unchanged - safe to keep) ...
    fig_llr = go.Figure()
    iterations_llr = history.get('iteration', [])
    llrs_history_list = history.get('total_llrs', [])
    if iterations_llr and llrs_history_list and len(iterations_llr) == len(llrs_history_list):
         try:
             llrs_history_array = np.array(llrs_history_list) # Shape (iters+1, n)
             if llrs_history_array.ndim == 2 and llrs_history_array.shape[1] == n_val:
                 signs = np.sign(llrs_history_array)
                 oscillating_indices = np.where(np.any(signs != signs[0,:], axis=0))[0]
                 always_pos_indices = np.where(np.all(signs >= 0, axis=0))[0]
                 always_neg_indices = np.where(np.all(signs <= 0, axis=0))[0]
                 avg_pos_llrs = np.mean(llrs_history_array[:, always_pos_indices], axis=1) if len(always_pos_indices) > 0 else np.full(len(iterations_llr), np.nan)
                 avg_neg_llrs = np.mean(llrs_history_array[:, always_neg_indices], axis=1) if len(always_neg_indices) > 0 else np.full(len(iterations_llr), np.nan)
                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_pos_llrs, mode='lines', name='Avg Always Pos LLR', line=dict(color='blue', width=3)))
                 fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=avg_neg_llrs, mode='lines', name='Avg Always Neg LLR', line=dict(color='red', width=3)))
                 max_oscillating_to_plot = 30
                 num_oscillating = len(oscillating_indices)
                 indices_to_plot = oscillating_indices
                 plot_title = f"LLR Evolution"
                 if num_oscillating > max_oscillating_to_plot:
                     indices_to_plot = np.random.choice(oscillating_indices, size=max_oscillating_to_plot, replace=False)
                 for bit_index in indices_to_plot:
                     fig_llr.add_trace(go.Scattergl(x=iterations_llr, y=llrs_history_array[:, bit_index], mode='lines', name=f'Bit {bit_index}', line=dict(width=1, dash='dot'), opacity=0.7))
                 fig_llr.update_layout(title=plot_title, xaxis_title="Decoder Iteration", yaxis_title="Total LLR", uirevision=n_clicks)
             else:
                 raise ValueError(f"LLR history array shape mismatch: Expected (iters, {n_val}), got {llrs_history_array.shape}")
         except Exception as e:
             print(f"Error creating LLR plot: {e}")
             traceback.print_exc()
             fig_llr.update_layout(title="LLR Plot Error")
    else:
        fig_llr.update_layout(title="LLR Plot (No data)")

    # --- Prepare Slider Outputs ---
    # ... (Slider logic unchanged - safe to keep) ...
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


# --- Parse ALIST Data on Module Load ---
# This code now runs when the script is imported by Gunicorn OR run directly
print(f"Attempting to parse embedded ALIST string on module load...")
LDPC_PARAMS_GLOBAL = parse_alist_string(wifi_648_r083_alist_content) # Use the string variable

if LDPC_PARAMS_GLOBAL is None:
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     print("ERROR: Failed to load or parse LDPC parameters from embedded string.")
     print("Ensure 'wifi_648_r083_alist_content' is correctly pasted.")
     print("Application cannot start. Exiting.")
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     sys.exit(1) # Exit if parameters fail to load - app is useless without them

print(f"Successfully loaded LDPC parameters from embedded ALIST string.")
print(f"LDPC Params Keys: {list(LDPC_PARAMS_GLOBAL.keys())}") # Add a check print


# --- Run Development Server (Only when executed directly) ---
if __name__ == '__main__':
    print(f"Starting Dash server for local development on http://127.0.0.1:8050/ ...")
    # debug=True should be False in production, but OK for testing Render deploy
    # host='0.0.0.0' makes it accessible on the network, often needed for containers/servers
    app.run(debug=False, port=8050, host='0.0.0.0') # Use host='0.0.0.0' and disable debug for deployment testing
