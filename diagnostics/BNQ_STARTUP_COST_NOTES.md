# Start-up cost of the compiled local search on blocked n-Queens (diagnostics, 26 Nov 2026)

Two small Kaggle CPU notebooks (bnq_cost_diag.ipynb and bnq_cost_diag2.ipynb in this folder) were run to investigate the roughly one-second minimum solve time of ls_nb on blocked n-Queens seen in main10 and the confirmation stage. Values below were read from the notebook logs at the time; the logs themselves are not archived.

1. bnq_cost_diag (direct calls after the harness warm-up on an 8 x 8 board): the first call in the parent process on a real-size board took 0.60 s; every later call, in the parent and in a forked child, took 0.0001 to 0.003 s.
2. bnq_cost_diag2 (the actual harness, smoke grid, ls_nb on bnq, T = 2 s): with four parallel jobs the solved runs took 0.98 to 1.12 s; with one job 0.60 to 0.62 s.

Reading: a one-time start-up cost (most likely just-in-time compilation of a kernel specialisation not covered by the warm-up) is paid by every forked child. This is an inference from timing, not a confirmed diagnosis. Stage v3e corrects the warm-up and logs the first-call times per child.
