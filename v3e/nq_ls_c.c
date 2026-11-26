/* nq_ls_c: min-conflicts local search for blocked n-Queens in C. Same algorithm as lsnb_bnq / bnq_init / bnq_steps in nq_v3.py:
   greedy row-by-row start (min of col + diag1 + diag2 conflicts + 5*blocked, ties uniformly at random), then repeat: pick a random
   conflicted queen (conflict = col+d1+d2-3+blocked > 0), move it to a min-conflict column of its row (ties random), noise 0;
   restart after 20000 moves; clock checked every 500 moves. The numba kernel draws one extra uniform per move for the noise test
   (noise 0, never taken); this program draws it too, so with the same RNG stream both give identical trajectories (selftest_local.py).
   RNG: xorshift64* (numba uses Mersenne Twister, so single runs differ from ls_nb; the algorithm is the same).
   usage: nq_ls_c BOARDFILE SEED BUDGET_SECONDS     BOARDFILE: line 1 = n, then n lines of n chars, '1' = blocked
   stdout: "SAT <internal_seconds> q0 q1 ... q(n-1)"  or  "TO <internal_seconds>" ; exit code 0 (2 on bad input). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static uint64_t S;
static double u01(void) { S ^= S >> 12; S ^= S << 25; S ^= S >> 27; return (double)((S * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0); }
static int rnd(int k) { return (int)(u01() * k); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    double t0 = now();
    if (argc != 4) { fprintf(stderr, "usage: nq_ls_c BOARDFILE SEED BUDGET\n"); return 2; }
    FILE *f = fopen(argv[1], "r"); int n;
    if (!f || fscanf(f, "%d", &n) != 1 || n < 1) { fprintf(stderr, "bad board file\n"); return 2; }
    long long seed = atoll(argv[2]); double budget = atof(argv[3]);
    int *H = malloc(sizeof(int) * n * n), *pos = malloc(sizeof(int) * n), *col = malloc(sizeof(int) * n), *d1 = malloc(sizeof(int) * 2 * n), *d2 = malloc(sizeof(int) * 2 * n);
    char *line = malloc(n + 8);
    for (int r = 0; r < n; r++) {
        if (fscanf(f, "%s", line) != 1 || (int)strlen(line) != n) { fprintf(stderr, "bad board row\n"); return 2; }
        for (int c = 0; c < n; c++) H[r * n + c] = line[c] == '1';
    }
    fclose(f);
    S = ((uint64_t)seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) | 1ULL; for (int i = 0; i < 8; i++) u01();
    double end = t0 + budget;
    for (;;) {
        if (now() > end) { printf("TO %.4f\n", now() - t0); return 0; }
        memset(col, 0, sizeof(int) * n); memset(d1, 0, sizeof(int) * 2 * n); memset(d2, 0, sizeof(int) * 2 * n);
        for (int r = 0; r < n; r++) {
            int best = 1 << 30, k = 0, ch = 0;
            for (int c = 0; c < n; c++) {
                int x = col[c] + d1[r - c + n] + d2[r + c] + 5 * H[r * n + c];
                if (x < best) { best = x; k = 1; ch = c; } else if (x == best) { k++; if (rnd(k) == 0) ch = c; }
            }
            pos[r] = ch; col[ch]++; d1[r - ch + n]++; d2[r + ch]++;
        }
        for (int step = 0; step < 20000; step++) {
            if (step > 0 && step % 500 == 0 && now() > end) { printf("TO %.4f\n", now() - t0); return 0; }
            int nb = 0, r = -1;
            for (int i = 0; i < n; i++)
                if (col[pos[i]] + d1[i - pos[i] + n] + d2[i + pos[i]] - 3 + H[i * n + pos[i]] > 0) { nb++; if (rnd(nb) == 0) r = i; }
            if (nb == 0) {
                printf("SAT %.4f", now() - t0); for (int i = 0; i < n; i++) printf(" %d", pos[i]); printf("\n"); return 0;
            }
            col[pos[r]]--; d1[r - pos[r] + n]--; d2[r + pos[r]]--;
            (void)u01();   /* mirrors the numba noise draw (noise = 0) */
            int best = 1 << 30, k = 0, ch = 0;
            for (int c = 0; c < n; c++) {
                int x = col[c] + d1[r - c + n] + d2[r + c] + 5 * H[r * n + c];
                if (x < best) { best = x; k = 1; ch = c; } else if (x == best) { k++; if (rnd(k) == 0) ch = c; }
            }
            pos[r] = ch; col[ch]++; d1[r - ch + n]++; d2[r + ch]++;
        }
    }
}
