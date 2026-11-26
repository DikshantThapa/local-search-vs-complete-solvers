"""E5: CP-SAT tuning on bnq for v3e. Reuses tune_cpsat.tune_cp (unchanged, v3d) with the 8 documented variants, cells bnq n=100 p=.90 and .94,
tuning seeds 2000-2002, T = 5 s, 1 worker, winner rule of DESIGN_NOTE_V3D section 4 (>= 1 solved more over the 6 runs or >= 10% PAR2, else default).
Writes tuned_cpsat_bnq.json (always; 'params' empty = no signal) and tune_log_v3e.csv.  Usage: python tune_v3e.py [--jobs 4] [--T 5]"""
import argparse, csv, json, os, time
import tune_cpsat as tc
import nq_v3d as d
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--T", type=float, default=5.0); a = ap.parse_args()
    tc.CELLS = {"bnq": [(100, .90), (100, .94)]}
    f = open(os.path.join(d.HERE, "tune_log_v3e.csv"), "a", newline=""); log = csv.writer(f)
    if os.path.getsize(f.name) == 0: log.writerow(["part", "family", "size", "param", "seed", "setting", "solved", "t_or_2T"])
    t = time.time(); tmp = os.path.join(d.HERE, "tuned_cpsat.json")
    tc.tune_cp(a, log); f.flush()
    os.replace(tmp, os.path.join(d.HERE, "tuned_cpsat_bnq.json"))     # tune_cp writes tuned_cpsat.json; rename so v3d's file name is never reused
    r = json.load(open(os.path.join(d.HERE, "tuned_cpsat_bnq.json")))["bnq"]
    print("E5 result:", "SIGNAL variant=" + r["variant"] if r["params"] else "NO SIGNAL (default stays)", {v: x["solved"] for v, x in r["table"].items()}, f"{time.time() - t:.0f} s", flush=True)
