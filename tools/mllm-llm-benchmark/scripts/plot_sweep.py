import sys, os, csv, math
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_context/context_sweep_v2.csv"
out_dir  = sys.argv[2] if len(sys.argv) > 2 else "snapshots"
os.makedirs(out_dir, exist_ok=True)

def to_float(x):
    try: return float(x)
    except: return float("nan")

def to_int(x):
    try: return int(float(x))
    except: return 0

rows = []
with open(csv_path, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# normalize numeric fields
num_fields = ["ctx_len","pp","tg","threads","ttft_ms","prefill_ms","decode_ms","decode_ms_per_tok","peak_rss_kb","kv_est_kb"]
for row in rows:
    for k in num_fields:
        if k in row:
            row[k] = to_float(row[k])

# write summary
stamp = os.path.splitext(os.path.basename(csv_path))[0]
summary_path = os.path.join(out_dir, f"{stamp}.summary.csv")
fieldnames = ["ts","git","arch","model","mode","ctx_len","pp","tg","threads",
              "ttft_ms","prefill_ms","decode_ms","decode_ms_per_tok",
              "peak_rss_kb","kv_est_kb","peak_rss_gb","kv_est_mb"]
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in sorted(rows, key=lambda x: (x.get("mode",""), x.get("ctx_len",0))):
        peak_rss_kb = row.get("peak_rss_kb", float("nan"))
        kv_est_kb   = row.get("kv_est_kb", float("nan"))
        out = {k: row.get(k, "") for k in fieldnames}
        out["peak_rss_gb"] = (peak_rss_kb / (1024*1024)) if peak_rss_kb==peak_rss_kb else ""
        out["kv_est_mb"]   = (kv_est_kb / 1024) if kv_est_kb==kv_est_kb else ""
        w.writerow(out)

def plot_mode(mode, xkey, ykey, ylabel, fname):
    xs, ys = [], []
    for row in rows:
        if row.get("mode","") != mode: 
            continue
        x = row.get(xkey, float("nan"))
        y = row.get(ykey, float("nan"))
        if x==x and y==y:
            xs.append(x); ys.append(y)
    if not xs:
        return
    pts = sorted(zip(xs, ys), key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xkey)
    plt.ylabel(ylabel)
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=180)
    plt.close()

plot_mode("prefill_ttft", "ctx_len", "ttft_ms", "TTFT (ms)", f"{stamp}.prefill_ttft.ttft_ms.png")
plot_mode("prefill_ttft", "ctx_len", "prefill_ms", "Prefill latency (ms)", f"{stamp}.prefill_ttft.prefill_ms.png")
plot_mode("decode_heavy", "ctx_len", "decode_ms_per_tok", "Decode latency per token (ms)", f"{stamp}.decode_heavy.decode_ms_per_tok.png")
plot_mode("decode_heavy", "ctx_len", "decode_ms", "Decode latency total (ms)", f"{stamp}.decode_heavy.decode_ms.png")

# memory plots
mem = {}
for row in rows:
    ctx_len = row.get("ctx_len", float("nan"))
    if ctx_len != ctx_len: 
        continue
    ctx_len = int(ctx_len)
    peak = row.get("peak_rss_kb", float("nan"))
    kv   = row.get("kv_est_kb", float("nan"))
    cur = mem.get(ctx_len, {"peak": float("nan"), "kv": float("nan")})
    if peak==peak and (cur["peak"]!=cur["peak"] or peak>cur["peak"]): cur["peak"]=peak
    if kv==kv and (cur["kv"]!=cur["kv"] or kv>cur["kv"]): cur["kv"]=kv
    mem[ctx_len]=cur

ctx_lens = sorted(mem.keys())
peak_gb = [(mem[c]["peak"]/(1024*1024)) if mem[c]["peak"]==mem[c]["peak"] else float("nan") for c in ctx_lens]
kv_mb   = [(mem[c]["kv"]/1024) if mem[c]["kv"]==mem[c]["kv"] else float("nan") for c in ctx_lens]

plt.figure()
plt.plot(ctx_lens, peak_gb, marker="o")
plt.xlabel("ctx_len")
plt.ylabel("Peak RSS (GB)")
plt.xscale("log", base=2)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{stamp}.memory.peak_rss_gb.png"), dpi=180)
plt.close()

plt.figure()
plt.plot(ctx_lens, kv_mb, marker="o")
plt.xlabel("ctx_len")
plt.ylabel("KV estimate (MB)")
plt.xscale("log", base=2)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{stamp}.memory.kv_est_mb.png"), dpi=180)
plt.close()

print("Wrote:")
print(" ", summary_path)
print(" ", os.path.join(out_dir, f"{stamp}.prefill_ttft.ttft_ms.png"))
print(" ", os.path.join(out_dir, f"{stamp}.decode_heavy.decode_ms_per_tok.png"))
print(" ", os.path.join(out_dir, f"{stamp}.memory.peak_rss_gb.png"))
print(" ", os.path.join(out_dir, f"{stamp}.memory.kv_est_mb.png"))
