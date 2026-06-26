import json, numpy as np
from collections import defaultdict

mqt = json.load(open("results/core_mqt_perrun.json"))
rout = [v for v in mqt.values() if v and "error" not in v and v.get("routing")]

print("=== tab:audit  (makespan reduction by family, mean +/- std) ===")
order = ["qftentangled","qpeinexact","qpeexact","ae","qft","qaoa"]
fam = defaultdict(list)
for r in rout: fam[r["family"]].append(r["audit_gain_pct"])
for f in order:
    if fam[f]:
        print(f"  {f:<14} n={len(fam[f])}  {np.mean(fam[f]):+.1f} +/- {np.std(fam[f]):.1f}")
allg = [r["audit_gain_pct"] for r in rout]
print(f"  ALL {len(rout)} routing-meaningful: {np.mean(allg):+.1f} +/- {np.std(allg):.1f}")

print("\n=== tab:pool-decomp  (rescore / algorithm / total / frac) by family ===")
for f in order:
    rs = [r for r in rout if r["family"]==f]
    if not rs: continue
    resc = np.mean([r["rescore_pct"] for r in rs])
    algo = np.mean([r["algorithm_pct"] for r in rs])
    tot  = np.mean([r["audit_gain_pct"] for r in rs])
    print(f"  {f:<14} rescore {resc:+.1f}  algo {algo:+.1f}  total {tot:+.1f}  frac {100*resc/tot if tot else 0:.0f}%")
resc=np.mean([r["rescore_pct"] for r in rout]); algo=np.mean([r["algorithm_pct"] for r in rout]); tot=np.mean(allg)
print(f"  ALL {len(rout)}: rescore {resc:+.1f}  algo {algo:+.1f}  total {tot:+.1f}  frac {100*resc/tot:.0f}%")

print("\n=== channel split (raw scheduling -> total, absorption) ===")
raw=np.mean([r["ms_raw_gain_pct"] for r in rout])
print(f"  raw(scheduling) {raw:+.1f}%  total {tot:+.1f}%  absorption {tot-raw:+.1f}pp")

print("\n=== reachability ===")
nbeat=sum(1 for r in rout if r["reach_residual_pct"]<-0.5)
resid=np.mean([r["reach_residual_pct"] for r in rout])
print(f"  MS@K20 beats 200-pool on {nbeat}/{len(rout)}  mean residual {resid:+.1f}%")

# ESP
esp = json.load(open("results/core_esp_perrun.json"))
er = [v for v in esp.values() if v and "error" not in v and v.get("routing")]
mar=[v for v in er if v["esp_valid"].get("marrakesh") and v["esp_ratio"].get("marrakesh")]
ratios=[v["esp_ratio"]["marrakesh"] for v in mar]
heavy=sorted(mar,key=lambda r:-r['prod_2q'])[:36]
hr=[r['esp_ratio']['marrakesh'] for r in heavy]
nadd=sum(1 for v in er if v['n2q_delta']>0)
print("\n=== ESP ===")
print(f"  all routing cells: n={len(er)}  median {np.median(ratios):.2f}x")
print(f"  heavy (top-36 by 2q): median {np.median(hr):.2f}x  wins {sum(1 for x in hr if x>1)}/{len(heavy)}  worst {min(hr):.2f}x")
print(f"  adds gates on {nadd}/{len(er)}")

# absorption rate
ab = json.load(open("results/swap_absorption_perrun.json"))
abc=[v for v in ab.values() if v and 'ms_k20' in v]
upl=[v['ms_k20']['mean_absorbed_pct']-v['sabre_k20']['mean_absorbed_pct'] for v in abc]
mkr=[100*(v['sabre_k20']['mean_mk_opt']-v['ms_k20']['mean_mk_opt'])/v['sabre_k20']['mean_mk_opt'] for v in abc]
print("\n=== absorption ===")
print(f"  uplift mean {np.mean(upl):+.1f}pp median {np.median(upl):+.1f}pp  r={np.corrcoef(upl,mkr)[0,1]:.2f}  n={len(abc)}")

# synthetic
syn = json.load(open("results/real_full_perrun.json"))
sc=[v for v in syn.values() if v]
g=[v['gain_pct'] for v in sc]
sig=sum(1 for v in sc if v['gain_pct']>1 and v['p_ms_less']<0.05)
sfam=defaultdict(list)
for v in sc: sfam[v['family']].append(v['gain_pct'])
print("\n=== synthetic ===")
print(f"  mean {np.mean(g):+.1f}%  n={len(sc)}  significant {sig}/{len(sc)}")
for f in sorted(sfam): print(f"    {f}: {np.mean(sfam[f]):+.1f}%")
