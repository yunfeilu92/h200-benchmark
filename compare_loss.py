import torch, glob, sys

n_dir = "/nuplan/exp/exp/training/pluto/2026.03.23.06.38.30/checkpoints"
l_dir = "/nuplan/exp/exp/training/pluto/2026.03.23.07.54.26/checkpoints"

nearest = {}
for p in sorted(glob.glob(n_dir + "/epoch=*.ckpt")):
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    e = ckpt["epoch"]
    cb = [k for k in ckpt["callbacks"] if "ModelCheckpoint" in k][0]
    nearest[e] = ckpt["callbacks"][cb]["current_score"].item()

linear = {}
for p in sorted(glob.glob(l_dir + "/epoch=*.ckpt")):
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    e = ckpt["epoch"]
    cb = [k for k in ckpt["callbacks"] if "ModelCheckpoint" in k][0]
    linear[e] = ckpt["callbacks"][cb]["current_score"].item()

print("Epoch  | Nearest    | Linear     | Diff     | Pct")
print("-" * 55)
all_epochs = sorted(set(list(nearest.keys()) + list(linear.keys())))
for e in all_epochs:
    n = nearest.get(e)
    l = linear.get(e)
    n_str = "%.4f" % n if n else "     -"
    l_str = "%.4f" % l if l else "     -"
    if n and l:
        diff = l - n
        pct = diff / n * 100
        d_str = "%+.4f" % diff
        p_str = "%+.2f%%" % pct
    else:
        d_str = "     -"
        p_str = "     -"
    print("  %4d | %10s | %10s | %8s | %7s" % (e, n_str, l_str, d_str, p_str))

# Summary
if nearest and linear:
    n_final = nearest[max(nearest.keys())]
    l_final = linear[max(linear.keys())]
    print("\nNearest final (epoch %d): %.4f" % (max(nearest.keys()), n_final))
    print("Linear  final (epoch %d): %.4f" % (max(linear.keys()), l_final))
    if max(nearest.keys()) == max(linear.keys()):
        diff_pct = (l_final - n_final) / n_final * 100
        print("Difference: %+.2f%%" % diff_pct)
