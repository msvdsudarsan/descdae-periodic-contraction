"""Record the software environment and SHA-256 checksums of all generated results."""
import hashlib, json, os, platform, sys
import numpy, scipy, matplotlib
root = os.path.join(os.path.dirname(__file__), "..")
env = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
           matplotlib=matplotlib.__version__, platform=platform.platform())
json.dump(env, open(os.path.join(root, "results", "environment.json"), "w"), indent=1)
lines = []
for d in ("results", "results/tables"):
    for f in sorted(os.listdir(os.path.join(root, d))):
        p = os.path.join(root, d, f)
        if os.path.isfile(p) and not f.endswith("log.txt") and f not in ("MANIFEST.sha256",):
            lines.append(hashlib.sha256(open(p, "rb").read()).hexdigest() + "  " + os.path.join(d, f))
open(os.path.join(root, "results", "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")
print("manifest:", len(lines), "files;", env)
