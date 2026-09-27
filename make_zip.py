import zipfile
import os
import time
from pathlib import Path

t0 = time.time()
root_dir = Path(__file__).resolve().parent
zpath = root_dir / "PraxisCode_X_submission.zip"

files = [
    ("business_entity_resolution/output/matching_results.tsv", "output/matching_results.tsv"),
    ("business_entity_resolution/output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ("DATA SET/student_resource/Documentation_template.md", "Documentation_template.md"),
    ("README.md", "code/business_entity_resolution/README.md"),
    ("business_entity_resolution/src/requirements.txt", "code/business_entity_resolution/requirements.txt"),
]

print(f"Creating submission zip archive at: {zpath}")
with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # 1. Add key files
    for src, arc in files:
        full_src = root_dir / src
        if full_src.exists():
            z.write(full_src, arc)
            print(f"  + added {arc}")
        else:
            print(f"  ⚠️ missing: {full_src}")
    
    # 2. Add source code tree
    src_dir = root_dir / "business_entity_resolution" / "src"
    for dp, dn, fn in os.walk(src_dir):
        if "__pycache__" in dp or ".git" in dp:
            continue
        for f in fn:
            if f.endswith(".pyc"):
                continue
            full_path = Path(dp) / f
            rel_path = full_path.relative_to(src_dir)
            arc_name = Path("code/business_entity_resolution/src") / rel_path
            z.write(full_path, str(arc_name))

size_mb = zpath.stat().st_size / (1024 * 1024)
print(f"✅ ZIP CREATED SUCCESSFULLY: {zpath.name} ({size_mb:.2f} MB in {time.time()-t0:.2f}s)")
