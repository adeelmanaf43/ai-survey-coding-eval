# Reads column NAMES only, no answers, so the output is safe to share
import pyreadstat

STRINGS = "data/raw/BES2024_W30Strings_v30.1.sav"
PANEL = "data/raw/BES2024_W31_Panel_v31.05.sav"


for name, path in [("strings", STRINGS), ("panel", PANEL)]:
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    w25 = [c for c in meta.column_names if "mii" in c.lower()
           and c.endswith("W25")]
    print(f"\n{name}: {meta.number_rows} rows, {len(meta.column_names)} columns")
    print(" has an 'id' column:", "id" in meta.column_names)
    print(" wave 25 MII columns:", w25)
