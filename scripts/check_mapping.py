# Show how BES groups the 50 categories into 12 in wave 25 (category names and counts only).
import pyreadstat
df, _ = pyreadstat.read_sav("data/raw/BES2024_W31_Panel_v31.05.sav",
                            usecols=["mii_catW25", 'small_mii_catW25'],
                            apply_value_formats=True)
df = df.dropna(subset=['mii_catW25']).astype(object).fillna("(none)")
print(df.value_counts().sort_index().to_string())
