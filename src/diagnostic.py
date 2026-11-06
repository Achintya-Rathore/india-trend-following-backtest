import numpy as np, pandas as pd
from pathlib import Path

RAW = Path(r"C:\Users\Rathores.000\Desktop\Python Project\data\raw")
y = pd.read_csv(RAW / "usdinr_yahoo.csv", index_col=0, parse_dates=True)["Close"]
f = pd.read_csv(RAW / "usdinr_fred.csv", index_col=0, parse_dates=True)["DEXINUS"]
f = pd.to_numeric(f, errors="coerce").dropna()

y1 = y.shift(-1)   # Yahoo ek din late hai, to agli row ki value is row pe laao
both = pd.concat([f.rename("f"), y1.rename("y1")], axis=1, sort=True).dropna()

r = np.log(both).diff().dropna()
print("daily corr after shift:", round(r["f"].corr(r["y1"]), 4))

last = both.groupby(both.index.to_period("M")).last()
m = np.log(last).diff().dropna()
print("month-end corr after shift:", round(m["f"].corr(m["y1"]), 4))

print("\nyear-wise daily corr after shift:")
print(r.groupby(r.index.year).apply(lambda d: round(d["f"].corr(d["y1"]), 2)))