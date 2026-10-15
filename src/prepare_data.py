"""Merge NOAA daily rainfall and USGS daily discharge into data/leaf_river.csv.

    python prepare_data.py --rain data/raw_noaa_rain.csv --flow data/raw_usgs_discharge.csv

Inputs
  rain: NOAA CDO "Custom GHCN-Daily CSV" (standard units -> PRCP in inches)
  flow: USGS daily mean discharge (parameter 00060, ft^3/s)
Output columns: date, rainfall (inch), runoff (ft^3/s)
"""
import argparse
import pandas as pd


def load_rain(path, start, end):
    r = pd.read_csv(path)
    r["DATE"] = pd.to_datetime(r["DATE"])
    if r["STATION"].nunique() > 1:  # keep the most complete station
        best = r.groupby("STATION")["PRCP"].apply(lambda s: s.notna().sum()).idxmax()
        print("multiple stations; using", best)
        r = r[r["STATION"] == best]
    name = r["NAME"].iloc[0]
    print(f"rain station: {r['STATION'].iloc[0]} ({name})")
    r = r.drop_duplicates("DATE").set_index("DATE")
    r = r.reindex(pd.date_range(start, end))
    prcp = r["PRCP"].copy()

    # Multi-day totals: when PRCP is empty but MDPR/DAPR are given, the total fell
    # over the DAPR days ending on that date; spread it evenly across them.
    spread = 0
    for date, row in r[r["PRCP"].isna() & r["MDPR"].notna() & r["DAPR"].notna()].iterrows():
        n = int(row["DAPR"])
        days = pd.date_range(end=date, periods=n)
        if prcp.reindex(days).isna().all():
            prcp.loc[days] = row["MDPR"] / n
            spread += 1
    print(f"multi-day totals spread over their days: {spread}")
    return prcp.rename("rainfall")


def load_flow(path, start, end):
    q = pd.read_csv(path)
    q = q[(q["parameter_code"].astype(str).str.lstrip("0") == "60") &
          (q["statistic_id"].astype(str).str.lstrip("0") == "3")]
    q["time"] = pd.to_datetime(q["time"])
    q = q.drop_duplicates("time").set_index("time")["value"]
    return q.reindex(pd.date_range(start, end)).rename("runoff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rain", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--out", default="data/leaf_river.csv")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2012-12-31")
    a = ap.parse_args()

    df = pd.concat([load_rain(a.rain, a.start, a.end),
                    load_flow(a.flow, a.start, a.end)], axis=1)
    df.index.name = "date"
    n = len(df)
    print(f"days: {n} | missing rainfall: {df.rainfall.isna().sum()} "
          f"({100 * df.rainfall.isna().mean():.1f}%) | missing runoff: {df.runoff.isna().sum()}")
    print(f"max rainfall {df.rainfall.max():.2f} in | max runoff {df.runoff.max():.0f} ft^3/s")
    df.reset_index().to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
