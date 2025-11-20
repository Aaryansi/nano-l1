import pandas as pd

def load_ticks_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    # ensure correct ordering
    df = df.sort_values("ts").reset_index(drop=True)
    return df
