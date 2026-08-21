from __future__ import annotations

import numpy as np
import pandas as pd


def derive_abs_rel_outcomes(df: pd.DataFrame, post_col: str, follow_col: str,
                            enlarged_col: str | None = None) -> pd.DataFrame:
    """Derive the preregistered absolute and relative RROC outcomes."""
    out = df.copy()
    post = pd.to_numeric(out[post_col], errors="coerce")
    follow = pd.to_numeric(out[follow_col], errors="coerce")
    valid = post.notna() & follow.notna()
    y_abs = pd.Series(np.nan, index=out.index, dtype=float)
    y_rel = pd.Series(np.nan, index=out.index, dtype=float)
    y_abs.loc[follow.notna()] = (follow.loc[follow.notna()] >= 2).astype(float)
    relative = follow > post
    if enlarged_col is not None:
        enlarged = out[enlarged_col].astype("boolean").fillna(False)
        relative = relative | ((follow == post) & enlarged)
    y_rel.loc[valid] = relative.loc[valid].astype(float)
    out["y_abs"] = y_abs
    out["y_rel"] = y_rel
    return out

