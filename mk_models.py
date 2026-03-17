import numpy as np
import pandas as pd


def mk_model(
    df: pd.DataFrame,
    word_col: str = "word",
    count_col: str = "count",
    model_type: str = "ordering",
    n: int | None = None,
    set_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a model from word-count data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a column of words and a column of non-negative integer counts.
    word_col : str
        Name of the column containing words.
    count_col : str
        Name of the column containing counts.
    model_type : str
        Algorithm to use. Currently supported: "ordering", "set".
    n : int or None
        Number of items to return for the ordering model. If None, returns the
        full ordering. Ignored by the set model.
    set_size : int or None
        Number of items to return for the set model. Required when
        model_type="set".
    rng : numpy.random.Generator or None
        Random number generator for reproducibility. If None, a new default
        Generator is created.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the words and their original counts, ordered by the
        selected model.
    """
    match model_type:
        case "ordering":
            return _ordering_model(df, word_col, count_col, n, rng)
        case "set":
            if set_size is None:
                raise ValueError("set_size is required when model_type='set'")
            return _set_model(df, word_col, count_col, set_size, rng)
        case _:
            raise ValueError(f"Unknown model_type: {model_type!r}")


def _set_model(
    df: pd.DataFrame,
    word_col: str,
    count_col: str,
    set_size: int,
    rng: np.random.Generator | None,
) -> pd.DataFrame:
    """Weighted sampling without replacement of exactly set_size items.

    Uses the same Gumbel-max trick as the ordering model. The output is sorted
    by descending key for consistency, but the ordering is not semantically
    meaningful for the set model.
    """
    return _ordering_model(df, word_col, count_col, n=set_size, rng=rng)


def _ordering_model(
    df: pd.DataFrame,
    word_col: str,
    count_col: str,
    n: int | None,
    rng: np.random.Generator | None,
) -> pd.DataFrame:
    """Weighted sampling without replacement via the Gumbel-max trick.

    For each item with count w_i, we compute a key:
        key_i = log(w_i) + Gumbel(0, 1)
    where Gumbel(0, 1) = -log(-log(U)), U ~ Uniform(0, 1).

    Sorting items by descending key is equivalent to sequentially sampling
    without replacement with probabilities proportional to counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = df[count_col].values
    if not np.issubdtype(counts.dtype, np.integer):
        counts = counts.astype(np.int64)

    # Filter to positive counts only
    positive_mask = counts > 0
    if not positive_mask.all():
        df = df.loc[positive_mask].copy()
        counts = counts[positive_mask]

    num_items = len(counts)
    if num_items == 0:
        return df[[word_col, count_col]].iloc[:0]

    if n is not None:
        n = min(n, num_items)
    else:
        n = num_items

    # Gumbel-max trick: key_i = log(count_i) - log(-log(U_i))
    # Using log-space throughout to protect against underflow with small counts.
    log_counts = np.log(counts.astype(np.float64))
    uniforms = rng.random(num_items)
    gumbel_noise = -np.log(-np.log(uniforms))
    keys = log_counts + gumbel_noise

    # Partial sort: find the top-n indices in O(num_items) via argpartition,
    # then fully sort only those n items in O(n log n).
    if n < num_items:
        top_n_unsorted = np.argpartition(keys, -n)[-n:]
        top_n_order = np.argsort(keys[top_n_unsorted])[::-1]
        ordered_indices = top_n_unsorted[top_n_order]
    else:
        ordered_indices = np.argsort(keys)[::-1]

    return df.iloc[ordered_indices][[word_col, count_col]].reset_index(drop=True)
