#!/usr/bin/env python3
"""
quantumize_ca5.py

Quantum-inspired feature engineering ("quantumization") for CA5-like datasets.

This script:
- Loads a CSV with either:
    (A) count-vector columns like QV_1..QV_39 (+ a date column), OR
    (B) five pick columns like m_1..m_5 (+ a date column),
- Builds "quantum-inspired" features by mapping each row to a complex state vector,
  applying unitary transforms (DFT / "QFT-like") and a graph-based quantum walk,
- Exports features to CSV/Parquet, and (optionally) creates vector embeddings and a FAISS index.

The intent is not to claim physical quantum behavior, but to impose structured, interference-capable
representations that can be used by conventional ML pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm


EPS: float = 1e-12


@dataclass(frozen=True)
class QuantumizeConfig:
    """Configuration for quantum-inspired feature generation.

    Attributes:
        taus: Continuous-time quantum-walk step sizes (in arbitrary units).
        top_k_edges: For graph construction, keep only top-k strongest edges per node.
        phase_a: Phase coefficient tied to part index.
        phase_b: Phase coefficient tied to time index.
        phase_c: Phase coefficient tied to part*time interaction.
        embedding_dim: If provided, produce a dense embedding of this dimension using a
            fixed random projection (useful for vector databases).
        seed: RNG seed for any randomized steps (e.g., embeddings).
        output_format: "csv" or "parquet".
        include_full_distributions: If True, include p_i / q_fft_i / q_tau_i columns for all i.
            If False, include only summary scalars + small slices (still deterministic).
    """

    taus: Tuple[float, ...] = (0.15, 0.50, 1.25)
    top_k_edges: int = 12
    phase_a: float = 0.70
    phase_b: float = 1.30
    phase_c: float = 0.11
    embedding_dim: Optional[int] = 128
    seed: int = 7
    output_format: str = "parquet"
    include_full_distributions: bool = True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv list for programmatic use.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Quantum-inspired feature engineering for CA5-like datasets."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV (e.g., CA5_quantum.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quantumized_out",
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Output table format.",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="0.15,0.50,1.25",
        help="Comma-separated tau values for quantum walk (e.g., '0.1,0.3,1.0').",
    )
    parser.add_argument(
        "--top-k-edges",
        type=int,
        default=12,
        help="Top-k edges per node to keep in the adjacency graph.",
    )
    parser.add_argument(
        "--phase-a",
        type=float,
        default=0.70,
        help="Phase coefficient for part index.",
    )
    parser.add_argument(
        "--phase-b",
        type=float,
        default=1.30,
        help="Phase coefficient for time index.",
    )
    parser.add_argument(
        "--phase-c",
        type=float,
        default=0.11,
        help="Phase coefficient for part*time interaction.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="If >0, generate a dense embedding via random projection.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding generation (even if --embedding-dim is set).",
    )
    parser.add_argument(
        "--build-faiss",
        action="store_true",
        help="Build a FAISS cosine-similarity index (requires faiss-cpu installed).",
    )
    parser.add_argument(
        "--no-full-distributions",
        action="store_true",
        help="Only write summary features (smaller output).",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="",
        help="Optional explicit date column name (otherwise inferred).",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="",
        help="Optional explicit row id column name (otherwise inferred or created).",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Also write resolved config as JSON in the output directory.",
    )
    return parser.parse_args(argv)


def _infer_date_column(df: pd.DataFrame, user_date_col: str = "") -> str:
    """Infer a date column name.

    Args:
        df: Input dataframe.
        user_date_col: If provided and exists, use it.

    Returns:
        Column name to treat as date.

    Raises:
        ValueError: If no plausible date column is found.
    """
    if user_date_col and user_date_col in df.columns:
        return user_date_col

    candidates = [
        "date",
        "Date",
        "event-ID",
        "event_id",
        "eventID",
        "event",
        "draw_date",
        "timestamp",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: look for any column containing "date"
    for c in df.columns:
        if "date" in str(c).lower():
            return c

    raise ValueError(
        "Could not infer a date column. Provide --date-col explicitly."
    )


def _infer_id_column(df: pd.DataFrame, user_id_col: str = "") -> Optional[str]:
    """Infer a stable row id column, if present.

    Args:
        df: Input dataframe.
        user_id_col: If provided and exists, use it.

    Returns:
        Column name to treat as id, or None.
    """
    if user_id_col and user_id_col in df.columns:
        return user_id_col

    candidates = ["id", "ID", "row_id", "event_id", "eventID", "draw_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_qv_columns(df: pd.DataFrame) -> List[str]:
    """Extract count-vector columns like QV_1..QV_39.

    Args:
        df: Input dataframe.

    Returns:
        Sorted list of QV columns. Empty if not found.
    """
    pattern = re.compile(r"^QV_(\d+)$", re.IGNORECASE)
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        m = pattern.match(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]


def _extract_pick_columns(df: pd.DataFrame) -> List[str]:
    """Extract pick columns like m_1..m_5.

    Args:
        df: Input dataframe.

    Returns:
        Sorted list of pick columns. Empty if not found.
    """
    pattern = re.compile(r"^m_(\d+)$", re.IGNORECASE)
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        m = pattern.match(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]


def _build_count_matrix_from_picks(
    df: pd.DataFrame,
    pick_cols: Sequence[str],
    part_min: int = 1,
    part_max: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Build a count matrix X from pick columns (m_1..m_k).

    Each row contains k integers (part ids). This creates a matrix of shape (n_rows, n_parts)
    where X[t, i] is the count of part i+part_min selected on row t.

    Args:
        df: Input dataframe.
        pick_cols: Pick columns (e.g., m_1..m_5).
        part_min: Minimum part id (inclusive).
        part_max: Maximum part id (inclusive). If None, inferred from data.

    Returns:
        (X, part_ids) where:
            X is int matrix (n_rows, n_parts),
            part_ids is the list of part ids corresponding to columns.

    Raises:
        ValueError: If no pick columns are provided.
    """
    if not pick_cols:
        raise ValueError("No pick columns provided to build count matrix.")

    picks = df.loc[:, list(pick_cols)].to_numpy()
    picks = pd.DataFrame(picks).apply(pd.to_numeric, errors="coerce").to_numpy()

    if part_max is None:
        part_max_val = int(np.nanmax(picks))
    else:
        part_max_val = part_max

    n_parts = part_max_val - part_min + 1
    part_ids = list(range(part_min, part_max_val + 1))

    x = np.zeros((picks.shape[0], n_parts), dtype=np.int32)
    for row_idx in range(picks.shape[0]):
        for v in picks[row_idx, :]:
            if np.isnan(v):
                continue
            pid = int(v)
            if part_min <= pid <= part_max_val:
                x[row_idx, pid - part_min] += 1
    return x, part_ids


def load_input(
    csv_path: Path,
    user_date_col: str = "",
    user_id_col: str = "",
) -> Tuple[pd.DataFrame, str, Optional[str], np.ndarray, List[int]]:
    """Load the CSV and extract the count matrix.

    Supports:
    - QV_1..QV_N count columns, OR
    - m_1..m_k pick columns (one-hot aggregated into counts).

    Args:
        csv_path: Path to input CSV.
        user_date_col: Optional explicit date column.
        user_id_col: Optional explicit id column.

    Returns:
        (df, date_col, id_col, X, part_ids)

    Raises:
        ValueError: If neither QV_ columns nor m_ columns are present.
    """
    df = pd.read_csv(csv_path)
    date_col = _infer_date_column(df, user_date_col=user_date_col)
    id_col = _infer_id_column(df, user_id_col=user_id_col)

    qv_cols = _extract_qv_columns(df)
    if qv_cols:
        x = df.loc[:, qv_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        x = np.asarray(x, dtype=np.int32)
        part_ids = [int(re.findall(r"\d+", c)[0]) for c in qv_cols]
        return df, date_col, id_col, x, part_ids

    pick_cols = _extract_pick_columns(df)
    if pick_cols:
        x, part_ids = _build_count_matrix_from_picks(df, pick_cols=pick_cols)
        return df, date_col, id_col, x, part_ids

    raise ValueError(
        "Input must contain either QV_# columns (count-vector) or m_# columns (picks)."
    )


def _safe_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute row-wise Shannon entropy in nats.

    Args:
        probs: Array (n, d), non-negative, rows sum to 1.

    Returns:
        Entropy per row, shape (n,).
    """
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _gini(probs: np.ndarray) -> np.ndarray:
    """Compute a row-wise Gini coefficient for probability vectors.

    Args:
        probs: Array (n, d), non-negative, rows sum to 1.

    Returns:
        Gini per row, shape (n,).
    """
    p = np.sort(probs, axis=1)
    n = p.shape[1]
    idx = np.arange(1, n + 1, dtype=np.float64)[None, :]
    # Gini for distributions:
    # G = 1 - 2 * sum((n+1-i) * p_i) / n
    # Equivalent vectorized form:
    g = 1.0 - (2.0 * np.sum((n + 1 - idx) * p, axis=1)) / float(n)
    return g


def _row_normalize_counts(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert counts to probabilities row-wise.

    Args:
        x: Count matrix (n, d), non-negative.

    Returns:
        (p, row_sum) where p is float (n, d) and row_sum is float (n,).
    """
    row_sum = x.sum(axis=1).astype(np.float64)
    row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
    p = x.astype(np.float64) / row_sum[:, None]
    return p, row_sum


def _make_phases(
    n_rows: int,
    n_parts: int,
    phase_a: float,
    phase_b: float,
    phase_c: float,
) -> np.ndarray:
    """Create deterministic phases for each (time, part) entry.

    Args:
        n_rows: Number of rows (time steps).
        n_parts: Number of part dimensions.
        phase_a: Coefficient tied to part index.
        phase_b: Coefficient tied to time index.
        phase_c: Coefficient tied to interaction term.

    Returns:
        Phase matrix phi of shape (n_rows, n_parts) in radians.
    """
    part_idx = np.arange(n_parts, dtype=np.float64)[None, :]
    time_idx = np.arange(n_rows, dtype=np.float64)[:, None]

    denom_part = max(1.0, float(n_parts))
    denom_time = max(1.0, float(n_rows))

    raw = (
        phase_a * (part_idx / denom_part)
        + phase_b * (time_idx / denom_time)
        + phase_c * ((part_idx * time_idx) / (denom_part * denom_time))
    )
    phi = 2.0 * math.pi * (raw % 1.0)
    return phi


def _make_state_vector(
    p: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Map probabilities to a complex "state vector" per row.

    Uses amplitude encoding with phases:
        psi_i = sqrt(p_i) * exp(i * phi_i)
    and normalizes each row to unit norm.

    Args:
        p: Probabilities (n, d).
        phi: Phases (n, d).

    Returns:
        psi: Complex array (n, d) with ||psi||_2 = 1 per row.
    """
    amp = np.sqrt(np.clip(p, 0.0, 1.0))
    psi = amp * np.exp(1j * phi)

    norms = np.linalg.norm(psi, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return psi / norms


def _unitary_dft(psi: np.ndarray) -> np.ndarray:
    """Apply a unitary DFT (QFT-like) along the last axis.

    Args:
        psi: Complex states (n, d).

    Returns:
        psi_fft: Complex states (n, d), unitary-normalized.
    """
    d = psi.shape[1]
    return np.fft.fft(psi, axis=1) / math.sqrt(float(d))


def _build_adjacency_from_presence(
    x: np.ndarray,
    top_k_edges: int,
) -> np.ndarray:
    """Build a sparse-ish adjacency matrix from presence co-occurrence.

    Uses binary presence (x>0) and cosine similarity:
        w_ij = <b_i, b_j> / (||b_i|| ||b_j||)

    Then keeps top-k edges per node.

    Args:
        x: Count matrix (n, d).
        top_k_edges: Keep top-k edges per node.

    Returns:
        W: Symmetric adjacency (d, d) with zeros on diagonal.
    """
    b = (x > 0).astype(np.float64)
    cooc = b.T @ b  # (d, d)
    diag = np.sqrt(np.clip(np.diag(cooc), EPS, None))
    denom = diag[:, None] * diag[None, :]
    w = cooc / np.clip(denom, EPS, None)
    np.fill_diagonal(w, 0.0)

    d = w.shape[0]
    k = max(1, min(top_k_edges, d - 1))

    mask = np.zeros_like(w, dtype=bool)
    for i in range(d):
        idx = np.argpartition(w[i], -k)[-k:]
        mask[i, idx] = True

    w_sparse = np.where(mask, w, 0.0)
    w_sym = 0.5 * (w_sparse + w_sparse.T)
    np.fill_diagonal(w_sym, 0.0)
    return w_sym


def _laplacian(w: np.ndarray) -> np.ndarray:
    """Compute the combinatorial graph Laplacian L = D - W.

    Args:
        w: Adjacency matrix (d, d), symmetric.

    Returns:
        Laplacian matrix (d, d).
    """
    d = np.diag(np.sum(w, axis=1))
    return d - w


def _quantum_walk_unitary(l: np.ndarray, tau: float) -> np.ndarray:
    """Compute continuous-time quantum walk unitary U = exp(-i * L * tau).

    Args:
        l: Laplacian (d, d), real symmetric.
        tau: Time parameter.

    Returns:
        U: Complex unitary (d, d).
    """
    return expm((-1j) * l * float(tau))


def _topk_mass(p: np.ndarray, k: int) -> np.ndarray:
    """Compute row-wise mass contained in the top-k probabilities.

    Args:
        p: Probabilities (n, d).
        k: Top-k.

    Returns:
        Mass per row, shape (n,).
    """
    k = max(1, min(k, p.shape[1]))
    part = np.partition(p, -k, axis=1)[:, -k:]
    return np.sum(part, axis=1)


def _spectral_centroid(q: np.ndarray) -> np.ndarray:
    """Compute row-wise spectral centroid for a probability vector.

    Args:
        q: Probabilities (n, d), rows sum to 1.

    Returns:
        Centroid per row in [0, d-1], shape (n,).
    """
    d = q.shape[1]
    idx = np.arange(d, dtype=np.float64)[None, :]
    return np.sum(q * idx, axis=1)


def _l1_coherence_from_prob(p: np.ndarray) -> np.ndarray:
    """Compute an L1-coherence proxy for the pure amplitude-encoded state.

    For a pure state built as psi_i = sqrt(p_i) e^{i phi_i}, the magnitude-only
    L1 coherence (sum of off-diagonal magnitudes) is:
        C_l1 = (sum_i sqrt(p_i))^2 - 1

    Args:
        p: Probabilities (n, d), rows sum to 1.

    Returns:
        Coherence proxy per row, shape (n,).
    """
    s = np.sum(np.sqrt(np.clip(p, 0.0, 1.0)), axis=1)
    return (s * s) - 1.0


def quantumize_features(
    x: np.ndarray,
    part_ids: Sequence[int],
    cfg: QuantumizeConfig,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Generate quantum-inspired features and optional embeddings.

    Args:
        x: Count matrix (n, d).
        part_ids: Part ids for each column (length d).
        cfg: QuantumizeConfig.

    Returns:
        (features_df, embeddings) where embeddings is either None or a float32 matrix
        of shape (n, cfg.embedding_dim) L2-normalized.
    """
    n_rows, n_parts = x.shape
    p, row_sum = _row_normalize_counts(x)
    phi = _make_phases(
        n_rows=n_rows,
        n_parts=n_parts,
        phase_a=cfg.phase_a,
        phase_b=cfg.phase_b,
        phase_c=cfg.phase_c,
    )

    psi = _make_state_vector(p=p, phi=phi)
    psi_fft = _unitary_dft(psi)
    q_fft = np.abs(psi_fft) ** 2

    w = _build_adjacency_from_presence(x=x, top_k_edges=cfg.top_k_edges)
    l = _laplacian(w)

    q_walk_map: Dict[float, np.ndarray] = {}
    for tau in cfg.taus:
        u = _quantum_walk_unitary(l=l, tau=float(tau))  # (d, d)
        psi_tau = psi @ u.T  # (n, d)
        q_walk_map[float(tau)] = (np.abs(psi_tau) ** 2).astype(np.float64)

    # Summary features
    features: Dict[str, np.ndarray] = {}

    features["row_total"] = row_sum
    features["entropy_p"] = _safe_entropy(p)
    features["gini_p"] = _gini(p)
    features["purity_diag"] = np.sum(p * p, axis=1)
    features["coherence_l1_proxy"] = _l1_coherence_from_prob(p)
    features["top3_mass_p"] = _topk_mass(p, k=3)
    features["top5_mass_p"] = _topk_mass(p, k=5)

    features["entropy_fft"] = _safe_entropy(q_fft)
    features["top3_mass_fft"] = _topk_mass(q_fft, k=3)
    features["spectral_centroid_fft"] = _spectral_centroid(q_fft)

    for tau, q_tau in q_walk_map.items():
        tag = f"{tau:.3f}".replace(".", "p")
        features[f"entropy_qwalk_{tag}"] = _safe_entropy(q_tau)
        features[f"top3_mass_qwalk_{tag}"] = _topk_mass(q_tau, k=3)
        features[f"spectral_centroid_qwalk_{tag}"] = _spectral_centroid(q_tau)

    # Assemble feature table
    out = pd.DataFrame({k: v for k, v in features.items()})

    if cfg.include_full_distributions:
        # p_i
        for j, pid in enumerate(part_ids):
            out[f"p_{pid}"] = p[:, j]
        # q_fft_i
        for j in range(n_parts):
            out[f"qft_{part_ids[j]}"] = q_fft[:, j]
        # q_walk_i
        for tau, q_tau in q_walk_map.items():
            tag = f"{tau:.3f}".replace(".", "p")
            for j, pid in enumerate(part_ids):
                out[f"qwalk_{tag}_{pid}"] = q_tau[:, j]
    else:
        # Small, deterministic slices (first 8 bins) for compactness
        sl = min(8, n_parts)
        for j in range(sl):
            out[f"p_bin_{j}"] = p[:, j]
            out[f"qft_bin_{j}"] = q_fft[:, j]
        for tau, q_tau in q_walk_map.items():
            tag = f"{tau:.3f}".replace(".", "p")
            for j in range(sl):
                out[f"qwalk_{tag}_bin_{j}"] = q_tau[:, j]

    # Optional embedding for vector DB / ANN
    embeddings: Optional[np.ndarray] = None
    if cfg.embedding_dim is not None and cfg.embedding_dim > 0:
        # Use a stable feature matrix for embedding:
        # concatenate p + q_fft + first q_walk (smallest tau).
        tau0 = float(sorted(q_walk_map.keys())[0])
        f_mat = np.concatenate(
            [p, q_fft, q_walk_map[tau0]],
            axis=1,
        )
        rng = np.random.default_rng(cfg.seed)
        proj = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(f_mat.shape[1], int(cfg.embedding_dim)),
        ).astype(np.float32)
        proj /= math.sqrt(float(cfg.embedding_dim))

        emb = (f_mat.astype(np.float32) @ proj).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms <= 0.0, 1.0, norms).astype(np.float32)
        embeddings = emb / norms

    return out, embeddings


def _write_table(df: pd.DataFrame, out_path: Path, fmt: str) -> None:
    """Write a dataframe to disk with a safe fallback.

    Args:
        df: DataFrame to write.
        out_path: Output path (extension may be ignored for parquet).
        fmt: "csv" or "parquet".

    Notes:
        If fmt="parquet" but no parquet engine is installed (pyarrow/fastparquet),
        this falls back to CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        return

    try:
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except ImportError:
        df.to_csv(out_path.with_suffix(".csv"), index=False)


def _try_build_faiss_index(
    embeddings: np.ndarray,
    index_path: Path,
) -> None:
    """Build a FAISS index for cosine similarity.

    Uses IndexFlatIP with L2-normalized vectors (cosine = inner product).

    Args:
        embeddings: Float32 (n, d) L2-normalized.
        index_path: Path to save the FAISS index.

    Raises:
        ImportError: If faiss is not installed.
    """
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError(
            "faiss-cpu is required for --build-faiss. "
            "Install with: pip install faiss-cpu"
        ) from exc

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argv list for programmatic use.

    Returns:
        Exit code (0 on success).
    """
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    taus = tuple(float(t.strip()) for t in str(args.taus).split(",") if t.strip())
    include_full = not bool(args.no_full_distributions)

    cfg = QuantumizeConfig(
        taus=taus,
        top_k_edges=int(args.top_k_edges),
        phase_a=float(args.phase_a),
        phase_b=float(args.phase_b),
        phase_c=float(args.phase_c),
        embedding_dim=None if args.no_embeddings else int(args.embedding_dim),
        seed=7,
        output_format=str(args.output_format),
        include_full_distributions=include_full,
    )

    df, date_col, id_col, x, part_ids = load_input(
        csv_path=input_path,
        user_date_col=str(args.date_col),
        user_id_col=str(args.id_col),
    )

    # Parse and sort by date for determinism
    dates = pd.to_datetime(df[date_col], errors="coerce")
    order = np.argsort(dates.to_numpy(dtype="datetime64[ns]"))
    df = df.iloc[order].reset_index(drop=True)
    x_sorted = x[order]

    feat_df, embeddings = quantumize_features(
        x=x_sorted,
        part_ids=part_ids,
        cfg=cfg,
    )

    # Attach identifiers
    out_base = pd.DataFrame()
    if id_col is not None and id_col in df.columns:
        out_base["row_id"] = df[id_col].astype(str)
    else:
        out_base["row_id"] = np.arange(df.shape[0], dtype=np.int64).astype(str)

    out_base["date"] = df[date_col].astype(str)
    out_all = pd.concat([out_base, feat_df], axis=1)

    _write_table(out_all, out_dir / "quantumized_features", fmt=cfg.output_format)

    if embeddings is not None:
        np.save(out_dir / "quantumized_embeddings.npy", embeddings)
        # Also write a minimal metadata file for vector lookup
        meta = out_base.copy()
        meta.to_csv(out_dir / "quantumized_embedding_metadata.csv", index=False)

        if bool(args.build_faiss):
            _try_build_faiss_index(
                embeddings=embeddings,
                index_path=out_dir / "quantumized_faiss.index",
            )

    if bool(args.save_config):
        cfg_json = {
            "taus": list(cfg.taus),
            "top_k_edges": cfg.top_k_edges,
            "phase_a": cfg.phase_a,
            "phase_b": cfg.phase_b,
            "phase_c": cfg.phase_c,
            "embedding_dim": cfg.embedding_dim,
            "seed": cfg.seed,
            "output_format": cfg.output_format,
            "include_full_distributions": cfg.include_full_distributions,
            "date_col": date_col,
            "id_col": id_col,
            "n_rows": int(df.shape[0]),
            "n_parts": int(x_sorted.shape[1]),
        }
        with (out_dir / "quantumize_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg_json, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# requirements.txt (pin as desired)
# numpy==1.26.4
# pandas==2.2.2
# scipy==1.14.1
# pyarrow==16.1.0
# faiss-cpu==1.8.0.post1
