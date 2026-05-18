import io
import html
import os
import platform
import subprocess
import sys
import tempfile

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from joblib import dump, load
from scipy.signal import savgol_filter
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_tensorflow():
    try:
        import tensorflow as tf_mod
        return tf_mod
    except ModuleNotFoundError:
        package = "tensorflow-cpu" if platform.system().lower() == "linux" else "tensorflow"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        import tensorflow as tf_mod
        return tf_mod


tf = _import_tensorflow()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "exo_cnn_model.keras")
CLASS_LABELS = ["No Exoplanet Detected", "Exoplanet Detected"]
SIGNAL_LENGTH = 3197
ROBUST_SCALER_PATH = os.path.join(APP_DIR, "robust_scaler.joblib")
DEBUG_PREFIX = "[exo-debug]"
EXO_TEST_FILENAME = "exoTest.csv"
DEFAULT_PREDICTION_THRESHOLD = 0.2
EXO_THRESHOLD_CALIBRATION_CSV = os.getenv("EXO_THRESHOLD_CALIBRATION_CSV")


def _maybe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


TRAIN_INPUT_MEAN = _maybe_float(os.getenv("TRAIN_INPUT_MEAN"))
TRAIN_INPUT_STD = _maybe_float(os.getenv("TRAIN_INPUT_STD"))
PREDICTION_THRESHOLD = DEFAULT_PREDICTION_THRESHOLD
SCORE_POLARITY = os.getenv("EXO_SCORE_POLARITY", "auto").strip().lower()
SCORE_DIRECTION = "inverted" if SCORE_POLARITY == "inverted" else "normal"
SAMPLE_CALIBRATION_STATUS_MESSAGE = ""
SAMPLE_PIPELINE_STATUS_MESSAGE = ""


def _load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, False, f"Model file not found: {MODEL_PATH}"
    try:
        loaded = tf.keras.models.load_model(MODEL_PATH)
        return loaded, True, "Model loaded successfully."
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, False, f"Model load error: {exc}"


model, MODEL_LOADED, MODEL_STATUS_MESSAGE = _load_model()


def _debug_log(message):
    print(f"{DEBUG_PREFIX} {message}", flush=True)


def _array_stats(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return "empty"

    return (
        f"shape={array.shape} dtype={array.dtype} "
        f"min={float(np.min(array)):.6f} max={float(np.max(array)):.6f} "
        f"mean={float(np.mean(array)):.6f} std={float(np.std(array)):.6f}"
    )


def _probability_to_logit(probability):
    clipped = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return float(np.log(clipped / (1.0 - clipped)))


def _binary_prediction_label(exo_score):
    return CLASS_LABELS[1] if exo_score >= PREDICTION_THRESHOLD else CLASS_LABELS[0]


def _raw_score_to_exo_score(raw_score):
    raw_score = float(np.clip(raw_score, 0.0, 1.0))
    return 1.0 - raw_score if SCORE_DIRECTION == "inverted" else raw_score


def _sort_flux_columns(columns):
    return sorted(
        columns,
        key=lambda c: int("".join(ch for ch in str(c) if ch.isdigit()) or 0),
    )


def _log_pipeline_summary():
    _debug_log(f"class mapping: index 0 -> {CLASS_LABELS[0]!r}, index 1 -> {CLASS_LABELS[1]!r}")
    _debug_log(f"binary threshold: exo_score >= {PREDICTION_THRESHOLD:.3f} -> {CLASS_LABELS[1]!r}")
    _debug_log(f"score polarity setting: EXO_SCORE_POLARITY={SCORE_POLARITY!r}, active direction={SCORE_DIRECTION!r}")
    _debug_log(f"model input_shape={model.input_shape if model is not None else None}, output_shape={model.output_shape if model is not None else None}")
    _debug_log(
        "inference pipeline: raw flux -> np.asarray(dtype=float64).reshape(-1) -> "
        "abs(fft) -> savgol_filter(window=21, polyorder=4) -> sklearn.normalize(L2) -> "
        "robust_scaler.transform(rowwise) -> reshape(1, 3197, 1) -> astype(float32)"
    )
    _debug_log(
        "training pipeline: no separate training script/notebook or preprocessing layer was found in the repo; "
        "the exported Keras model contains only the classifier stack, so the UI must mirror the training transform externally"
    )
def model_status_html():
    state = "online" if MODEL_LOADED else "offline"
    color = "#1f8f4b" if MODEL_LOADED else "#b24a2d"
    return (
        "<div class='status-pill'>"
        f"<span class='status-dot' style='background:{color}'></span>"
        f"<strong>Model Status:</strong> {state}"
        "</div>"
        f"<p class='status-note'>{MODEL_STATUS_MESSAGE}</p>"
    )


def _signal_placeholder_html(message="Awaiting signal input"):
    return f"""
    <div class="signal-placeholder">
        <div class="signal-placeholder__title">{html.escape(message)}</div>
        <div class="signal-placeholder__body">Upload a CSV or load one of the sample curves to render the raw light curve here.</div>
    </div>
    """


def _prediction_card_html(predicted_label=None, exo_score=None, detail_text="", confidences=None):
    if predicted_label is None or exo_score is None:
        empty_detail = html.escape(detail_text) if detail_text else "Upload a CSV or load a sample curve, then press Analyze."
        return """
        <div class="prediction-card prediction-card--empty">
            <div class="prediction-card__eyebrow">Prediction</div>
            <div class="prediction-card__empty-title">Awaiting analysis</div>
            <div class="prediction-card__empty-body">%s</div>
        </div>
        """ % empty_detail

    is_exoplanet = predicted_label == CLASS_LABELS[1]
    score_pct = max(0.0, min(100.0, float(exo_score) * 100.0))
    theme_class = "prediction-card--exo" if is_exoplanet else "prediction-card--non"
    badge_text = html.escape(predicted_label)
    detail_html = html.escape(detail_text) if detail_text else "Analysis complete."
    confidence_bits = ""
    if confidences:
        ordered = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        confidence_bits = f"<div class='prediction-card__meta'>Top confidence: {html.escape(top_label)} {float(top_score) * 100.0:.1f}%</div>"

    return f"""
    <div class="prediction-card {theme_class}">
        <div class="prediction-card__eyebrow">Prediction</div>
        <div class="prediction-card__topline">
            <div class="prediction-badge">{badge_text}</div>
            <div class="prediction-score">{score_pct:.1f}%</div>
        </div>
        <div class="prediction-card__bar" aria-hidden="true">
            <div class="prediction-card__bar-fill" style="width: {score_pct:.1f}%"></div>
        </div>
        <div class="prediction-card__caption">Exoplanet probability</div>
        <div class="prediction-card__detail">{detail_html}</div>
        {confidence_bits}
    </div>
    """


def _visualization_updates(image, placeholder_message="Awaiting signal input"):
    if image is None:
        return gr.update(value=_signal_placeholder_html(placeholder_message), visible=True), gr.update(value=None, visible=False)

    return gr.update(value=_signal_placeholder_html(placeholder_message), visible=False), gr.update(value=image, visible=True)


def preprocess_flux(flux):
    x = np.asarray(flux, dtype=np.float64).reshape(-1)
    if x.size != SIGNAL_LENGTH:
        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values.")

    print("TRAIN mean/std:", TRAIN_INPUT_MEAN, TRAIN_INPUT_STD, flush=True)
    print("INPUT mean/std:", float(np.mean(x)), float(np.std(x)), flush=True)

    x = np.abs(np.fft.fft(x))
    x = savgol_filter(x, 21, 4)
    x = normalize([x])[0]
    x = robust_scaler.transform([x])[0]

    return x.reshape(1, SIGNAL_LENGTH, 1).astype(np.float32)


class _RowwiseRobustScaler:
    def transform(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        transformed_rows = []
        for row in array:
            median = np.median(row)
            q75, q25 = np.percentile(row, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-10:
                row = (row - median) / iqr
            transformed_rows.append(row)

        return np.asarray(transformed_rows, dtype=np.float64)


def _load_robust_scaler():
    return _RowwiseRobustScaler()


robust_scaler = _load_robust_scaler()


def _coerce_binary_label(value):
    if value is None or pd.isna(value):
        raise ValueError("Missing LABEL value in calibration CSV.")

    try:
        numeric = int(float(value))
        if numeric in (0, 1):
            return numeric
    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "exoplanet", "exo", "detected"}:
        return 1
    if text in {"0", "false", "no", "non-exoplanet", "no exoplanet", "not detected"}:
        return 0

    raise ValueError(f"Unsupported binary label value: {value!r}")


def _load_labeled_calibration_set(source):
    if hasattr(source, "name") and isinstance(source.name, str):
        source = source.name

    if isinstance(source, os.PathLike):
        source = os.fspath(source)

    if not (isinstance(source, str) and os.path.isfile(source)):
        raise FileNotFoundError(f"Calibration CSV not found: {source}")

    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("Calibration CSV is empty.")

    if "LABEL" not in df.columns:
        raise ValueError("Calibration CSV must include a LABEL column.")

    drop_cols = [c for c in df.columns if str(c).strip().upper().startswith("UNNAMED")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    flux_cols = [c for c in df.columns if "FLUX" in str(c).upper()]
    if flux_cols:
        flux_cols = _sort_flux_columns(flux_cols)
    else:
        raise ValueError("Calibration CSV must contain FLUX columns.")

    if len(flux_cols) != SIGNAL_LENGTH:
        raise ValueError(
            f"Calibration CSV must contain exactly {SIGNAL_LENGTH} FLUX columns; found {len(flux_cols)}."
        )

    labels = df["LABEL"].map(_coerce_binary_label).to_numpy(dtype=np.int64)
    flux_matrix = df[flux_cols].to_numpy(dtype=np.float32)
    if flux_matrix.shape[0] != labels.shape[0]:
        raise ValueError("Calibration CSV row count does not match label count.")

    return flux_matrix, labels


def _predict_positive_class_probability(raw_flux):
    sample = preprocess_flux(raw_flux)
    preds = model.predict(sample, verbose=0)
    print("PRED:", preds, flush=True)
    if preds.shape[-1] == 1:
        return float(preds[0][0])

    return float(preds[0][1])


def _calibrate_prediction_threshold_from_csv(source):
    flux_matrix, labels = _load_labeled_calibration_set(source)
    scores = np.asarray([_predict_positive_class_probability(row) for row in flux_matrix], dtype=np.float64)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Calibration set did not produce threshold candidates.")

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]
    f1_scores = (2.0 * precision_for_thresholds * recall_for_thresholds) / np.maximum(
        precision_for_thresholds + recall_for_thresholds,
        1e-12,
    )
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    _debug_log(
        f"calibration source={source!r} samples={len(labels)} roc_auc={roc_auc:.6f} "
        f"best_f1={best_f1:.6f} threshold={best_threshold:.6f}"
    )
    return best_threshold


def _resolve_prediction_threshold():
    explicit_threshold = _maybe_float(os.getenv("EXO_PREDICTION_THRESHOLD"))
    if explicit_threshold is not None:
        _debug_log(f"using explicit EXO_PREDICTION_THRESHOLD={explicit_threshold:.6f}")
        return float(explicit_threshold)

    calibration_candidates = []
    if EXO_THRESHOLD_CALIBRATION_CSV:
        calibration_candidates.append(EXO_THRESHOLD_CALIBRATION_CSV)

    sample_dir = APP_DIR
    calibration_candidates.extend(
        [
            os.path.join(sample_dir, "X_test.csv"),
            os.path.join(sample_dir, "x_test.csv"),
            os.path.join(sample_dir, "test.csv"),
        ]
    )

    for candidate in calibration_candidates:
        if candidate and os.path.isfile(candidate):
            try:
                return _calibrate_prediction_threshold_from_csv(candidate)
            except Exception as exc:
                _debug_log(f"threshold calibration skipped for {candidate!r}: {exc}")

    _debug_log(f"no labeled calibration CSV found; using default threshold={DEFAULT_PREDICTION_THRESHOLD:.3f}")
    return DEFAULT_PREDICTION_THRESHOLD


PREDICTION_THRESHOLD = _resolve_prediction_threshold()


def sanity_check(model_obj, df, threshold):
    if "LABEL" not in df.columns:
        raise ValueError("Sanity check CSV must contain a LABEL column.")

    exo_rows = df[df["LABEL"] == 2]
    non_rows = df[df["LABEL"] == 1]
    if exo_rows.empty:
        raise ValueError("Sanity check CSV does not contain a LABEL == 2 exoplanet row.")
    if non_rows.empty:
        raise ValueError("Sanity check CSV does not contain a LABEL == 1 non-exoplanet row.")

    def _row_to_sample(row):
        values = row.drop(labels=["LABEL"], errors="ignore")
        values = pd.to_numeric(values, errors="coerce")
        if values.isna().any():
            bad_columns = [str(col) for col in values.index[values.isna()].tolist()]
            raise ValueError(f"Sanity check row contains non-numeric values in columns: {', '.join(bad_columns)}")

        sample = values.to_numpy(dtype=np.float32)
        print("FINAL FEATURE LENGTH:", len(sample), flush=True)
        assert len(sample) == SIGNAL_LENGTH, "Feature length mismatch!"
        sample = sample.reshape(1, SIGNAL_LENGTH, 1)
        print("INPUT SHAPE:", sample.shape, flush=True)
        print("MIN/MAX:", sample.min(), sample.max(), flush=True)
        return sample

    exo = _row_to_sample(exo_rows.iloc[0])
    exo_preds = model_obj.predict(exo, verbose=0)
    print("PRED:", exo_preds, flush=True)
    exo_prob = float(exo_preds[0][0]) if exo_preds.shape[-1] == 1 else float(exo_preds[0][1])

    non = _row_to_sample(non_rows.iloc[0])
    non_preds = model_obj.predict(non, verbose=0)
    print("PRED:", non_preds, flush=True)
    non_prob = float(non_preds[0][0]) if non_preds.shape[-1] == 1 else float(non_preds[0][1])

    print("EXO PROB:", exo_prob, "=>", exo_prob > threshold, flush=True)
    print("NON PROB:", non_prob, "=>", non_prob > threshold, flush=True)

    return exo_prob, non_prob


def _run_sanity_check_from_csv():
    sanity_csv = os.getenv("EXO_SANITY_CHECK_CSV")
    if not sanity_csv:
        fallback_csv = os.path.join(APP_DIR, "exoTest.csv")
        if os.path.isfile(fallback_csv):
            sanity_csv = fallback_csv
    if not sanity_csv:
        return

    if hasattr(sanity_csv, "name") and isinstance(sanity_csv.name, str):
        sanity_csv = sanity_csv.name

    if isinstance(sanity_csv, os.PathLike):
        sanity_csv = os.fspath(sanity_csv)

    if not (isinstance(sanity_csv, str) and os.path.isfile(sanity_csv)):
        _debug_log(f"sanity check skipped; CSV not found: {sanity_csv!r}")
        return

    try:
        df = pd.read_csv(sanity_csv)
        sanity_check(model, df, PREDICTION_THRESHOLD)
    except Exception as exc:
        _debug_log(f"sanity check failed for {sanity_csv!r}: {exc}")


_log_pipeline_summary()


_run_sanity_check_from_csv()


def _parse_exact_flux_values(values):
    numeric = pd.to_numeric(pd.Series(values).astype(str).str.strip(), errors="coerce")
    if numeric.isna().any():
        raise ValueError("Input must contain only numeric flux values.")

    arr = numeric.to_numpy(dtype=np.float64).flatten()
    if len(arr) != SIGNAL_LENGTH:
        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values.")

    return arr.astype(np.float32)


_clean_flux_array = _parse_exact_flux_values


def _build_sample_sequence(numeric_values):
    if not numeric_values:
        raise ValueError("No numeric flux values found.")

    if len(numeric_values) <= SIGNAL_LENGTH + 1:
        label_val = numeric_values[0] if len(numeric_values) > 1 else None
        flux_values = numeric_values[1:] if len(numeric_values) > 1 else numeric_values
        return [
            {
                "flux": _clean_flux_array(flux_values),
                "row_index": 0,
                "label": None if label_val is None else float(label_val),
            }
        ]

    chunk_size = SIGNAL_LENGTH + 1
    if len(numeric_values) % chunk_size == 0:
        samples = []
        for chunk_index in range(0, len(numeric_values), chunk_size):
            chunk = numeric_values[chunk_index : chunk_index + chunk_size]
            samples.append(
                {
                    "flux": _clean_flux_array(chunk[1:]),
                    "row_index": chunk_index // chunk_size,
                    "label": float(chunk[0]),
                }
            )
        return samples

    return [
        {
            "flux": _clean_flux_array(numeric_values[1:]),
            "row_index": 0,
            "label": float(numeric_values[0]),
        }
    ]


def _source_to_text(source):
    if source is None:
        return None

    if hasattr(source, "name") and isinstance(source.name, str):
        source = source.name

    if isinstance(source, os.PathLike):
        source = os.fspath(source)

    if isinstance(source, str) and os.path.isfile(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()

    return str(source)


def _is_exo_test_source(source):
    if hasattr(source, "name") and isinstance(source.name, str):
        source = source.name

    if isinstance(source, os.PathLike):
        source = os.fspath(source)

    return isinstance(source, str) and os.path.basename(source) == EXO_TEST_FILENAME


def _load_exo_test_raw_row(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        if hasattr(source, "name") and isinstance(source.name, str):
            source = source.name

        if isinstance(source, os.PathLike):
            source = os.fspath(source)

        if isinstance(source, str) and os.path.isfile(source):
            df = pd.read_csv(source)
        else:
            source_text = _source_to_text(source)
            if source_text is None:
                raise ValueError("exoTest.csv input is empty.")
            df = pd.read_csv(io.StringIO(source_text))

    if df.empty:
        raise ValueError("exoTest.csv is empty.")

    if "LABEL" in df.columns:
        df = df.drop(columns=["LABEL"])

    flux_cols = [c for c in df.columns if "FLUX" in str(c).upper()]
    if not flux_cols:
        raise ValueError("exoTest.csv does not contain FLUX columns.")

    df = df[flux_cols]
    sample = df.iloc[0].values.astype(np.float32)

    print("FINAL FEATURE LENGTH:", len(sample), flush=True)
    assert len(sample) == SIGNAL_LENGTH, "Feature length mismatch!"

    flux = sample.reshape(-1)
    if flux.size != SIGNAL_LENGTH:
        raise ValueError(f"exoTest.csv row must contain exactly {SIGNAL_LENGTH} flux values after dropping LABEL/index columns.")

    return flux


def build_input_table():
    return pd.DataFrame({"Flux": [np.nan] * SIGNAL_LENGTH})


DEFAULT_INPUT_TABLE = build_input_table()


def flux_to_model_input_table(flux):
    values = np.asarray(flux, dtype=np.float64).reshape(-1)
    if values.size != SIGNAL_LENGTH:
        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values.")

    return pd.DataFrame(
        {
            "Time Step": np.arange(SIGNAL_LENGTH, dtype=np.int64),
            "Flux": values.astype(np.float32),
        }
    )


def table_to_flux(table):
    if table is None:
        raise ValueError("Model input table is empty.")

    df = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    if df.empty:
        raise ValueError("Model input table is empty.")

    if "Flux" not in df.columns:
        raise ValueError("Model input table must include a Flux column.")

    if "Time Step" in df.columns:
        df = df.copy()
        df["Time Step"] = pd.to_numeric(df["Time Step"], errors="coerce")
        if df["Time Step"].isna().any():
            raise ValueError("Model input table contains non-numeric Time Step values.")
        df = df.sort_values("Time Step")

    flux = pd.to_numeric(df["Flux"], errors="coerce")
    if flux.isna().any():
        raise ValueError("Model input table contains non-numeric Flux values.")

    values = flux.to_numpy(dtype=np.float32).reshape(-1)
    if values.size != SIGNAL_LENGTH:
        raise ValueError(f"Model input table must contain exactly {SIGNAL_LENGTH} rows; found {values.size}.")

    _debug_log(f"model input table: shape={df.shape} columns={list(df.columns)!r}")
    _debug_log(f"model input table flux_stats: {_array_stats(values)}")
    return values


def _model_table_records(flux):
    return flux_to_model_input_table(flux).to_dict(orient="records")


def _sample_record_from_flux(
    flux,
    row_index=None,
    label=None,
    title="Raw Light Curve",
    expected_label=None,
    filename=None,
    input_mode="raw",
):
    table = flux_to_model_input_table(flux)
    return {
        "table": table.to_dict(orient="records"),
        "row_index": row_index,
        "label": label,
        "title": title,
        "expected_label": expected_label,
        "filename": filename,
        "input_mode": input_mode,
    }


def _model_input_table_preview(table, edge_rows=8):
    df = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    if df.empty:
        return pd.DataFrame(columns=["Time Step", "Flux"])

    if len(df) <= edge_rows * 2:
        return df

    spacer = pd.DataFrame([{"Time Step": "...", "Flux": "..."}])
    return pd.concat([df.head(edge_rows), spacer, df.tail(edge_rows)], ignore_index=True)


def _parse_table_input(table):
    if table is None:
        raise ValueError("Input table is empty.")

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)

    table = table.dropna(how="all")
    if table.empty:
        raise ValueError("Input table is empty.")

    if "Flux" in table.columns:
        flux = table_to_flux(table)
    elif table.shape[1] == 1:
        flux = _parse_exact_flux_values(table.iloc[:, 0].to_list())
    else:
        flux = _parse_exact_flux_values(table.stack().to_list())

    return [
        _sample_record_from_flux(flux, row_index=0, label=None)
    ]


def parse_flux_csv(source):
    if isinstance(source, pd.DataFrame) or isinstance(source, list):
        return _parse_table_input(source)

    source_text = _source_to_text(source)
    if source_text is None:
        raise ValueError("Input is empty.")

    def _clean_flux_array(values):
        arr = pd.to_numeric(pd.Series(values).astype(str).str.strip(), errors="coerce").to_numpy(dtype=np.float64)
        arr = arr.flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) != SIGNAL_LENGTH:
            raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values; found {len(arr)}.")
        return arr.astype(np.float32)

    def _is_numeric_line(value):
        return pd.notna(pd.to_numeric(str(value).strip(), errors="coerce"))

    def _build_sample_sequence(numeric_values):
        if not numeric_values:
            raise ValueError("No numeric flux values found.")

        if len(numeric_values) == SIGNAL_LENGTH:
            return [_sample_record_from_flux(_clean_flux_array(numeric_values), row_index=None, label=None)]

        if len(numeric_values) == SIGNAL_LENGTH + 1:
            label_val = numeric_values[0]
            return [
                _sample_record_from_flux(
                    _clean_flux_array(numeric_values[1:]),
                    row_index=0,
                    label=None if pd.isna(label_val) else float(label_val),
                )
            ]

        chunk_size = SIGNAL_LENGTH + 1
        if len(numeric_values) % chunk_size == 0:
            samples = []
            for chunk_index in range(0, len(numeric_values), chunk_size):
                chunk = numeric_values[chunk_index : chunk_index + chunk_size]
                samples.append(
                    _sample_record_from_flux(
                        _clean_flux_array(chunk[1:]),
                        row_index=chunk_index // chunk_size,
                        label=float(chunk[0]),
                    )
                )
            return samples

        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values, or LABEL plus {SIGNAL_LENGTH} flux values.")

    headerless_loaders = [
        lambda text: pd.read_csv(io.StringIO(text), header=None),
        lambda text: pd.read_csv(io.StringIO(text), sep="\t", header=None),
        lambda text: pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", header=None),
    ]
    for loader in headerless_loaders:
        try:
            df_headerless = loader(source_text)
        except Exception:
            continue

        if df_headerless is None or df_headerless.empty:
            continue

        numeric = pd.to_numeric(df_headerless.stack(), errors="coerce")
        if numeric.notna().all():
            try:
                return _build_sample_sequence(numeric.to_list())
            except ValueError:
                pass

    # Tolerant load: supports CSV/TSV, mixed whitespace separators, and Kepler FLUX columns.
    df = None
    loaders = [
        lambda text: pd.read_csv(io.StringIO(text)),
        lambda text: pd.read_csv(io.StringIO(text), sep="\t"),
        lambda text: pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python"),
    ]
    for loader in loaders:
        try:
            maybe = loader(source_text)
            if maybe is not None and not maybe.empty:
                df = maybe
                break
        except Exception:
            continue

    if df is not None and not df.empty:
        # Drop unnamed index columns that appear after DataFrame exports.
        drop_cols = [c for c in df.columns if str(c).strip().upper().startswith("UNNAMED")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        col_names = [str(c).strip().upper() for c in df.columns]
        flux_cols = [
            c
            for c in df.columns
            if str(c).strip().upper().startswith("FLUX")
        ]

        # Kepler format: LABEL + FLUX.1..FLUX.3197 (or FLUX-1 style).
        if flux_cols and len(flux_cols) >= 100:
            flux_cols = _sort_flux_columns(flux_cols)
            label_col = next((c for c in df.columns if str(c).strip().upper() == "LABEL"), None)
            samples = []
            for row_idx, row in df.iterrows():
                flux = _clean_flux_array(row[flux_cols].to_numpy())
                label_val = row[label_col] if label_col is not None else None
                samples.append(
                    _sample_record_from_flux(
                        flux,
                        row_index=int(row_idx),
                        label=None if label_val is None or pd.isna(label_val) else float(label_val),
                    )
                )
            if samples:
                return samples

        # One-row wide numeric table without FLUX column names.
        numeric_cols = []
        for c in df.columns:
            cu = str(c).strip().upper()
            if cu in {"LABEL", "TARGET"}:
                continue
            maybe_num = pd.to_numeric(df[c], errors="coerce")
            if maybe_num.notna().sum() > 0:
                numeric_cols.append(c)
        if len(df) == 1 and len(numeric_cols) >= 100:
            return [_sample_record_from_flux(_clean_flux_array(df.iloc[0][numeric_cols].to_numpy()), row_index=0, label=None)]

        numeric = pd.to_numeric(df.stack(), errors="coerce")
        if numeric.notna().all():
            try:
                return _build_sample_sequence(numeric.to_list())
            except ValueError:
                pass

        # Tall format: single numeric column (with or without header).
        if len(df.columns) == 1:
            if source_text is not None:
                lines = [line.strip() for line in source_text.splitlines() if line.strip()]
                numeric_start = next((idx for idx, line in enumerate(lines) if _is_numeric_line(line)), None)
                if numeric_start is not None:
                    numeric_values = pd.to_numeric(pd.Series(lines[numeric_start:]), errors="coerce").dropna().to_list()
                    if numeric_values:
                        return _build_sample_sequence(numeric_values)
            return [_sample_record_from_flux(_clean_flux_array(df.iloc[:, 0].to_numpy()), row_index=None, label=None)]

    if source_text is not None:
        lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        numeric_start = next((idx for idx, line in enumerate(lines) if _is_numeric_line(line)), None)
        if numeric_start is not None:
            numeric_values = pd.to_numeric(pd.Series(lines[numeric_start:]), errors="coerce").dropna().to_list()
            if numeric_values:
                return _build_sample_sequence(numeric_values)

    # Backward-compatible path: single-column flux file.
    raw = np.genfromtxt(io.StringIO(source_text) if source_text is not None else source, delimiter=",", skip_header=1)
    if raw.ndim > 1:
        raw = raw[:, 0]
    flux = _clean_flux_array(raw)
    return [_sample_record_from_flux(flux, row_index=None, label=None)]


def plot_flux_figure(x, title):
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    index = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(14, 5.4))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(index, values, color="#1f77b4", linewidth=1.5)

    ax.set_title(title, color="black", fontsize=12, fontweight="normal")
    ax.set_xlabel("Time Step", color="black", fontsize=10)
    ax.set_ylabel("Flux", color="black", fontsize=10)
    ax.tick_params(colors="black", labelsize=10)
    ax.grid(True, color="#b0b0b0", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_flux(flux, title="Raw Light Curve"):
    fig = plot_flux_figure(flux, title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_transformed_signal(signal):
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    t = np.linspace(0, 80, len(values))
    fig, ax = plt.subplots(figsize=(9, 3.4))

    fig.patch.set_facecolor("#0f141b")
    ax.set_facecolor("#121b25")

    ax.plot(t, values, color="#ffcb6b", linewidth=1.05, alpha=0.95)
    ax.fill_between(t, values, np.median(values), color="#ffcb6b", alpha=0.08)

    ax.set_title("Transformed Signal", color="#f7f4ea", fontsize=12, fontweight="bold")
    ax.set_xlabel("Index", color="#f7f4ea", fontsize=10)
    ax.set_ylabel("Transformed Value", color="#f7f4ea", fontsize=10)
    ax.tick_params(colors="#d3dfef", labelsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2f4d60")

    ax.grid(True, color="#2b4050", alpha=0.35, linestyle="--", linewidth=0.6)
    ax.set_xlim(t[0], t[-1])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor="#0f141b", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _read_raw_sample_flux(sample_path):
    with open(sample_path, "r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]

    return _parse_exact_flux_values(values)


SAMPLE_GRAPH_DEFINITIONS = [
    {
        "filename": "sample_graph_1.csv",
        "title": "Exoplanet Sample (Index 0)",
        "pattern": "exo_0",
        "expected_label": "Exoplanet Detected",
        "input_mode": "model_ready",
    },
    {
        "filename": "sample_graph_2.csv",
        "title": "Exoplanet Sample (Index 1)",
        "pattern": "exo_1",
        "expected_label": "Exoplanet Detected",
        "input_mode": "model_ready",
    },
    {
        "filename": "sample_graph_3.csv",
        "title": "Non-Exoplanet Sample (Index 5)",
        "pattern": "non_5",
        "expected_label": "No Exoplanet Detected",
        "input_mode": "model_ready",
    },
    {
        "filename": "sample_graph_4.csv",
        "title": "Non-Exoplanet Sample (Index 6)",
        "pattern": "non_6",
        "expected_label": "No Exoplanet Detected",
        "input_mode": "model_ready",
    },
    {
        "filename": "sample_graph_5.csv",
        "title": "Non-Exoplanet Sample (Index 7)",
        "pattern": "non_7",
        "expected_label": "No Exoplanet Detected",
        "input_mode": "model_ready",
    },
]


def _edge_weight(x, scale):
    return np.exp(-x / scale) + np.exp(-(SIGNAL_LENGTH - 1 - x) / scale)


def _add_spike(flux, x, center, height, width=4.0):
    flux += height * np.exp(-0.5 * ((x - center) / width) ** 2)


def _generated_sample_graph_flux(pattern):
    seed = 20260517 + sum(ord(ch) for ch in pattern)
    rng = np.random.default_rng(seed)
    x = np.arange(SIGNAL_LENGTH, dtype=np.float64)

    if pattern == "exo_0":
        edge = _edge_weight(x, 165.0)
        flux = -0.25 + 0.12 * np.sin(x / 45.0) + rng.normal(0.0, 0.22 + 0.62 * edge)
        spikes = [
            (85, 1.6, 5), (145, 2.0, 5), (260, 6.0, 4), (430, 1.6, 7),
            (535, 3.0, 4), (795, 1.6, 5), (1065, 1.3, 5), (1320, 1.0, 4),
            (1590, 1.3, 6), (2130, 1.2, 5), (2400, 1.4, 5), (2665, 3.0, 5),
            (2930, 6.0, 4), (3060, 2.1, 5),
        ]
    elif pattern == "exo_1":
        edge = _edge_weight(x, 210.0)
        flux = -0.82 + 0.12 * np.sin(x / 42.0) + rng.normal(0.0, 0.16 + 0.48 * edge)
        spikes = [
            (5, 2.5, 6), (120, 2.5, 5), (265, 6.0, 4), (535, 1.7, 5),
            (790, 1.4, 5), (1065, 0.8, 6), (1325, 0.7, 6), (1595, 0.7, 6),
            (1865, 0.7, 6), (2135, 0.9, 6), (2400, 1.2, 5), (2665, 1.4, 5),
            (2930, 6.0, 4), (3000, 1.6, 5), (3040, 2.7, 4), (3192, 3.0, 5),
        ]
    elif pattern == "non_5":
        dome = np.sin(np.pi * x / (SIGNAL_LENGTH - 1))
        flux = -0.18 + 1.05 * dome + rng.normal(0.0, 0.28 + 0.08 * dome)
        spikes = [
            (165, -1.3, 4), (790, 0.9, 5), (1010, 0.8, 5), (1315, 1.0, 4),
            (1410, 0.9, 4), (1765, 0.9, 5), (1885, 1.0, 4), (2400, 1.0, 4),
            (3035, -1.45, 4),
        ]
    elif pattern == "non_6":
        edge = _edge_weight(x, 250.0)
        flux = 0.14 + 0.08 * np.sin(x / 70.0) + rng.normal(0.0, 0.22 + 0.55 * edge)
        spikes = [
            (35, 1.4, 4), (65, 2.1, 5), (95, 1.6, 5), (210, 1.0, 5),
            (365, 1.0, 5), (1285, 0.7, 5), (1765, -0.7, 5), (2190, 0.8, 5),
            (2690, 0.9, 5), (3010, 1.5, 5), (3135, 2.3, 5),
        ]
    else:
        edge = _edge_weight(x, 170.0)
        flux = -0.55 + 0.08 * np.sin(x / 60.0) + rng.normal(0.0, 0.16 + 0.75 * edge)
        spikes = [
            (55, 1.9, 5), (110, 3.2, 4), (285, 3.9, 4), (580, 1.4, 5),
            (870, 0.8, 5), (1165, 0.9, 5), (1455, 0.8, 5), (1745, 0.8, 5),
            (2030, 0.9, 5), (2325, 0.9, 5), (2620, 1.3, 5), (2905, 3.8, 4),
            (3125, 2.8, 5),
        ]

    for center, height, width in spikes:
        _add_spike(flux, x, center, height, width)

    return flux.astype(np.float32)


def _write_generated_sample_file(path, definition):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flux = _generated_sample_graph_flux(definition["pattern"])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(f"{float(value):.8f}" for value in flux))
        handle.write("\n")

    table_path = _sample_table_path(path)
    flux_to_model_input_table(flux).to_csv(table_path, index=False)


def _sample_table_path(sample_path):
    root, ext = os.path.splitext(sample_path)
    return f"{root}_table{ext}"


def _load_sample_model_input_table(sample_path):
    table_path = _sample_table_path(sample_path)
    if os.path.isfile(table_path):
        table = pd.read_csv(table_path)
        table_to_flux(table)
        return table

    raw_flux = _read_raw_sample_flux(sample_path)
    return flux_to_model_input_table(raw_flux)


def _resolve_sample_graph_files():
    sample_files = []
    fallback_dir = os.path.join(tempfile.gettempdir(), "exosearch_sample_graphs")

    for definition in SAMPLE_GRAPH_DEFINITIONS:
        bundled_path = os.path.join(APP_DIR, definition["filename"])
        try:
            _write_generated_sample_file(bundled_path, definition)
            sample_files.append(bundled_path)
            continue
        except OSError as exc:
            fallback_path = os.path.join(fallback_dir, definition["filename"])
            _debug_log(f"could not write {definition['filename']} to app dir; using temp path: {exc}")
            _write_generated_sample_file(fallback_path, definition)
            sample_files.append(fallback_path)

    return sample_files


def _interpret_predictions(preds):
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2 or preds.shape[0] < 1:
        raise ValueError(f"Unexpected model output shape: {preds.shape}")

    if preds.shape[-1] == 1:
        raw_score = float(preds[0][0])
        exo_score = _raw_score_to_exo_score(raw_score)
        confidences = {
            CLASS_LABELS[0]: 1.0 - exo_score,
            CLASS_LABELS[1]: exo_score,
        }
        predicted_label = _binary_prediction_label(exo_score)
    else:
        confidences = {
            CLASS_LABELS[i]: float(preds[0][i])
            for i in range(min(len(CLASS_LABELS), preds.shape[-1]))
        }
        exo_score = float(confidences.get(CLASS_LABELS[1], 0.0))
        predicted_label = CLASS_LABELS[int(np.argmax(preds[0]))]

    return predicted_label, exo_score, confidences


def _prepare_model_input(raw_flux, input_mode):
    raw_flux = np.asarray(raw_flux, dtype=np.float32).reshape(-1)
    if raw_flux.size != SIGNAL_LENGTH:
        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values.")

    if input_mode == "model_ready":
        return raw_flux.reshape(1, SIGNAL_LENGTH, 1).astype(np.float32)

    return preprocess_flux(raw_flux)


def _binary_expected_label(label):
    return 1 if label == CLASS_LABELS[1] else 0


def _best_threshold_for_scores(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    candidates = {0.0, 1.0}

    for score in scores:
        candidates.add(float(score))

    ordered = np.sort(np.unique(scores))
    for left, right in zip(ordered[:-1], ordered[1:]):
        candidates.add(float((left + right) / 2.0))

    best_accuracy = -1.0
    best_threshold = DEFAULT_PREDICTION_THRESHOLD
    for threshold in sorted(candidates):
        predicted = (scores >= threshold).astype(np.int64)
        accuracy = float(np.mean(predicted == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_accuracy, best_threshold


def _calibrate_score_interpretation_from_samples():
    global PREDICTION_THRESHOLD, SCORE_DIRECTION

    if not MODEL_LOADED or model is None:
        return ""

    if SCORE_POLARITY not in {"auto", "normal", "inverted"}:
        _debug_log(f"unknown EXO_SCORE_POLARITY={SCORE_POLARITY!r}; using auto")

    raw_scores = []
    expected_labels = []

    try:
        for sample_path, definition in zip(SAMPLE_GRAPH_FILES, SAMPLE_GRAPH_DEFINITIONS):
            model_input_table = _load_sample_model_input_table(sample_path)
            raw_flux = table_to_flux(model_input_table)
            signal = _prepare_model_input(raw_flux, definition.get("input_mode", "model_ready"))
            preds = np.asarray(model.predict(signal, verbose=0), dtype=np.float64)
            if preds.shape[-1] != 1:
                return "Sample score calibration skipped: model output is not a single sigmoid score."

            raw_score = float(preds[0][0])
            raw_scores.append(raw_score)
            expected_labels.append(_binary_expected_label(definition["expected_label"]))
            _debug_log(
                f"sample calibration raw {definition['filename']}: raw_score={raw_score:.6f} "
                f"expected={definition['expected_label']!r}"
            )
    except Exception as exc:
        message = f"Sample score calibration could not run: {exc}"
        _debug_log(message)
        return message

    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    expected_labels = np.asarray(expected_labels, dtype=np.int64)
    normal_accuracy, normal_threshold = _best_threshold_for_scores(raw_scores, expected_labels)
    inverted_accuracy, inverted_threshold = _best_threshold_for_scores(1.0 - raw_scores, expected_labels)

    if SCORE_POLARITY == "normal":
        chosen_direction = "normal"
        chosen_accuracy = normal_accuracy
        chosen_threshold = normal_threshold
    elif SCORE_POLARITY == "inverted":
        chosen_direction = "inverted"
        chosen_accuracy = inverted_accuracy
        chosen_threshold = inverted_threshold
    elif inverted_accuracy > normal_accuracy:
        chosen_direction = "inverted"
        chosen_accuracy = inverted_accuracy
        chosen_threshold = inverted_threshold
    else:
        chosen_direction = "normal"
        chosen_accuracy = normal_accuracy
        chosen_threshold = normal_threshold

    SCORE_DIRECTION = chosen_direction
    if os.getenv("EXO_PREDICTION_THRESHOLD") is None:
        PREDICTION_THRESHOLD = chosen_threshold

    calibrated_scores = 1.0 - raw_scores if SCORE_DIRECTION == "inverted" else raw_scores
    score_bits = ", ".join(f"{idx + 1}={score:.4f}" for idx, score in enumerate(calibrated_scores))
    message = (
        f"Sample-calibrated scoring: direction={SCORE_DIRECTION}, "
        f"threshold={PREDICTION_THRESHOLD:.4f}, labeled-sample accuracy={chosen_accuracy * 100:.1f}%, "
        f"exo scores: {score_bits}"
    )
    _debug_log(message)
    return message


def _predict_single_flux(raw_flux, context, title="Raw Light Curve", input_mode="raw"):
    raw_flux = np.asarray(raw_flux, dtype=np.float32).reshape(-1)
    if raw_flux.size != SIGNAL_LENGTH:
        raise ValueError(f"Input must contain exactly {SIGNAL_LENGTH} flux values.")

    _debug_log(f"{context} raw_input_stats: {_array_stats(raw_flux)}")
    signal = _prepare_model_input(raw_flux, input_mode)
    _debug_log(f"{context} tensor_before_prediction: shape={signal.shape} dtype={signal.dtype}")
    _debug_log(f"{context} model_input_mode={input_mode!r} model_input_stats: {_array_stats(signal)}")
    preds = model.predict(signal, verbose=0)
    _debug_log(f"{context} raw_model_output: shape={preds.shape} values={np.asarray(preds).tolist()}")

    predicted_label, exo_score, confidences = _interpret_predictions(preds)
    _debug_log(
        f"{context} class_probs: {{'No Exoplanet Detected': {confidences[CLASS_LABELS[0]]:.6f}, "
        f"'Exoplanet Detected': {confidences[CLASS_LABELS[1]]:.6f}}}"
    )
    _debug_log(
        f"{context} calibrated_exo_score={exo_score:.6f} threshold={PREDICTION_THRESHOLD:.3f} "
        f"score_direction={SCORE_DIRECTION!r} "
        f"predicted_label={predicted_label!r}"
    )

    return {
        "raw_flux": raw_flux,
        "signal": signal,
        "predicted_label": predicted_label,
        "exo_score": exo_score,
        "confidences": confidences,
        "chart": plot_flux(raw_flux, title=title),
        "transformed_chart": plot_transformed_signal(signal),
    }


def _predict_single_table(model_input_table, context, title="Raw Light Curve", input_mode="raw"):
    raw_flux = table_to_flux(model_input_table)
    result = _predict_single_flux(raw_flux, context, title=title, input_mode=input_mode)
    result["model_input_table"] = flux_to_model_input_table(raw_flux)
    result["model_input_table_preview"] = _model_input_table_preview(result["model_input_table"])
    return result


def _source_from_loaded_sample(sample_input):
    if isinstance(sample_input, dict) and "table" in sample_input:
        return {
            "table": sample_input["table"],
            "row_index": None,
            "label": None,
            "title": sample_input.get("title") or "Raw Light Curve",
            "expected_label": sample_input.get("expected_label"),
            "filename": sample_input.get("filename"),
            "input_mode": sample_input.get("input_mode", "raw"),
        }

    if isinstance(sample_input, (list, tuple, np.ndarray)) and np.asarray(sample_input).size > 0:
        flux = _parse_exact_flux_values(np.asarray(sample_input).reshape(-1))
        return _sample_record_from_flux(flux, row_index=None, label=None, title="Raw Light Curve")

    return None


def load_sample_preview(sample_path, title, expected_label=None):
    definition = next((item for item in SAMPLE_GRAPH_DEFINITIONS if item["filename"] == os.path.basename(sample_path)), {})
    input_mode = definition.get("input_mode", "model_ready")
    model_input_table = _load_sample_model_input_table(sample_path)
    raw_flux = table_to_flux(model_input_table)
    filename = os.path.basename(sample_path)
    sample = np.asarray(raw_flux, dtype=np.float32).reshape(-1)
    print("LOADED SAMPLE:", filename, flush=True)
    print("FIRST 10 VALUES:", sample[:10], flush=True)

    sample_state = {
        "table": model_input_table.to_dict(orient="records"),
        "row_index": None,
        "label": None,
        "title": title,
        "expected_label": expected_label,
        "filename": filename,
        "input_mode": input_mode,
    }
    table_preview = _model_input_table_preview(sample_state["table"])
    chart_image = plot_flux(np.copy(sample), title=title)
    transformed_image = None
    if not MODEL_LOADED or model is None:
        return (
            sample_state,
            chart_image,
            transformed_image,
            table_preview,
            {},
        )

    try:
        result = _predict_single_table(sample_state["table"], f"sample button {filename}", title=title, input_mode=input_mode)
        best_score = result["exo_score"]
        prediction_text = f"{result['predicted_label']} ({best_score * 100:.2f}% exoplanet probability)"
        return (
            sample_state,
            result["chart"],
            result["transformed_chart"],
            result["model_input_table_preview"],
            result["confidences"],
        )
    except Exception as exc:
        return (
            sample_state,
            chart_image,
            transformed_image,
            table_preview,
            {},
        )


def load_sample_1():
    return load_sample_preview(
        SAMPLE_GRAPH_FILES[0],
        SAMPLE_GRAPH_DEFINITIONS[0]["title"],
        SAMPLE_GRAPH_DEFINITIONS[0]["expected_label"],
    )


def load_sample_2():
    return load_sample_preview(
        SAMPLE_GRAPH_FILES[1],
        SAMPLE_GRAPH_DEFINITIONS[1]["title"],
        SAMPLE_GRAPH_DEFINITIONS[1]["expected_label"],
    )


def load_sample_3():
    return load_sample_preview(
        SAMPLE_GRAPH_FILES[2],
        SAMPLE_GRAPH_DEFINITIONS[2]["title"],
        SAMPLE_GRAPH_DEFINITIONS[2]["expected_label"],
    )


def load_sample_4():
    return load_sample_preview(
        SAMPLE_GRAPH_FILES[3],
        SAMPLE_GRAPH_DEFINITIONS[3]["title"],
        SAMPLE_GRAPH_DEFINITIONS[3]["expected_label"],
    )


def load_sample_5():
    return load_sample_preview(
        SAMPLE_GRAPH_FILES[4],
        SAMPLE_GRAPH_DEFINITIONS[4]["title"],
        SAMPLE_GRAPH_DEFINITIONS[4]["expected_label"],
    )


def predict_exoplanet(sample_input, table_input, graph_input, pasted_input):
    loaded_sample = _source_from_loaded_sample(sample_input)
    if loaded_sample is not None:
        samples = [loaded_sample]
    else:
        source = None
        if graph_input is not None:
            source = graph_input
        elif isinstance(pasted_input, str) and pasted_input.strip():
            source = pasted_input
        else:
            source = table_input

        if source is None:
            return (
                None,
                None,
                None,
                {},
            )

        try:
            samples = parse_flux_csv(source)
        except Exception as exc:
            return (
                None,
                None,
                None,
                {},
            )

    if not samples:
        return (
            None,
            None,
            None,
            {},
        )

    if not MODEL_LOADED or model is None:
        return (
            None,
            None,
            None,
            {},
        )

    try:
        best_result = None
        found_exoplanet = None

        for idx, sample in enumerate(samples, start=1):
            model_input_table = sample["table"]
            title = sample.get("title") or "Raw Light Curve"
            input_mode = sample.get("input_mode", "raw")
            prediction = _predict_single_table(model_input_table, f"sample {idx}", title=title, input_mode=input_mode)
            result = {
                "sample": sample,
                "confidences": prediction["confidences"],
                "predicted_label": prediction["predicted_label"],
                "exo_score": prediction["exo_score"],
                "chart": prediction["chart"],
                "transformed_chart": prediction["transformed_chart"],
                "model_input_table_preview": prediction["model_input_table_preview"],
                "scanned_rows": idx,
            }

            if best_result is None or result["exo_score"] > best_result["exo_score"]:
                best_result = result

            if result["predicted_label"] == CLASS_LABELS[1]:
                found_exoplanet = result
                break

        chosen = found_exoplanet if found_exoplanet is not None else best_result
        row_index = chosen["sample"]["row_index"]
        label_val = chosen["sample"]["label"]

        location_bits = []
        if row_index is not None:
            location_bits.append(f"row {row_index}")
        if label_val is not None:
            location_bits.append(f"LABEL={int(label_val)}")
        location = f" | {' | '.join(location_bits)}" if location_bits else ""

        if found_exoplanet is not None and len(samples) > 1:
            prediction_text = (
                f"{chosen['predicted_label']} ({chosen['exo_score'] * 100:.2f}% exoplanet probability)"
                f"{location} | found after scanning {chosen['scanned_rows']} rows"
            )
        elif len(samples) > 1:
            prediction_text = (
                f"No row crossed the exoplanet threshold after scanning {len(samples)} rows. "
                f"Best candidate: {chosen['predicted_label']} "
                f"(exo score {chosen['exo_score'] * 100:.2f}%){location}"
            )
        else:
            best_score = chosen["exo_score"]
            prediction_text = f"{chosen['predicted_label']} ({best_score * 100:.2f}% exoplanet probability)"

        return (
            chosen["chart"],
            chosen["transformed_chart"],
            chosen["model_input_table_preview"],
            chosen["confidences"],
        )

    except Exception as exc:
        return (
            None,
            None,
            None,
            {},
        )


def _run_sample_pipeline_check():
    if not MODEL_LOADED or model is None:
        return ""

    results = []
    try:
        for sample_path, definition in zip(SAMPLE_GRAPH_FILES, SAMPLE_GRAPH_DEFINITIONS):
            model_input_table = _load_sample_model_input_table(sample_path)
            raw_flux = table_to_flux(model_input_table)
            signal = _prepare_model_input(raw_flux, definition.get("input_mode", "model_ready"))
            preds = model.predict(signal, verbose=0)
            predicted_label, exo_score, _ = _interpret_predictions(preds)
            expected_label = definition["expected_label"]
            passed = predicted_label == expected_label
            results.append(
                {
                    "title": definition["title"],
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "exo_score": exo_score,
                    "passed": passed,
                }
            )
            _debug_log(
                f"sample sanity {definition['filename']}: expected={expected_label!r} "
                f"predicted={predicted_label!r} exo_score={exo_score:.6f} passed={passed}"
            )
    except Exception as exc:
        message = f"Sample pipeline sanity check could not run: {exc}"
        _debug_log(message)
        return message

    if results and all(result["predicted"] == CLASS_LABELS[0] for result in results):
        score_bits = ", ".join(f"{idx + 1}={result['exo_score']:.4f}" for idx, result in enumerate(results))
        message = (
            "Sample pipeline warning: all five labeled sample graphs classified as non-exoplanet. "
            f"Raw exoplanet scores: {score_bits}. Check preprocessing/training data match."
        )
        _debug_log(message)
        return message

    failures = [result for result in results if not result["passed"]]
    if failures:
        mismatch_bits = ", ".join(
            f"{failure['title']} expected {failure['expected']} got {failure['predicted']} "
            f"(exo_score={failure['exo_score']:.4f})"
            for failure in failures
        )
        message = f"Sample pipeline warning: {mismatch_bits}"
        _debug_log(message)
        return message

    message = "Sample pipeline sanity check passed for all five labeled sample graphs."
    _debug_log(message)
    return message


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg0: #050912;
    --bg1: #08131d;
    --bg2: #0b1723;
    --card: rgba(10, 20, 31, 0.82);
    --card-2: rgba(13, 24, 36, 0.92);
    --text: #f3fbff;
    --muted: #9fb8cb;
    --accent: #36f4ff;
    --accent-2: #9bff72;
    --accent-3: #7ea8ff;
    --line: rgba(75, 141, 181, 0.65);
    --line-strong: rgba(102, 199, 255, 0.95);
    --input-bg: #09141f;
    --input-text: #f4fbff;
    --input-border: rgba(79, 146, 187, 0.72);
}

.gradio-container {
    width: 100vw !important;
    max-width: 100vw !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
  background:
                radial-gradient(900px 460px at 12% 0%, rgba(54, 244, 255, 0.16) 0%, transparent 55%),
                radial-gradient(700px 420px at 88% 8%, rgba(155, 255, 114, 0.10) 0%, transparent 52%),
                radial-gradient(600px 380px at 50% 105%, rgba(126, 168, 255, 0.08) 0%, transparent 48%),
                linear-gradient(160deg, #04070d 0%, #08111a 48%, #0b1723 100%);
  min-height: 100vh;
}

html,
body {
    width: 100%;
    height: 100%;
    margin: 0 !important;
    overflow-x: hidden;
    background: #050910;
}

.gradio-container .main {
    width: 100% !important;
    max-width: none !important;
    padding: 16px clamp(14px, 2.2vw, 32px) 24px !important;
}

.gradio-container .wrap {
    width: 100% !important;
    max-width: none !important;
}

.page-header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    border: 1px solid var(--line);
    border-radius: 24px;
    background:
        linear-gradient(180deg, rgba(12, 22, 34, 0.96), rgba(8, 16, 25, 0.96)),
        linear-gradient(120deg, rgba(54, 244, 255, 0.10), rgba(155, 255, 114, 0.04));
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    padding: 20px 24px;
    margin: 0 0 16px 0;
    position: relative;
    overflow: hidden;
}

.page-header::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(54, 244, 255, 0.08), transparent);
    pointer-events: none;
}

.page-header__copy {
    flex: 1 1 auto;
    min-width: 0;
}

.page-header__eyebrow {
    color: var(--accent) !important;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 8px;
}

.page-header h1 {
    margin: 0;
    font-size: clamp(34px, 4.8vw, 62px);
    line-height: 1.02;
    letter-spacing: 0.01em;
    color: #effcff !important;
}

.page-header p {
    margin: 10px 0 0;
    color: var(--muted) !important;
    font-size: 14px;
    line-height: 1.6;
    max-width: 74ch;
}

.page-header__status {
    flex: 0 0 360px;
    align-self: stretch;
}

.dashboard-grid {
    display: grid !important;
    grid-template-columns: minmax(320px, 0.88fr) minmax(680px, 1.12fr);
    align-items: stretch;
    gap: 14px !important;
}

.dashboard-grid > * {
    min-width: 0;
}

.card {
    border: 1px solid var(--line) !important;
    border-radius: 24px !important;
    background: linear-gradient(180deg, var(--card-2), var(--card)) !important;
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    padding: 20px 20px 18px;
    height: 100%;
}

.card > .gr-markdown:first-child h3,
.card > .gr-markdown:first-child h4 {
    margin-top: 0;
    margin-bottom: 6px;
}

.card > .gr-markdown:first-child h3 {
    font-size: 21px;
}

.card > .gr-markdown:first-child p {
    margin-bottom: 0;
    color: var(--muted) !important;
}

.card .gr-markdown h3,
.card .gr-markdown h4 {
    margin-top: 0;
    margin-bottom: 8px;
}

.card .gr-markdown p {
    margin-top: 0;
}

.input-card,
.processing-card,
.signal-card,
.prediction-card-shell {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.signal-card {
    min-height: 920px;
}

.signal-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.signal-placeholder {
    min-height: 370px;
    border-radius: 22px;
    border: 1px dashed rgba(89, 164, 205, 0.78);
    background:
        radial-gradient(500px 220px at 50% 30%, rgba(54, 244, 255, 0.08), transparent 65%),
        rgba(9, 18, 28, 0.74);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 30px 24px;
}

.signal-placeholder__title {
    font-size: clamp(24px, 2.5vw, 36px);
    font-weight: 700;
    color: #f4fbff;
}

.signal-placeholder__body {
    margin-top: 8px;
    color: var(--muted) !important;
    max-width: 46ch;
    font-size: 13px;
    line-height: 1.5;
}

.signal-image {
    width: 100% !important;
}

.signal-image img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain;
}

.signal-image--raw {
    min-height: 680px;
}

.signal-image--transformed {
    min-height: 290px;
}

.sample-button-row {
    gap: 10px !important;
    flex-wrap: wrap;
}

.sample-button {
    flex: 1 1 112px;
    min-width: 112px;
}

.sample-button button,
.sample-button .gr-button {
    width: 100%;
    border: 1px solid rgba(102, 199, 255, 0.88) !important;
    background: linear-gradient(180deg, rgba(9, 20, 31, 0.92), rgba(6, 14, 22, 0.92)) !important;
    color: #e8fbff !important;
    box-shadow: inset 0 0 0 1px rgba(54, 244, 255, 0.05);
    transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease;
}

.sample-button button:hover,
.sample-button .gr-button:hover {
    transform: translateY(-1px);
    border-color: var(--accent) !important;
    color: #ffffff !important;
    background: linear-gradient(180deg, rgba(54, 244, 255, 0.12), rgba(9, 20, 31, 0.96)) !important;
    box-shadow: 0 12px 26px rgba(54, 244, 255, 0.14);
}

.action-row {
    gap: 10px !important;
}

.action-row .gr-button {
    min-height: 48px;
}

.action-row .gr-button.primary {
    box-shadow: 0 10px 24px rgba(54, 244, 255, 0.22);
}

.pipeline-steps {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 8px;
}

.pipeline-step {
    border: 1px solid rgba(102, 199, 255, 0.55);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(10, 20, 32, 0.95), rgba(8, 16, 25, 0.9));
    padding: 11px 12px;
    text-align: center;
    font-size: 12px;
    line-height: 1.35;
}

.pipeline-step span {
    display: block;
    margin-bottom: 4px;
    color: var(--accent);
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

.pipeline-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted) !important;
    font-size: 18px;
    font-weight: 700;
}

.prediction-card {
    border-radius: 24px;
    border: 1px solid rgba(102, 199, 255, 0.62);
    background:
        radial-gradient(500px 180px at 50% 0%, rgba(54, 244, 255, 0.08), transparent 60%),
        rgba(8, 17, 27, 0.82);
    padding: 18px 18px 16px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 14px 30px rgba(0, 0, 0, 0.18);
}

.prediction-card--exo {
    --prediction-accent: #43d17f;
    --prediction-accent-soft: rgba(67, 209, 127, 0.14);
}

.prediction-card--non {
    --prediction-accent: #4da3ff;
    --prediction-accent-soft: rgba(77, 163, 255, 0.14);
}

.prediction-card--empty {
    min-height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.prediction-card__eyebrow {
    color: var(--muted) !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-family: 'IBM Plex Mono', monospace;
}

.prediction-card__topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 10px;
}

.prediction-badge {
    display: inline-flex;
    align-items: center;
    padding: 10px 14px;
    border-radius: 999px;
    border: 1px solid var(--prediction-accent);
    background: var(--prediction-accent-soft);
    color: var(--prediction-accent) !important;
    font-size: 15px;
    font-weight: 700;
}

.prediction-score {
    font-size: clamp(28px, 4vw, 48px);
    line-height: 1;
    font-weight: 700;
    color: var(--prediction-accent) !important;
}

.prediction-card__bar {
    width: 100%;
    height: 15px;
    margin-top: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.08);
}

.prediction-card__bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--prediction-accent), rgba(255, 255, 255, 0.28));
}

.prediction-card__caption,
.prediction-card__meta,
.prediction-card__detail,
.prediction-card__empty-body {
    margin-top: 8px;
    color: var(--muted) !important;
    font-size: 12px;
    line-height: 1.45;
}

.prediction-card__detail {
    color: #edf8ff !important;
}

.prediction-card__meta {
    color: var(--accent-3) !important;
}

.prediction-card__empty-title {
    margin-top: 8px;
    color: #f6fdff !important;
    font-size: 21px;
    font-weight: 700;
}

@media (max-width: 1100px) {
    .page-header {
        flex-direction: column;
    }

    .page-header__status {
        width: 100%;
        flex-basis: auto;
    }

    .pipeline-steps {
        grid-template-columns: 1fr;
    }

    .pipeline-arrow {
        display: none;
    }

    .signal-image--raw {
        min-height: 440px;
    }

    .dashboard-grid {
        grid-template-columns: 1fr;
    }
}

html,
body {
    width: 100%;
    height: 100%;
    margin: 0 !important;
    overflow-x: hidden;
}

.gradio-container .main,
.gradio-container .wrap {
    width: 100% !important;
    max-width: none !important;
}

.gradio-container * {
  color: var(--text) !important;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose span,
.gradio-container label,
.gradio-container legend {
  color: var(--text) !important;
}

.hero {
    border: 1px solid var(--line);
        border-radius: 20px;
        background:
                radial-gradient(650px 220px at 10% 0%, rgba(54, 244, 255, 0.10), transparent 60%),
                linear-gradient(180deg, rgba(12, 22, 34, 0.96), rgba(8, 16, 25, 0.96));
        padding: 18px 20px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
        margin: 0 0 14px 0;
}

.hero h1 {
  margin: 0;
        font-size: clamp(28px, 3.8vw, 42px);
    letter-spacing: 0.02em;
    color: #edfaff !important;
}

.hero p {
    margin-top: 8px;
  color: var(--muted) !important;
    font-size: 14px;
    line-height: 1.55;
    max-width: 72ch;
}

.panel {
  border: 1px solid var(--line) !important;
        border-radius: 20px !important;
        background: linear-gradient(180deg, rgba(13, 23, 35, 0.96), rgba(10, 18, 28, 0.94)) !important;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255, 255, 255, 0.02);
}

.panel * {
  color: var(--text) !important;
}

.panel code {
    color: #111827 !important;
    background: #eef2f7 !important;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 0 4px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--line);
  border-radius: 999px;
    padding: 5px 11px;
    background: rgba(8, 17, 27, 0.72);
    font-size: 12px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  box-shadow: 0 0 10px currentColor;
}

.status-note {
    margin: 7px 0 0 2px;
  color: var(--muted) !important;
    font-size: 12px;
}

button, .gr-button {
    border-radius: 12px !important;
    border: 1px solid rgba(79, 146, 187, 0.82) !important;
    font-weight: 600 !important;
}

button.primary, .gr-button.primary {
        background: linear-gradient(120deg, #20b0d0, #43d17f) !important;
        color: #031a24 !important;
  text-shadow: none !important;
}

button.primary:hover, .gr-button.primary:hover {
    filter: brightness(1.05);
        box-shadow: 0 10px 22px rgba(54, 244, 255, 0.18);
}

#clear-btn {
        background: linear-gradient(120deg, #16273a, #203a52) !important;
  color: #dff6ff !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-textbox textarea,
.gradio-container .gr-textbox input {
  background: var(--input-bg) !important;
  color: var(--input-text) !important;
  border: 1px solid var(--input-border) !important;
        border-radius: 12px !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #95bad1 !important;
  opacity: 1 !important;
}

.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-group,
.gradio-container .gr-accordion,
.gradio-container .gr-accordion .label-wrap,
.gradio-container .gr-label,
.gradio-container .gr-image,
.gradio-container .gr-file,
.gradio-container .gr-dataframe,
.gradio-container .gr-markdown,
.gradio-container .block {
        background: rgba(13, 24, 36, 0.92) !important;
  color: var(--text) !important;
  border-color: var(--line) !important;
}

.gradio-container table,
.gradio-container th,
.gradio-container td {
  background: #0d1a28 !important;
  color: #effbff !important;
  border-color: #38627d !important;
}

.gradio-container .gr-button {
        box-shadow: none;
}

.gradio-container button:not(.primary),
.gradio-container .gr-button:not(.primary) {
  background: #17324a !important;
  color: #dff7ff !important;
  border: 1px solid #427a9e !important;
}

.gr-input, .gr-file, .gr-textbox, .gr-label, .gr-image {
  border-color: var(--line) !important;
}

.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .gr-label span,
.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="file-upload-dropzone"],
.gradio-container [data-testid="file-upload-dropzone"] * {
  color: #ecfbff !important;
}

.gradio-container [data-testid="file-upload-dropzone"] {
        background: #0b1623 !important;
        border: 1px dashed rgba(78, 141, 178, 0.85) !important;
}

.gradio-container .gr-accordion summary,
.gradio-container .gr-accordion button {
    background: #102334 !important;
  color: #e7f9ff !important;
}

#input-table,
#input-table * {
    color: #111827 !important;
}

#input-table {
    background: #ffffff !important;
}

#input-table table,
#input-table th,
#input-table td,
#input-table input,
#input-table textarea {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: #cbd5e1 !important;
}

.gr-markdown p, .gr-markdown li {
  color: var(--muted) !important;
}

.footer-note {
        margin-top: 10px;
  color: var(--muted) !important;
  font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}
"""

SAMPLE_GRAPH_FILES = _resolve_sample_graph_files()
IDEAL_SAMPLE_FILES = SAMPLE_GRAPH_FILES
SAMPLE_FILE = SAMPLE_GRAPH_FILES[0]
SAMPLE_GRAPH_FILE = SAMPLE_GRAPH_FILES[0]
SAMPLE_CALIBRATION_STATUS_MESSAGE = _calibrate_score_interpretation_from_samples()
SAMPLE_PIPELINE_STATUS_MESSAGE = _run_sample_pipeline_check()

with gr.Blocks(title="ExoSearch - Light Curve Analyzer") as demo:
    gr.Markdown("""
<div class="hero">
    <h1>ExoSearch Mission Console</h1>
    <p>This model uses a convolutional neural network (CNN) to classify light-curve graphs from stars. Flux is the light intensity we measure from the star, and because exoplanets do not emit their own light, a repeating dip in flux can indicate that a planet is orbiting the star and passing in front of it.</p>
    <p>Upload a raw CSV first. The sample button renders a raw preview immediately, stores the same 3197-point curve, and Analyze runs prediction after the graph is visible.</p>
</div>
    """)

    gr.HTML(model_status_html())

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("### Input")
            gr.Markdown("Upload a raw CSV first. Each sample button below renders its own saved raw curve immediately, and Analyze uses that same array for prediction.")
            graph_input = gr.File(
                label="Upload Raw Light-Curve CSV",
                file_types=[".csv", ".txt"],
                type="filepath",
            )
            sample_state = gr.State(None)

            with gr.Row():
                sample_buttons = [
                    gr.Button(f"Sample Graph {index}", variant="secondary")
                    for index in range(1, 6)
                ]

            with gr.Row():
                submit_btn = gr.Button("Analyze Curve", variant="primary")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")

            gr.Markdown("### Model Processing")
            gr.Markdown(
                """
1. FFT magnitude transform
2. Savitzky-Golay smoothing
3. L2 normalization
4. Robust scaling
                """
            )
            model_input_table_output = gr.Dataframe(
                headers=["Time Step", "Flux"],
                datatype=["str", "str"],
                interactive=False,
                row_count=(17, "dynamic"),
                column_count=(2, "fixed"),
                elem_id="input-table",
                visible=False,
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("### Outputs")
            chart_output = gr.Image(
                label="Raw Flux Preview",
                type="pil",
                interactive=False,
                height=260,
            )
            with gr.Accordion("Optional Transformed Signal", open=False):
                transformed_output = gr.Image(
                    label="Transformed Signal Preview",
                    type="pil",
                    interactive=False,
                    height=260,
                )
            result_label = gr.Label(label="Class Confidence", num_top_classes=2)
            gr.Markdown(
                "Confidence is the model score on this input, not absolute scientific certainty."
            )

    submit_btn.click(
        fn=lambda sample_data, uploaded_csv: predict_exoplanet(sample_data, None, uploaded_csv, None),
        inputs=[sample_state, graph_input],
        outputs=[chart_output, transformed_output, model_input_table_output, result_label],
    )

    sample_handlers = [
        load_sample_1,
        load_sample_2,
        load_sample_3,
        load_sample_4,
        load_sample_5,
    ]

    for sample_button, handler in zip(sample_buttons, sample_handlers):
        sample_button.click(
            fn=handler,
            inputs=[],
            outputs=[sample_state, chart_output, transformed_output, model_input_table_output, result_label],
        )

    clear_btn.click(
        fn=lambda: (
            None,
            None,
            None,
            None,
            {},
        ),
        inputs=[],
        outputs=[sample_state, graph_input, chart_output, transformed_output, model_input_table_output, result_label],
    )

    gr.Markdown(
        "<div class='footer-note'>Model artifact: exo_cnn_model.keras | UI: ExoSearch AI</div>"
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(css=CUSTOM_CSS)
