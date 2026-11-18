"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 056 ===
from itertools import product
from sklearn.ensemble import RandomForestRegressor

RF_TUNE_N, RF_FINAL_N, RF_BATCH = 200_000, train_end, 200_000
# Keep the original 200k tuning rows fixed; final fits use the full Train split.
rf_tune_pool = np.sort(
    np.random.default_rng(SEED).choice(train_end, 1_000_000, replace=False)
)
rf_tune_indices = np.sort(
    np.random.default_rng(SEED + 1).choice(rf_tune_pool, RF_TUNE_N, replace=False)
)
del rf_tune_pool
assert len(rf_tune_indices) == RF_TUNE_N and rf_tune_indices[-1] < train_end
assert RF_FINAL_N == splits['Train'][1]

def rf_x_from_records(batch):
    x = np.empty((len(batch), 3), dtype=np.float32)
    x[:, 0] = batch['x_correct']
    x[:, 1] = batch['x_wrong']
    x[:, 2] = 1.0
    return x

def rf_logh_predictions(model, batch):
    score = model.predict(rf_x_from_records(batch))
    h_pred = np.exp2(np.clip(score, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P)
    return p_pred, h_pred

rf_tune_records = records[rf_tune_indices]
rf_tune_x = rf_x_from_records(rf_tune_records)
rf_tune_y = np.log2(rf_tune_records["h_observed"])
val_start, val_stop = splits["Validation"]
rf_candidate_rows = []
for depth, leaf in product((8, 12, 16), (50, 100)):
    candidate = RandomForestRegressor(
        criterion="squared_error", n_estimators=100, max_depth=depth,
        min_samples_leaf=leaf, max_features=1.0, n_jobs=-1, random_state=SEED,
    )
    fit_started = time.perf_counter()
    candidate.fit(rf_tune_x, rf_tune_y)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    absolute_error_sum = 0.0
    for left in range(val_start, val_stop, RF_BATCH):
        right = min(left + RF_BATCH, val_stop)
        p_pred, _ = rf_logh_predictions(candidate, records[left:right])
        absolute_error_sum += float(np.abs(
            p_pred - records["p_target"][left:right]
        ).sum())
    predict_seconds = time.perf_counter() - predict_started
    rf_candidate_rows.append({
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "n_train": RF_TUNE_N,
        "n_validation": val_stop - val_start,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "MAE(p)": absolute_error_sum / (val_stop - val_start),
    })
    del candidate

rf_candidate_results = pd.DataFrame(rf_candidate_rows).sort_values(
    ["MAE(p)", "max_depth", "min_samples_leaf"],
    ascending=[True, True, False],
).reset_index(drop=True)
rf_selected_depth = int(rf_candidate_results.loc[0, "max_depth"])
rf_selected_leaf = int(rf_candidate_results.loc[0, "min_samples_leaf"])
display(rf_candidate_results.round({"fit_seconds": 2, "predict_seconds": 2, "MAE(p)": 6}))

# === ORIGINAL CELL 058 ===
import threading
import psutil

rf_process = psutil.Process()
rf_rss_before = rf_process.memory_info().rss
rf_peak_rss = [rf_rss_before]
rf_monitor_stop = threading.Event()

def rf_watch_rss():
    while not rf_monitor_stop.wait(0.1):
        rf_peak_rss[0] = max(rf_peak_rss[0], rf_process.memory_info().rss)

rf_monitor = threading.Thread(target=rf_watch_rss, daemon=True)
rf_monitor.start()
try:
    rf_final_records = records[:train_end]
    rf_final_x = rf_x_from_records(rf_final_records)
    rf_final_y = np.log2(rf_final_records["h_observed"])
    rf_e1_model = RandomForestRegressor(
        criterion="squared_error", n_estimators=100,
        max_depth=rf_selected_depth, min_samples_leaf=rf_selected_leaf,
        max_features=1.0, n_jobs=-1, random_state=SEED,
    )
    fit_started = time.perf_counter()
    rf_e1_model.fit(rf_final_x, rf_final_y)
    rf_final_fit_seconds = time.perf_counter() - fit_started
    del rf_final_records, rf_final_x, rf_final_y, rf_tune_records, rf_tune_x, rf_tune_y

    n_val = val_stop - val_start
    p_true = records["p_target"][val_start:val_stop]
    h_true = records["h_observed"][val_start:val_stop]
    days = records["delta_days"][val_start:val_stop]
    rf_e1_p_pred = np.empty(n_val, dtype=np.float64)
    rf_e1_h_pred = np.empty(n_val, dtype=np.float64)
    predict_started = time.perf_counter()
    for left in range(val_start, val_stop, RF_BATCH):
        right = min(left + RF_BATCH, val_stop)
        batch = records[left:right]
        offset_left, offset_right = left - val_start, right - val_start
        p_batch, h_batch = rf_logh_predictions(rf_e1_model, batch)
        rf_e1_p_pred[offset_left:offset_right] = p_batch
        rf_e1_h_pred[offset_left:offset_right] = h_batch
    rf_final_predict_seconds = time.perf_counter() - predict_started
finally:
    rf_monitor_stop.set()
    rf_monitor.join()
    rf_peak_rss[0] = max(rf_peak_rss[0], rf_process.memory_info().rss)

assert np.isfinite(rf_e1_p_pred).all() and np.isfinite(rf_e1_h_pred).all()
assert np.all((MIN_P <= rf_e1_p_pred) & (rf_e1_p_pred <= MAX_P))
assert np.all((MIN_H <= rf_e1_h_pred) & (rf_e1_h_pred <= MAX_H))

interior = (h_true > MIN_H) & (h_true < MAX_H) & (days > 0)
n_interior = int(interior.sum())

def rf_e1_metric_row(name, n_train, p_pred, h_pred):
    error = p_pred - p_true
    return {
        "model": name, "n_train": n_train, "n_validation": n_val,
        "MAE(p)": float(np.abs(error).mean()),
        "RMSE(p)": float(np.sqrt(np.mean(error ** 2))),
        "AUC": auc_fixed(np.rint(p_true), p_pred),
        "Spearman(h)": spearman_with_ties(h_true, h_pred),
        "n_spearman_h": n_val,
        "n_h_interior": n_interior,
        "h_interior_share": n_interior / n_val,
        "Spearman(h)_interior": spearman_with_ties(
            h_true[interior], h_pred[interior]
        ),
        "p_hat_at_bounds": float(np.mean((p_pred == MIN_P) | (p_pred == MAX_P))),
        "h_hat_at_bounds": float(np.mean((h_pred == MIN_H) | (h_pred == MAX_H))),
    }

rf_e1_validation = pd.DataFrame([
    rf_e1_metric_row("RF-logh (E1)", RF_FINAL_N, rf_e1_p_pred, rf_e1_h_pred),
])
rf_e1_resource = pd.DataFrame([{
    "n_train": RF_FINAL_N, "max_depth": rf_selected_depth,
    "min_samples_leaf": rf_selected_leaf,
    "fit_seconds": rf_final_fit_seconds,
    "predict_seconds": rf_final_predict_seconds,
    "rss_before_GiB": rf_rss_before / 2**30,
    "peak_rss_GiB": rf_peak_rss[0] / 2**30,
}])


def rf_e1_report_batch(batch, left, right):
    raw_logh = rf_e1_model.predict(rf_x_from_records(batch))
    h_pred = np.exp2(np.clip(raw_logh, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(
        np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P
    )
    error = raw_logh - np.log2(batch["h_observed"])
    return p_pred, float(np.dot(error, error))

rf_e1_fit_report = report_model_fit(
    "RF-logh (E1)", 3, "dense [x_correct, x_wrong, bias]",
    "log2(h_observed)",
    {"criterion": "squared_error", "n_estimators": 100,
     "max_depth": rf_selected_depth, "min_samples_leaf": rf_selected_leaf,
     "max_features": 1.0, "random_state": SEED},
    "MSE(raw log2(h) prediction, log2(h_observed))",
    rf_e1_resource, rf_e1_report_batch,
)

# === ORIGINAL CELL 061 ===
def rf_p_x_from_records(batch):
    x = np.empty((len(batch), 4), dtype=np.float32)
    x[:, 0] = batch['x_correct']
    x[:, 1] = batch['x_wrong']
    x[:, 2] = 1.0
    x[:, 3] = batch['delta_days']
    return x

def rf_p_predictions(model, batch):
    p_pred = np.clip(model.predict(rf_p_x_from_records(batch)), MIN_P, MAX_P)
    h_pred = np.clip(-batch["delta_days"] / np.log2(p_pred), MIN_H, MAX_H)
    return p_pred, h_pred

rf_p_tune_x = rf_p_x_from_records(records[rf_tune_indices])
rf_p_tune_y = records["p_target"][rf_tune_indices].copy()
rf_p_candidate_rows = []
for depth, leaf in product((8, 12, 16), (50, 100)):
    candidate = RandomForestRegressor(
        criterion="squared_error", n_estimators=100, max_depth=depth,
        min_samples_leaf=leaf, max_features=1.0, n_jobs=-1, random_state=SEED,
    )
    fit_started = time.perf_counter()
    candidate.fit(rf_p_tune_x, rf_p_tune_y)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    absolute_error_sum = 0.0
    for left in range(val_start, val_stop, RF_BATCH):
        right = min(left + RF_BATCH, val_stop)
        p_pred, _ = rf_p_predictions(candidate, records[left:right])
        absolute_error_sum += float(np.abs(
            p_pred - records["p_target"][left:right]
        ).sum())
    predict_seconds = time.perf_counter() - predict_started
    rf_p_candidate_rows.append({
        "max_depth": depth, "min_samples_leaf": leaf,
        "n_train": RF_TUNE_N, "n_validation": val_stop - val_start,
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "MAE(p)": absolute_error_sum / (val_stop - val_start),
    })
    del candidate

rf_p_candidate_results = pd.DataFrame(rf_p_candidate_rows).sort_values(
    ["MAE(p)", "max_depth", "min_samples_leaf"],
    ascending=[True, True, False],
).reset_index(drop=True)
rf_p_selected_depth = int(rf_p_candidate_results.loc[0, "max_depth"])
rf_p_selected_leaf = int(rf_p_candidate_results.loc[0, "min_samples_leaf"])
display(rf_p_candidate_results.round({"fit_seconds": 2, "predict_seconds": 2, "MAE(p)": 6}))

# === ORIGINAL CELL 063 ===
rf_p_process = psutil.Process()
rf_p_rss_before = rf_p_process.memory_info().rss
rf_p_peak_rss = [rf_p_rss_before]
rf_p_monitor_stop = threading.Event()

def rf_p_watch_rss():
    while not rf_p_monitor_stop.wait(0.1):
        rf_p_peak_rss[0] = max(rf_p_peak_rss[0], rf_p_process.memory_info().rss)

rf_p_monitor = threading.Thread(target=rf_p_watch_rss, daemon=True)
rf_p_monitor.start()
try:
    rf_p_final_x = rf_p_x_from_records(records[:train_end])
    rf_p_final_y = records["p_target"][:train_end]
    rf_e2_model = RandomForestRegressor(
        criterion="squared_error", n_estimators=100,
        max_depth=rf_p_selected_depth, min_samples_leaf=rf_p_selected_leaf,
        max_features=1.0, n_jobs=-1, random_state=SEED,
    )
    fit_started = time.perf_counter()
    rf_e2_model.fit(rf_p_final_x, rf_p_final_y)
    rf_e2_fit_seconds = time.perf_counter() - fit_started
    del rf_p_final_x, rf_p_final_y, rf_p_tune_x, rf_p_tune_y

    rf_e2_p_pred = np.empty(n_val, dtype=np.float64)
    rf_e2_h_pred = np.empty(n_val, dtype=np.float64)
    predict_started = time.perf_counter()
    for left in range(val_start, val_stop, RF_BATCH):
        right = min(left + RF_BATCH, val_stop)
        p_batch, h_batch = rf_p_predictions(rf_e2_model, records[left:right])
        offset_left, offset_right = left - val_start, right - val_start
        rf_e2_p_pred[offset_left:offset_right] = p_batch
        rf_e2_h_pred[offset_left:offset_right] = h_batch
    rf_e2_predict_seconds = time.perf_counter() - predict_started
finally:
    rf_p_monitor_stop.set()
    rf_p_monitor.join()
    rf_p_peak_rss[0] = max(rf_p_peak_rss[0], rf_p_process.memory_info().rss)

assert np.isfinite(rf_e2_p_pred).all() and np.isfinite(rf_e2_h_pred).all()
assert np.all((MIN_P <= rf_e2_p_pred) & (rf_e2_p_pred <= MAX_P))
assert np.all((MIN_H <= rf_e2_h_pred) & (rf_e2_h_pred <= MAX_H))
rf_e2_validation = pd.DataFrame([
    rf_e1_metric_row("RF-p (E2)", RF_FINAL_N, rf_e2_p_pred, rf_e2_h_pred),
])
rf_e2_resource = pd.DataFrame([{
    "n_train": RF_FINAL_N, "max_depth": rf_p_selected_depth,
    "min_samples_leaf": rf_p_selected_leaf,
    "fit_seconds": rf_e2_fit_seconds,
    "predict_seconds": rf_e2_predict_seconds,
    "rss_before_GiB": rf_p_rss_before / 2**30,
    "peak_rss_GiB": rf_p_peak_rss[0] / 2**30,
}])


def rf_e2_report_batch(batch, left, right):
    raw_p = rf_e2_model.predict(rf_p_x_from_records(batch))
    p_pred = np.clip(raw_p, MIN_P, MAX_P)
    error = raw_p - batch["p_target"]
    return p_pred, float(np.dot(error, error))

rf_e2_fit_report = report_model_fit(
    "RF-p (E2)", 4, "dense [x_correct, x_wrong, bias, delta_days]",
    "p_target",
    {"criterion": "squared_error", "n_estimators": 100,
     "max_depth": rf_p_selected_depth, "min_samples_leaf": rf_p_selected_leaf,
     "max_features": 1.0, "random_state": SEED},
    "MSE(raw p prediction, p_target)",
    rf_e2_resource, rf_e2_report_batch,
)

