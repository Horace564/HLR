"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 066 ===
import gc
import lightgbm as lgb
from lightgbm import LGBMRegressor

lgbm_tune_x = rf_x_from_records(records[rf_tune_indices])
lgbm_tune_y = np.log2(records["h_observed"][rf_tune_indices])
lgbm_val_x = rf_x_from_records(records[val_start:val_stop])
lgbm_val_y = np.log2(records["h_observed"][val_start:val_stop])
lgbm_val_days = records["delta_days"][val_start:val_stop]
lgbm_val_p = records["p_target"][val_start:val_stop]

def lgbm_logh_predictions(model, x, delta_days):
    score = model.predict(x, num_iteration=model.best_iteration_)
    h_pred = np.exp2(np.clip(score, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(np.exp2(-delta_days / h_pred), MIN_P, MAX_P)
    return p_pred, h_pred

lgbm_e3_candidate_rows = []
for leaves, min_child in product((15, 31, 63), (50, 200)):
    candidate = LGBMRegressor(
        objective="regression", n_estimators=1000, learning_rate=0.05,
        num_leaves=leaves, min_child_samples=min_child,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    fit_started = time.perf_counter()
    candidate.fit(
        lgbm_tune_x, lgbm_tune_y,
        eval_set=[(lgbm_val_x, lgbm_val_y)], eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    p_pred, _ = lgbm_logh_predictions(candidate, lgbm_val_x, lgbm_val_days)
    predict_seconds = time.perf_counter() - predict_started
    lgbm_e3_candidate_rows.append({
        "num_leaves": leaves, "min_child_samples": min_child,
        "best_iteration": int(candidate.best_iteration_),
        "n_train": RF_TUNE_N, "n_validation": len(lgbm_val_p),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "MAE(p)": float(np.abs(p_pred - lgbm_val_p).mean()),
    })
    del candidate, p_pred
    gc.collect()

lgbm_e3_candidate_results = pd.DataFrame(lgbm_e3_candidate_rows).sort_values(
    ["MAE(p)", "num_leaves", "min_child_samples"],
    ascending=[True, True, False],
).reset_index(drop=True)
lgbm_e3_selected_leaves = int(lgbm_e3_candidate_results.loc[0, "num_leaves"])
lgbm_e3_selected_min_child = int(lgbm_e3_candidate_results.loc[0, "min_child_samples"])
display(lgbm_e3_candidate_results.round({"fit_seconds": 2, "predict_seconds": 2, "MAE(p)": 6}))

# === ORIGINAL CELL 068 ===
lgbm_e3_process = psutil.Process()
lgbm_e3_rss_before = lgbm_e3_process.memory_info().rss
lgbm_e3_peak_rss = [lgbm_e3_rss_before]
lgbm_e3_monitor_stop = threading.Event()

def lgbm_e3_watch_rss():
    while not lgbm_e3_monitor_stop.wait(0.1):
        lgbm_e3_peak_rss[0] = max(
            lgbm_e3_peak_rss[0], lgbm_e3_process.memory_info().rss
        )

lgbm_e3_monitor = threading.Thread(target=lgbm_e3_watch_rss, daemon=True)
lgbm_e3_monitor.start()
try:
    lgbm_e3_train_x = rf_x_from_records(records[:train_end])
    lgbm_e3_train_y = np.log2(records["h_observed"][:train_end])
    lgbm_e3_model = LGBMRegressor(
        objective="regression", n_estimators=1000, learning_rate=0.05,
        num_leaves=lgbm_e3_selected_leaves,
        min_child_samples=lgbm_e3_selected_min_child,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    fit_started = time.perf_counter()
    lgbm_e3_model.fit(
        lgbm_e3_train_x, lgbm_e3_train_y,
        eval_set=[(lgbm_val_x, lgbm_val_y)], eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgbm_e3_fit_seconds = time.perf_counter() - fit_started
    del lgbm_e3_train_x, lgbm_e3_train_y, lgbm_tune_x, lgbm_tune_y
    predict_started = time.perf_counter()
    lgbm_e3_p_pred, lgbm_e3_h_pred = lgbm_logh_predictions(
        lgbm_e3_model, lgbm_val_x, lgbm_val_days
    )
    lgbm_e3_predict_seconds = time.perf_counter() - predict_started
finally:
    lgbm_e3_monitor_stop.set()
    lgbm_e3_monitor.join()
    lgbm_e3_peak_rss[0] = max(
        lgbm_e3_peak_rss[0], lgbm_e3_process.memory_info().rss
    )

assert np.isfinite(lgbm_e3_p_pred).all() and np.isfinite(lgbm_e3_h_pred).all()
assert np.all((MIN_P <= lgbm_e3_p_pred) & (lgbm_e3_p_pred <= MAX_P))
assert np.all((MIN_H <= lgbm_e3_h_pred) & (lgbm_e3_h_pred <= MAX_H))
lgbm_e3_validation = pd.DataFrame([
    rf_e1_metric_row("LGBM-logh (E3)", RF_FINAL_N,
                     lgbm_e3_p_pred, lgbm_e3_h_pred),
])
lgbm_e3_resource = pd.DataFrame([{
    "n_train": RF_FINAL_N,
    "num_leaves": lgbm_e3_selected_leaves,
    "min_child_samples": lgbm_e3_selected_min_child,
    "best_iteration": int(lgbm_e3_model.best_iteration_),
    "fit_seconds": lgbm_e3_fit_seconds,
    "predict_seconds": lgbm_e3_predict_seconds,
    "rss_before_GiB": lgbm_e3_rss_before / 2**30,
    "peak_rss_GiB": lgbm_e3_peak_rss[0] / 2**30,
}])


def lgbm_e3_report_batch(batch, left, right):
    raw_logh = lgbm_e3_model.predict(
        rf_x_from_records(batch), num_iteration=lgbm_e3_model.best_iteration_
    )
    h_pred = np.exp2(np.clip(raw_logh, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(
        np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P
    )
    error = raw_logh - np.log2(batch["h_observed"])
    return p_pred, float(np.dot(error, error))

lgbm_e3_fit_report = report_model_fit(
    "LGBM-logh (E3)", 3, "dense [x_correct, x_wrong, bias]",
    "log2(h_observed)",
    {"objective": "regression", "n_estimators": 1000,
     "learning_rate": 0.05, "num_leaves": lgbm_e3_selected_leaves,
     "min_child_samples": lgbm_e3_selected_min_child,
     "best_iteration": int(lgbm_e3_model.best_iteration_),
     "early_stopping_rounds": 50, "random_state": SEED},
    "MSE(raw log2(h) prediction, log2(h_observed))",
    lgbm_e3_resource, lgbm_e3_report_batch,
)

# === ORIGINAL CELL 071 ===
lgbm_p_tune_x = rf_p_x_from_records(records[rf_tune_indices])
lgbm_p_tune_y = records["p_target"][rf_tune_indices].copy()
lgbm_p_val_x = rf_p_x_from_records(records[val_start:val_stop])

def lgbm_p_predictions(model, x, delta_days):
    p_pred = np.clip(
        model.predict(x, num_iteration=model.best_iteration_), MIN_P, MAX_P
    )
    h_pred = np.clip(-delta_days / np.log2(p_pred), MIN_H, MAX_H)
    return p_pred, h_pred

lgbm_e4_candidate_rows = []
for leaves, min_child in product((15, 31, 63), (50, 200)):
    candidate = LGBMRegressor(
        objective="regression", n_estimators=1000, learning_rate=0.05,
        num_leaves=leaves, min_child_samples=min_child,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    fit_started = time.perf_counter()
    candidate.fit(
        lgbm_p_tune_x, lgbm_p_tune_y,
        eval_set=[(lgbm_p_val_x, lgbm_val_p)], eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    p_pred, _ = lgbm_p_predictions(candidate, lgbm_p_val_x, lgbm_val_days)
    predict_seconds = time.perf_counter() - predict_started
    lgbm_e4_candidate_rows.append({
        "num_leaves": leaves, "min_child_samples": min_child,
        "best_iteration": int(candidate.best_iteration_),
        "n_train": RF_TUNE_N, "n_validation": len(lgbm_val_p),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "MAE(p)": float(np.abs(p_pred - lgbm_val_p).mean()),
    })
    del candidate, p_pred
    gc.collect()

lgbm_e4_candidate_results = pd.DataFrame(lgbm_e4_candidate_rows).sort_values(
    ["MAE(p)", "num_leaves", "min_child_samples"],
    ascending=[True, True, False],
).reset_index(drop=True)
lgbm_e4_selected_leaves = int(lgbm_e4_candidate_results.loc[0, "num_leaves"])
lgbm_e4_selected_min_child = int(lgbm_e4_candidate_results.loc[0, "min_child_samples"])
display(lgbm_e4_candidate_results.round({"fit_seconds": 2, "predict_seconds": 2, "MAE(p)": 6}))

# === ORIGINAL CELL 073 ===
lgbm_e4_process = psutil.Process()
lgbm_e4_rss_before = lgbm_e4_process.memory_info().rss
lgbm_e4_peak_rss = [lgbm_e4_rss_before]
lgbm_e4_monitor_stop = threading.Event()

def lgbm_e4_watch_rss():
    while not lgbm_e4_monitor_stop.wait(0.1):
        lgbm_e4_peak_rss[0] = max(
            lgbm_e4_peak_rss[0], lgbm_e4_process.memory_info().rss
        )

lgbm_e4_monitor = threading.Thread(target=lgbm_e4_watch_rss, daemon=True)
lgbm_e4_monitor.start()
try:
    lgbm_e4_train_x = rf_p_x_from_records(records[:train_end])
    lgbm_e4_train_y = records["p_target"][:train_end]
    lgbm_e4_model = LGBMRegressor(
        objective="regression", n_estimators=1000, learning_rate=0.05,
        num_leaves=lgbm_e4_selected_leaves,
        min_child_samples=lgbm_e4_selected_min_child,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    fit_started = time.perf_counter()
    lgbm_e4_model.fit(
        lgbm_e4_train_x, lgbm_e4_train_y,
        eval_set=[(lgbm_p_val_x, lgbm_val_p)], eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgbm_e4_fit_seconds = time.perf_counter() - fit_started
    del lgbm_e4_train_x, lgbm_e4_train_y, lgbm_p_tune_x, lgbm_p_tune_y
    predict_started = time.perf_counter()
    lgbm_e4_p_pred, lgbm_e4_h_pred = lgbm_p_predictions(
        lgbm_e4_model, lgbm_p_val_x, lgbm_val_days
    )
    lgbm_e4_predict_seconds = time.perf_counter() - predict_started
finally:
    lgbm_e4_monitor_stop.set()
    lgbm_e4_monitor.join()
    lgbm_e4_peak_rss[0] = max(
        lgbm_e4_peak_rss[0], lgbm_e4_process.memory_info().rss
    )

assert np.isfinite(lgbm_e4_p_pred).all() and np.isfinite(lgbm_e4_h_pred).all()
assert np.all((MIN_P <= lgbm_e4_p_pred) & (lgbm_e4_p_pred <= MAX_P))
assert np.all((MIN_H <= lgbm_e4_h_pred) & (lgbm_e4_h_pred <= MAX_H))
lgbm_e4_validation = pd.DataFrame([
    rf_e1_metric_row("LGBM-p (E4)", RF_FINAL_N,
                     lgbm_e4_p_pred, lgbm_e4_h_pred),
])
lgbm_e4_resource = pd.DataFrame([{
    "n_train": RF_FINAL_N,
    "num_leaves": lgbm_e4_selected_leaves,
    "min_child_samples": lgbm_e4_selected_min_child,
    "best_iteration": int(lgbm_e4_model.best_iteration_),
    "fit_seconds": lgbm_e4_fit_seconds,
    "predict_seconds": lgbm_e4_predict_seconds,
    "rss_before_GiB": lgbm_e4_rss_before / 2**30,
    "peak_rss_GiB": lgbm_e4_peak_rss[0] / 2**30,
}])


def lgbm_e4_report_batch(batch, left, right):
    raw_p = lgbm_e4_model.predict(
        rf_p_x_from_records(batch), num_iteration=lgbm_e4_model.best_iteration_
    )
    p_pred = np.clip(raw_p, MIN_P, MAX_P)
    error = raw_p - batch["p_target"]
    return p_pred, float(np.dot(error, error))

lgbm_e4_fit_report = report_model_fit(
    "LGBM-p (E4)", 4, "dense [x_correct, x_wrong, bias, delta_days]",
    "p_target",
    {"objective": "regression", "n_estimators": 1000,
     "learning_rate": 0.05, "num_leaves": lgbm_e4_selected_leaves,
     "min_child_samples": lgbm_e4_selected_min_child,
     "best_iteration": int(lgbm_e4_model.best_iteration_),
     "early_stopping_rounds": 50, "random_state": SEED},
    "MSE(raw p prediction, p_target)",
    lgbm_e4_resource, lgbm_e4_report_batch,
)

