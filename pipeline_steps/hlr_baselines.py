"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 040 ===
hlr_weights = train_hlr(dimension=3 + vocab_size)
hlr_results = show_model_result("Baseline 1 · HLR", hlr_weights)

# === ORIGINAL CELL 043 ===
hlr_lexless_weights = train_hlr(dimension=3)
hlr_lexless_results = show_model_result("Baseline 2 · HLR -lex", hlr_lexless_weights)

# === ORIGINAL CELL 046 ===
mean_constant_p = float(np.mean(records["p_target"][:train_end]))
assert MIN_P <= mean_constant_p <= MAX_P

def mean_constant_split_row(split_name):
    start, stop = splits[split_name]
    p_true = records["p_target"][start:stop]
    error = mean_constant_p - p_true
    sse = float(np.dot(error, error))
    centered = p_true - float(np.mean(p_true))
    sst = float(np.dot(centered, centered))
    return {
        "model": "Train mean",
        "split": split_name,
        "n_rows": stop - start,
        "MAE(p)": float(np.abs(error).mean()),
        "RMSE(p)": float(np.sqrt(sse / (stop - start))),
        "R²": 1.0 - sse / sst if sst > 0 else np.nan,
    }

mean_constant_baseline = pd.DataFrame([
    mean_constant_split_row("Train"),
    mean_constant_split_row("Validation"),
])
print(f"Train p_target mean = {mean_constant_p:.6f}; constant input X has 0 features.")
print("Selection loss: MSE(p); the prediction is the Train mean.")
display(mean_constant_baseline.round(6))

