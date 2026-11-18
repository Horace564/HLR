"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 076 ===
import gc
import threading
import psutil

NN_HIDDEN = 4
NN_BATCH = 8192
NN_EPOCHS = 5
NN_LR = 1e-4
NN_H_WEIGHT = 1e-4
NN_L2 = 1e-6
NN_CLIP_NORM = 10.0
NN_EVAL_BATCH = 200_000

class NeuralHLR:
    """Four-ReLU HLR curve; lexeme rows are looked up rather than one-hot-expanded."""
    def __init__(self, linear_weights, seed=SEED):
        linear_weights = np.asarray(linear_weights, dtype=np.float64)
        self.with_lexeme = len(linear_weights) > 3
        self.w1 = np.zeros((len(linear_weights), NN_HIDDEN), dtype=np.float64)
        self.w1[:, 0] = linear_weights
        self.w1[:, 1] = -linear_weights
        self.w1[:, 2:] = np.random.default_rng(seed).normal(
            0.0, 0.01, size=(len(linear_weights), 2)
        )
        self.w1[2, 2:] = 0.1
        self.w2 = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)

    def state(self):
        return self.w1.copy(), self.w2.copy(), self.b2.copy()

    def restore(self, state):
        self.w1[:], self.w2[:], self.b2[:] = state

    def forward(self, batch):
        base = np.column_stack((
            batch["x_correct"], batch["x_wrong"],
            np.ones(len(batch), dtype=np.float64),
        ))
        pre = base @ self.w1[:3]
        if self.with_lexeme:
            lex = batch["lexeme_column"]
            known = (lex >= 3) & (lex < len(self.w1))
            pre[known] += self.w1[lex[known]]
        hidden = np.maximum(pre, 0.0)
        score = np.clip(hidden @ self.w2 + self.b2[0], LOG_MIN_H, LOG_MAX_H)
        h_pred = np.clip(np.exp2(score), MIN_H, MAX_H)
        p_pred = np.clip(np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P)
        return base, pre, hidden, score, h_pred, p_pred

    def predict(self, batch):
        _, _, _, _, h_pred, p_pred = self.forward(batch)
        return p_pred, h_pred


def nn_adam_update(param, grad, state, step):
    m, v = state
    m *= 0.9
    m += 0.1 * grad
    v *= 0.999
    v += 0.001 * np.square(grad)
    param -= NN_LR * (m / (1.0 - 0.9 ** step)) / (
        np.sqrt(v / (1.0 - 0.999 ** step)) + 1e-8
    )


def nn_train_step(model, batch, states, step):
    base, pre, hidden, score, h_pred, p_pred = model.forward(batch)
    days = batch["delta_days"]
    grad_score = (2.0 / len(batch)) * (
        (p_pred - batch["p_target"]) * p_pred
        * (np.log(2.0) ** 2) * (days / h_pred)
        + NN_H_WEIGHT * (score - np.log2(batch["h_observed"]))
    )
    grad_w2 = hidden.T @ grad_score + 2.0 * NN_L2 * model.w2
    grad_b2 = np.array([grad_score.sum()], dtype=np.float64)
    grad_hidden = grad_score[:, None] * model.w2[None, :] * (pre > 0)
    grad_w1 = np.zeros_like(model.w1)
    grad_w1[:3] = base.T @ grad_hidden
    if model.with_lexeme:
        lex = batch["lexeme_column"]
        known = (lex >= 3) & (lex < len(model.w1))
        lex_rows = lex[known] - 3
        for j in range(NN_HIDDEN):
            grad_w1[3:, j] = np.bincount(
                lex_rows, weights=grad_hidden[known, j],
                minlength=len(model.w1) - 3,
            )
    grad_w1 += 2.0 * NN_L2 * model.w1
    norm = np.sqrt(np.sum(grad_w1 ** 2) + np.sum(grad_w2 ** 2)
                   + np.sum(grad_b2 ** 2))
    if norm > NN_CLIP_NORM:
        factor = NN_CLIP_NORM / norm
        grad_w1 *= factor
        grad_w2 *= factor
        grad_b2 *= factor
    nn_adam_update(model.w1, grad_w1, states[0], step)
    nn_adam_update(model.w2, grad_w2, states[1], step)
    nn_adam_update(model.b2, grad_b2, states[2], step)


def nn_validation_metrics(model):
    start, stop = splits["Validation"]
    abs_sum = sq_sum = h_abs_sum = 0.0
    for left in range(start, stop, NN_EVAL_BATCH):
        right = min(left + NN_EVAL_BATCH, stop)
        batch = records[left:right]
        p_pred, h_pred = model.predict(batch)
        error = p_pred - batch["p_target"]
        abs_sum += float(np.abs(error).sum())
        sq_sum += float(np.dot(error, error))
        h_abs_sum += float(np.abs(h_pred - batch["h_observed"]).sum())
    n = stop - start
    return {"n_validation": n, "MAE(p)": abs_sum / n,
            "RMSE(p)": np.sqrt(sq_sum / n), "MAE(h)_days": h_abs_sum / n}


def nn_validation_predictions(model):
    start, stop = splits["Validation"]
    p_pred = np.empty(stop-start, dtype=np.float64)
    h_pred = np.empty(stop-start, dtype=np.float64)
    for left in range(start, stop, NN_EVAL_BATCH):
        right = min(left + NN_EVAL_BATCH, stop)
        p_batch, h_batch = model.predict(records[left:right])
        p_pred[left-start:right-start] = p_batch
        h_pred[left-start:right-start] = h_batch
    return p_pred, h_pred


def train_neural_hlr(linear_weights, baseline_results, name):
    model = NeuralHLR(linear_weights)
    initial = nn_validation_metrics(model)
    baseline_mae = float(baseline_results.loc[
        baseline_results["split"] == "Validation", "MAE(p)"
    ].iloc[0])
    assert abs(initial["MAE(p)"] - baseline_mae) < 1e-10
    trace = [{"epoch": 0, **initial}]
    best_mae, best_epoch, best_state = initial["MAE(p)"], 0, model.state()
    states = [(np.zeros_like(p), np.zeros_like(p))
              for p in (model.w1, model.w2, model.b2)]
    order = np.arange(train_end, dtype=np.int32)
    rng = np.random.default_rng(SEED + 10)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = [rss_before]
    stop_monitor = threading.Event()

    def watch_rss():
        while not stop_monitor.wait(0.1):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=watch_rss, daemon=True)
    monitor.start()
    started = time.perf_counter()
    step = 0
    try:
        for epoch in range(1, NN_EPOCHS + 1):
            rng.shuffle(order)
            for left in range(0, train_end, NN_BATCH):
                rows = order[left:min(left + NN_BATCH, train_end)]
                step += 1
                nn_train_step(model, records[rows], states, step)
            metrics = nn_validation_metrics(model)
            trace.append({"epoch": epoch, **metrics})
            if metrics["MAE(p)"] < best_mae:
                best_mae, best_epoch, best_state = (
                    metrics["MAE(p)"], epoch, model.state()
                )
    finally:
        stop_monitor.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    fit_seconds = time.perf_counter() - started
    model.restore(best_state)
    final = nn_validation_metrics(model)
    assert np.isclose(final["MAE(p)"], best_mae)
    resource = pd.DataFrame([{
        "model": name, "n_train": train_end, "input_dim": len(model.w1),
        "hidden": NN_HIDDEN, "epochs": NN_EPOCHS,
        "selected_epoch": best_epoch, "batch_size": NN_BATCH,
        "fit_seconds": fit_seconds,
        "rss_before_GiB": rss_before / 2**30,
        "peak_rss_GiB": peak_rss[0] / 2**30,
    }])
    del states, order
    gc.collect()
    return model, pd.DataFrame(trace), resource, pd.DataFrame([{
        "model": name, "n_train": train_end, **final,
    }])


def nn_report_batch(model, batch):
    _, _, _, score, h_pred, p_pred = model.forward(batch)
    p_error = p_pred - batch["p_target"]
    logh_error = score - np.log2(batch["h_observed"])
    loss_sum = (
        np.dot(p_error, p_error)
        + NN_H_WEIGHT * np.dot(logh_error, logh_error)
    )
    return p_pred, float(loss_sum)

def report_neural_hlr_fit(model, name, resource):
    penalty = NN_L2 * (
        float(np.sum(model.w1 ** 2)) + float(np.sum(model.w2 ** 2))
    )
    return report_model_fit(
        name, len(model.w1),
        "sparse HLR inputs; W1 shape " + str(model.w1.shape),
        "p_target and log2(h_observed)",
        {"hidden_units": NN_HIDDEN, "epochs": NN_EPOCHS,
         "selected_epoch": int(resource["selected_epoch"].iloc[0]),
         "batch_size": NN_BATCH, "learning_rate": NN_LR,
         "h_weight": NN_H_WEIGHT, "L2": NN_L2,
         "clip_norm": NN_CLIP_NORM, "seed": SEED},
        "MSE(p) + h_weight*MSE(log2(h)) + L2*(||W1||^2+||W2||^2)",
        resource,
        lambda batch, left, right: nn_report_batch(model, batch),
        regularization_per_row=lambda n: penalty,
    )

# === ORIGINAL CELL 078 ===
nn_hlr_model, nn_hlr_trace, nn_hlr_resource, nn_hlr_validation = train_neural_hlr(
    hlr_weights, hlr_results, "N-HLR"
)
nn_hlr_val_p_pred, nn_hlr_val_h_pred = nn_validation_predictions(nn_hlr_model)
nn_hlr_fit_report = report_neural_hlr_fit(nn_hlr_model, 'N-HLR', nn_hlr_resource)
display(nn_hlr_trace.round(6))

# === ORIGINAL CELL 080 ===
nn_hlr_lexless_model, nn_hlr_lexless_trace, nn_hlr_lexless_resource, nn_hlr_lexless_validation = train_neural_hlr(
    hlr_lexless_weights, hlr_lexless_results, "N-HLR -lex"
)
nn_hlr_lexless_val_p_pred, nn_hlr_lexless_val_h_pred = nn_validation_predictions(
    nn_hlr_lexless_model
)
nn_hlr_lexless_fit_report = report_neural_hlr_fit(nn_hlr_lexless_model, 'N-HLR -lex', nn_hlr_lexless_resource)
display(nn_hlr_lexless_trace.round(6))

