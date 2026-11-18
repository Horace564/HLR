"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 048 ===
def hlr_extended_layout(use_user, use_languages):
    offset = 3 + vocab_size
    layout = {
        "lex_end": offset,
        "user_start": None,
        "learning_start": None,
        "ui_start": None,
    }
    if use_user:
        layout["user_start"] = offset
        offset += n_users
    if use_languages:
        layout["learning_start"] = offset
        offset += n_learning_languages
        layout["ui_start"] = offset
        offset += n_ui_languages
    layout["dimension"] = offset
    return layout


def train_hlr_extended(name, layout):
    weights = [0.0] * layout["dimension"]
    feature_counts = [0] * layout["dimension"]
    order = np.arange(train_end, dtype=np.int32)
    np.random.default_rng(SEED).shuffle(order)
    x0_all, x1_all = records["x_correct"], records["x_wrong"]
    lex_all, p_all = records["lexeme_column"], records["p_target"]
    days_all, h_all = records["delta_days"], records["h_observed"]
    user_all = categorical_records["user_index"]
    learning_all = categorical_records["learning_index"]
    ui_all = categorical_records["ui_index"]
    started = time.perf_counter()
    for step, row_index in enumerate(order, start=1):
        x0, x1 = float(x0_all[row_index]), float(x1_all[row_index])
        lex_col = int(lex_all[row_index])
        p = float(p_all[row_index])
        days = float(days_all[row_index])
        h_true = float(h_all[row_index])
        score = weights[0] * x0 + weights[1] * x1 + weights[2]
        active = ((0, x0), (1, x1), (2, 1.0))
        if 3 <= lex_col < layout["lex_end"]:
            score += weights[lex_col]
            active += ((lex_col, 1.0),)
        if layout["user_start"] is not None:
            user_index = int(user_all[row_index])
            if user_index >= 0:
                column = layout["user_start"] + user_index
                score += weights[column]
                active += ((column, 1.0),)
        if layout["learning_start"] is not None:
            learning_index = int(learning_all[row_index])
            ui_index = int(ui_all[row_index])
            if learning_index >= 0:
                column = layout["learning_start"] + learning_index
                score += weights[column]
                active += ((column, 1.0),)
            if ui_index >= 0:
                column = layout["ui_start"] + ui_index
                score += weights[column]
                active += ((column, 1.0),)
        h_pred = math.exp2(min(max(score, LOG_MIN_H), LOG_MAX_H))
        p_pred = min(max(math.exp2(-days / h_pred), MIN_P), MAX_P)
        grad_p = (
            2.0 * (p_pred - p) * math.log(2.0)**2
            * p_pred * (days / h_pred)
        )
        grad_h = 2.0 * (h_pred - h_true) * math.log(2.0) * h_pred
        for column, value in active:
            rate = LEARNING_RATE / (
                (1.0 + p) * math.sqrt(1 + feature_counts[column])
            )
            weights[column] -= rate * grad_p * value
            weights[column] -= rate * ALPHA * grad_h * value
            weights[column] -= rate * L2 * weights[column]
            feature_counts[column] += 1
        if step % 1_000_000 == 0 or step == train_end:
            print(
                f"{name}: {step:,}/{train_end:,} Train rows; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return np.asarray(weights, dtype=np.float64), time.perf_counter() - started


def predict_hlr_extended_batch(weights, layout, batch, categories):
    assert len(batch) == len(categories)
    score = (
        weights[0] * batch["x_correct"]
        + weights[1] * batch["x_wrong"] + weights[2]
    )
    lexeme = batch["lexeme_column"]
    known_lexeme = (lexeme >= 3) & (lexeme < layout["lex_end"])
    score[known_lexeme] += weights[lexeme[known_lexeme]]
    if layout["user_start"] is not None:
        user_index = categories["user_index"]
        known_user = user_index >= 0
        score[known_user] += weights[
            layout["user_start"] + user_index[known_user]
        ]
    if layout["learning_start"] is not None:
        for field, start in (
            ("learning_index", layout["learning_start"]),
            ("ui_index", layout["ui_start"]),
        ):
            index = categories[field]
            known = index >= 0
            score[known] += weights[start + index[known].astype(np.int32)]
    h_pred = np.exp2(np.clip(score, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(
        np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P
    )
    return p_pred, h_pred


def evaluate_hlr_extended_validation(name, weights, layout):
    start, stop = splits["Validation"]
    count = stop - start
    p_true = records["p_target"][start:stop]
    h_true = records["h_observed"][start:stop]
    days = records["delta_days"][start:stop]
    p_pred = np.empty(count, dtype=np.float64)
    h_pred = np.empty(count, dtype=np.float64)
    for left in range(start, stop, 200_000):
        right = min(left + 200_000, stop)
        p_batch, h_batch = predict_hlr_extended_batch(
            weights, layout, records[left:right], categorical_records[left:right]
        )
        p_pred[left-start:right-start] = p_batch
        h_pred[left-start:right-start] = h_batch
    error = p_pred - p_true
    sse = float(np.dot(error, error))
    centered = p_true - float(np.mean(p_true))
    sst = float(np.dot(centered, centered))
    interior = (
        (h_true > MIN_H) & (h_true < MAX_H) & (days > 0)
    )
    n_interior = int(interior.sum())
    row = {
        "model": name,
        "split": "Validation",
        "n_train": train_end,
        "n_eval": count,
        "MAE(p)": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(sse / count)),
        "R²": 1.0 - sse / sst if sst > 0 else np.nan,
        "AUC": auc_fixed(np.rint(p_true), p_pred),
        "Spearman(h)": spearman_with_ties(h_true, h_pred),
        "n_spearman_h": count,
        "n_h_interior": n_interior,
        "h_interior_share": n_interior / count,
        "Spearman(h)_interior": (
            spearman_with_ties(h_true[interior], h_pred[interior])
            if n_interior > 1 else np.nan
        ),
    }
    abs_error = np.abs(error)
    language_index = categorical_records["learning_index"][start:stop]
    language_rows = []
    for index, language in enumerate(category_manifest["learning_values"]):
        mask = language_index == index
        count_in_group = int(mask.sum())
        language_rows.append({
            "learning_language": language,
            "rows": count_in_group,
            "MAE(p)": (
                float(abs_error[mask].mean()) if count_in_group else np.nan
            ),
        })
    unknown_language = language_index < 0
    if unknown_language.any():
        language_rows.append({
            "learning_language": "unseen",
            "rows": int(unknown_language.sum()),
            "MAE(p)": float(abs_error[unknown_language].mean()),
        })
    by_language = pd.DataFrame(language_rows)
    assert int(by_language["rows"].sum()) == count
    user_index = categorical_records["user_index"][start:stop]
    by_user = pd.DataFrame([
        {
            "user_status": status,
            "rows": int(np.count_nonzero(mask)),
            "MAE(p)": float(abs_error[mask].mean()) if mask.any() else np.nan,
        }
        for status, mask in (
            ("known", user_index >= 0),
            ("unseen", user_index < 0),
        )
    ])
    return pd.DataFrame([row]), by_language, by_user


def run_hlr_extended(name, use_user, use_languages):
    layout = hlr_extended_layout(use_user, use_languages)
    weights, fit_seconds = train_hlr_extended(name, layout)
    validation, by_language, by_user = evaluate_hlr_extended_validation(
        name, weights, layout
    )
    resources = pd.DataFrame([{
        "model": name,
        "n_train": train_end,
        "dimension": layout["dimension"],
        "fit_seconds": fit_seconds,
        "user_features": n_users if use_user else 0,
        "language_features": (
            n_learning_languages + n_ui_languages if use_languages else 0
        ),
    }])
    def predict_and_loss(batch, left, right):
        p_pred, h_pred = predict_hlr_extended_batch(
            weights, layout, batch, categorical_records[left:right]
        )
        p_error = p_pred - batch["p_target"]
        h_error = h_pred - batch["h_observed"]
        loss_sum = float(
            np.dot(p_error, p_error) + ALPHA * np.dot(h_error, h_error)
        )
        return p_pred, loss_sum

    report_model_fit(
        name, layout["dimension"], "sparse HLR base + lexeme + selected category one-hot",
        "p_target and h_observed",
        {"alpha": ALPHA, "L2": L2, "learning_rate": LEARNING_RATE,
         "epochs": EPOCHS, "seed": SEED},
        "sum[(p-p_hat)^2 + alpha*(h-h_hat)^2] + L2*||weights||^2; shown per row",
        resources, predict_and_loss,
        regularization_per_row=lambda n: L2 * float(np.dot(weights, weights)) / n,
    )
    return weights, layout, resources, validation, by_language, by_user


def report_model_fit(name, input_dim, input_description, target_description,
                     hyperparameters, loss_description, resource,
                     predict_and_loss_batch, regularization_per_row=None):
    """Report the same Train/Validation fields without touching Test."""
    print(f"Model: {name}")
    print(f"Input X shape: Train ({train_end}, {input_dim}); "
          f"Validation ({splits['Validation'][1]-splits['Validation'][0]}, {input_dim})")
    print(f"Input representation: {input_description}")
    print(f"Target: {target_description}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Loss: {loss_description}")
    rows = []
    for split_name in ("Train", "Validation"):
        start, stop = splits[split_name]
        n = stop - start
        abs_sum = sq_sum = target_sum = target_sq_sum = loss_sum = 0.0
        for left in range(start, stop, 200_000):
            right = min(left + 200_000, stop)
            batch = records[left:right]
            p_pred, batch_loss = predict_and_loss_batch(batch, left, right)
            p_true = batch["p_target"]
            error = p_pred - p_true
            abs_sum += float(np.abs(error).sum())
            sq_sum += float(np.dot(error, error))
            target_sum += float(p_true.sum())
            target_sq_sum += float(np.dot(p_true, p_true))
            loss_sum += float(batch_loss)
        sst = target_sq_sum - target_sum ** 2 / n
        penalty = (
            float(regularization_per_row(n))
            if regularization_per_row is not None else 0.0
        )
        rows.append({
            "model": name, "split": split_name, "n_rows": n,
            "Loss/row": loss_sum / n + penalty,
            "MAE(p)": abs_sum / n,
            "RMSE(p)": math.sqrt(sq_sum / n),
            "R²": 1.0 - sq_sum / sst if sst > 0 else np.nan,
        })
    report = pd.DataFrame(rows)
    print("Fit resources:")
    display(resource.round(3))
    print("Train and Validation (Test is reported in Section 8):")
    display(report.round(6))
    return report

# === ORIGINAL CELL 050 ===
hlr_user_weights, hlr_user_layout, hlr_user_resource, hlr_user_validation, hlr_user_by_language, hlr_user_by_user = run_hlr_extended(
    "HLR-User", use_user=True, use_languages=False
)

# === ORIGINAL CELL 052 ===
hlr_lang_weights, hlr_lang_layout, hlr_lang_resource, hlr_lang_validation, hlr_lang_by_language, hlr_lang_by_user = run_hlr_extended(
    "HLR-Lang", use_user=False, use_languages=True
)

# === ORIGINAL CELL 054 ===
hlr_both_weights, hlr_both_layout, hlr_both_resource, hlr_both_validation, hlr_both_by_language, hlr_both_by_user = run_hlr_extended(
    "HLR-User+Lang", use_user=True, use_languages=True
)

