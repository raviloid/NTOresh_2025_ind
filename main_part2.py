import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import gc

WORKING_DIR = "/kaggle/working/"
PROCESSED_PATH = "/kaggle/input/embedd"

SEEDS = [2, 2, 2]

LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.004,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "lambda_l1": 0.2,
    "lambda_l2": 0.3,
    "num_leaves": 128,
    "max_depth": 16,
    "min_data_in_leaf": 30,
    "verbose": -1,
    "n_jobs": -1,
    "boosting_type": "gbdt",
}

EARLY_STOPPING_ROUNDS = 100
TEMPORAL_SPLIT_RATIO = 0.8

COL_SOURCE = "source"
COL_TARGET = "rating"
COL_TIMESTAMP = "timestamp"
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_PREDICTION = "rating_predict"
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

featured_df = pd.read_parquet("/kaggle/input/embedd/processed_features.parquet")
print(f"✅ Загружено: {featured_df.shape}")

train_set = featured_df[featured_df[COL_SOURCE] == VAL_SOURCE_TRAIN].copy()
test_set = featured_df[featured_df[COL_SOURCE] == VAL_SOURCE_TEST].copy()

train_set = train_set.sort_values(COL_TIMESTAMP)
split_idx = int(len(train_set) * TEMPORAL_SPLIT_RATIO)
train_split = train_set.iloc[:split_idx].copy()
val_split = train_set.iloc[split_idx:].copy()

print(f"Train: {len(train_split):,}  //  Val: {len(val_split):,}")

#ДОБАВЛЕНИЕ АГР
def add_aggregates(df, ref_train):
    # User
    user_agg = ref_train.groupby("user_id")["rating"].agg(["mean", "count"]).reset_index()
    user_agg.columns = ["user_id", "user_mean_rating", "user_ratings_count"]
    # Book
    book_agg = ref_train.groupby("book_id")["rating"].agg(["mean", "count"]).reset_index()
    book_agg.columns = ["book_id", "book_mean_rating", "book_ratings_count"]
    # Author
    author_agg = ref_train.groupby("author_id")["rating"].agg(["mean"]).reset_index()
    author_agg.columns = ["author_id", "author_mean_rating"]
    df = df.merge(user_agg, on="user_id", how="left")
    df = df.merge(book_agg, on="book_id", how="left")
    df = df.merge(author_agg, on="author_id", how="left")
    return df

print("📊 Добавление агрегатов...")
train_final = add_aggregates(train_split.copy(), train_split)
val_final = add_aggregates(val_split.copy(), train_split)
test_final = add_aggregates(test_set.copy(), train_set)

#ОБРАБОТКА ПРОПУСКОВ
def fill_missing(df, ref_train):
    global_mean = ref_train["rating"].mean()
    fill_map = {
        "user_mean_rating": global_mean,
        "book_mean_rating": global_mean,
        "author_mean_rating": global_mean,
        "user_ratings_count": 0,
        "book_ratings_count": 0,
        "book_genres_count": 0,
        "age": df["age"].median() if "age" in df.columns else 30,
        "avg_rating": global_mean,
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col in ["gender", "language", "publisher", "main_genre_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("category")

    for col in [c for c in df.columns if c.startswith(("tfidf_", "bert_"))]:
        df[col] = df[col].fillna(0.0)
    return df

print("Обработка пропусков")
train_final = fill_missing(train_final, train_split)
val_final = fill_missing(val_final, train_split)
test_final = fill_missing(test_final, train_set)

#ПРИЗНАКИ
exclude_cols = [COL_SOURCE, COL_TARGET, COL_TIMESTAMP, "has_read", "title", "author_name", "description"]
features = [c for c in train_final.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_final[c])]

cat_features = ["gender", "language", "publisher", "main_genre_id"]
for cf in cat_features:
    if cf in train_final.columns and cf not in features:
        features.append(cf)

cat_indices = [features.index(c) for c in cat_features if c in features]

X_train = train_final[features]
y_train = train_final[COL_TARGET]
X_val = val_final[features]
y_val = val_final[COL_TARGET]
X_test = test_final[features]

print(f"Признаков: {len(features)}")


print("Скоро обучение")
models = []
weights = []

for i, seed in enumerate(SEEDS):
    print(f"\n🌱 Модель {i+1}/{len(SEEDS)} | seed = {seed}")
    params = LGB_PARAMS.copy()
    params["seed"] = seed
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=params["n_estimators"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(200)
        ]
    )
    
    # Валидация
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    weight = 1.0 / (rmse + 1e-6)
    
    models.append(model)
    weights.append(weight)
    print(f"RMSE:{rmse:} Вес: {weight:}")

#Срхранение
print(" Генерация предсказаний")
preds = np.zeros(len(X_test))
for model, w in zip(models, weights):
    preds += w * model.predict(X_test)
preds /= sum(weights)
preds = np.clip(preds, 0, 10)

submission = test_set[[COL_USER_ID, COL_BOOK_ID]].copy()
submission[COL_PREDICTION] = preds
submission.to_csv(f"{WORKING_DIR}/submission_seed_222.csv", index=False)

print('Готово!!!')
