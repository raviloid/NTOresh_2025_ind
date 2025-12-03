import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import os
import gc
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "/kaggle/input/dataset"
WORKING_DIR = "/kaggle/working"

print('Загрузка данных')

def load_data():
    train_df = pd.read_csv(f'{INPUT_DIR}/train.csv', parse_dates=["timestamp"])
    test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')
    user_df = pd.read_csv(f'{INPUT_DIR}/users.csv')
    book_df = pd.read_csv(f'{INPUT_DIR}/books.csv')
    book_genres_df = pd.read_csv(f'{INPUT_DIR}/book_genres.csv')
    book_desc_df = pd.read_csv(f'{INPUT_DIR}/book_descriptions.csv')

    # Фильтрация
    train_df = train_df[train_df["has_read"] == 1].copy()
    train_df["source"] = "train"
    test_df["source"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.merge(user_df, on="user_id", how="left")
    book_df = book_df.drop_duplicates(subset=["book_id"])
    combined = combined.merge(book_df, on="book_id", how="left")
    return combined, book_genres_df, book_desc_df

# ================================================
# FEATURE ENGINEERING
# ================================================
def add_basic_aggregates(df, train_df):
    # User
    user_agg = train_df.groupby("user_id")["rating"].agg(["mean", "count"]).reset_index()
    user_agg.columns = ["user_id", "user_mean_rating", "user_ratings_count"]
    # Book
    book_agg = train_df.groupby("book_id")["rating"].agg(["mean", "count"]).reset_index()
    book_agg.columns = ["book_id", "book_mean_rating", "book_ratings_count"]
    # Author
    author_agg = train_df.groupby("author_id")["rating"].agg(["mean"]).reset_index()
    author_agg.columns = ["author_id", "author_mean_rating"]
    df = df.merge(user_agg, on="user_id", how="left")
    df = df.merge(book_agg, on="book_id", how="left")
    df = df.merge(author_agg, on="author_id", how="left")
    return df

def add_genre_features(df, book_genres_df):
    genre_counts = book_genres_df.groupby("book_id").size().reset_index(name="book_genres_count")
    main_genre = book_genres_df.groupby("book_id")["genre_id"].first().reset_index()
    main_genre.columns = ["book_id", "main_genre_id"]
    df = df.merge(genre_counts, on="book_id", how="left")
    df = df.merge(main_genre, on="book_id", how="left")
    return df

def add_temporal_features(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["hour"] = df["timestamp"].dt.hour
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def add_text_features(df, train_df, desc_df):
    # TF-IDF
    vectorizer_path = f"{WORKING_DIR}/tfidf_vectorizer.pkl"
    train_books = train_df["book_id"].unique()
    train_desc = desc_df[desc_df["book_id"].isin(train_books)].copy()
    train_desc["description"] = train_desc["description"].fillna("")

    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        vectorizer.fit(train_desc["description"])
        joblib.dump(vectorizer, vectorizer_path)

    desc_map = dict(zip(desc_df["book_id"], desc_df["description"].fillna("")))
    df_desc = df["book_id"].map(desc_map).fillna("")
    tfidf_mat = vectorizer.transform(df_desc)
    tfidf_names = [f"tfidf_{i}" for i in range(tfidf_mat.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=tfidf_names, index=df.index)
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

def add_bert_features(df, desc_df):
    embeddings_path = f"{WORKING_DIR}/bert_embeddings.pkl"
    
    if os.path.exists(embeddings_path):
        print("📂 Загружаем кэшированные BERT-эмбеддинги...")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("🤖 Вычисляем BERT-эмбеддинги...")
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        BERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(BERT_DEVICE)
        model.eval()

        desc_clean = desc_df[["book_id", "description"]].copy()
        desc_clean["description"] = desc_clean["description"].fillna("")
        unique_books = desc_clean.drop_duplicates("book_id")
        book_ids = unique_books["book_id"].values
        descs = unique_books["description"].tolist()

        embeddings_dict = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(descs), 8), desc="BERT batches"):
                batch_desc = descs[i:i+8]
                batch_ids = book_ids[i:i+8]
                inputs = tokenizer(batch_desc, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(BERT_DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                for bid, e in zip(batch_ids, emb):
                    embeddings_dict[bid] = e

        joblib.dump(embeddings_dict, embeddings_path)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Создаём признаки
    bert_array = np.zeros((len(df), 768))
    for i, bid in enumerate(df["book_id"]):
        if bid in embeddings_dict:
            bert_array[i] = embeddings_dict[bid]

    bert_names = [f"bert_{i}" for i in range(768)]
    bert_df = pd.DataFrame(bert_array, columns=bert_names, index=df.index)
    return pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

def handle_missing(df, train_df):
    global_mean = train_df["rating"].mean() if len(train_df) > 0 else 5.0
    fill_vals = {
        "age": df["age"].median() if "age" in df.columns else 30,
        "avg_rating": global_mean,
        "user_mean_rating": global_mean,
        "book_mean_rating": global_mean,
        "author_mean_rating": global_mean,
        "user_ratings_count": 0,
        "book_ratings_count": 0,
        "book_genres_count": 0,
    }
    for col, val in fill_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    cat_cols = ["gender", "language", "publisher", "main_genre_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype('category')
    
    for col in [c for c in df.columns if c.startswith(("tfidf_", "bert_"))]:
        df[col] = df[col].fillna(0.0)
    return df

# ================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ================================================
def main():
    print("📥 Загрузка данных...")
    df, book_genres_df, book_desc_df = load_data()
    
    print("🔧 Feature engineering (без агрегатов)...")
    df = add_genre_features(df, book_genres_df)
    df = add_temporal_features(df)
    df = add_text_features(df, df[df["source"] == "train"], book_desc_df)
    df = add_bert_features(df, book_desc_df)
    
    # Сохраняем обработанные признаки
    processed_path = f"{WORKING_DIR}/processed_features.parquet"
    df.to_parquet(processed_path, index=False)
    print(f"✅ Признаки сохранены: {processed_path}")
    
    # Разделение
    train_set = df[df["source"] == "train"].copy()
    test_set = df[df["source"] == "test"].copy()
    
    # Временное разделение
    train_set = train_set.sort_values("timestamp")
    split_idx = int(len(train_set) * 0.8)
    train_split = train_set.iloc[:split_idx].copy()
    val_split = train_set.iloc[split_idx:].copy()
    
    # Агрегаты + обработка пропусков
    train_final = add_basic_aggregates(train_split.copy(), train_split)
    val_final = add_basic_aggregates(val_split.copy(), train_split)
    train_final = handle_missing(train_final, train_split)
    val_final = handle_missing(val_final, train_split)
    
    # Признаки
    exclude_cols = ["source", "rating", "timestamp", "has_read", "title", "author_name"]
    features = [c for c in train_final.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_final[c])]
    cat_features = ["gender", "language", "publisher", "main_genre_id"]
    for cf in cat_features:
        if cf in train_final.columns and cf not in features:
            features.append(cf)
    
    X_train = train_final[features]
    y_train = train_final["rating"]
    X_val = val_final[features]
    y_val = val_final["rating"]
    cat_indices = [features.index(c) for c in cat_features if c in features]
    
    # Обучение одной модели
    print(f"\n Обучение модели с seed")
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "n_estimators": 3000,
        "learning_rate": 0.005,
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
        "seed": 2
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=params["n_estimators"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(200)
        ]
    )
    
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"   → RMSE: {rmse:.4f}")
    
    # Предсказания на тесте
    test_agg = add_basic_aggregates(test_set.copy(), train_set)
    test_final = handle_missing(test_agg, train_set)
    X_test = test_final[features]
    
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 10)
    
    # Сохранение
    submission = test_set[["user_id", "book_id"]].copy()
    submission["rating_predict"] = preds
    submission.to_csv(f"{WORKING_DIR}/submission.csv", index=False)
    
    print(" ГОТОВО!")

if __name__ == "__main__":
    main()
