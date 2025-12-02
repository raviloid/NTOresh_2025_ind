# АНСАМБЛЬ 3 LGB
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

SEEDS = [1, 1, 1]

COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"

F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"

VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# Параметры
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 2)

BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_STOPPING_ROUNDS = 100
TEMPORAL_SPLIT_RATIO = 0.8


BASE_LGB_PARAMS = {
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
}

print('Загрузка данных')

def load_data():
    train_df = pd.read_csv(f'{INPUT_DIR}/train.csv', parse_dates=[COL_TIMESTAMP])
    test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')
    user_df = pd.read_csv(f'{INPUT_DIR}/users.csv')
    book_df = pd.read_csv(f'{INPUT_DIR}/books.csv')
    book_genres_df = pd.read_csv(f'{INPUT_DIR}/book_genres.csv')
    book_desc_df = pd.read_csv(f'{INPUT_DIR}/book_descriptions.csv')

    # Фильтрация
    train_df = train_df[train_df[COL_HAS_READ] == 1].copy()
    train_df[COL_SOURCE] = VAL_SOURCE_TRAIN
    test_df[COL_SOURCE] = VAL_SOURCE_TEST
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.merge(user_df, on=COL_USER_ID, how="left")
    book_df = book_df.drop_duplicates(subset=[COL_BOOK_ID])
    combined = combined.merge(book_df, on=COL_BOOK_ID, how="left")
    return combined, book_genres_df, book_desc_df

# ================================================
# FEATURE ENGINEERING
# ================================================
def add_basic_aggregates(df, train_df):
    # User
    user_agg = train_df.groupby(COL_USER_ID)[COL_TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [COL_USER_ID, F_USER_MEAN_RATING, F_USER_RATINGS_COUNT]
    # Book
    book_agg = train_df.groupby(COL_BOOK_ID)[COL_TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [COL_BOOK_ID, F_BOOK_MEAN_RATING, F_BOOK_RATINGS_COUNT]
    # Author
    author_agg = train_df.groupby(COL_AUTHOR_ID)[COL_TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [COL_AUTHOR_ID, F_AUTHOR_MEAN_RATING]
    df = df.merge(user_agg, on=COL_USER_ID, how="left")
    df = df.merge(book_agg, on=COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=COL_AUTHOR_ID, how="left")
    return df

def add_genre_features(df, book_genres_df):
    genre_counts = book_genres_df.groupby(COL_BOOK_ID).size().reset_index(name=F_BOOK_GENRES_COUNT)
    main_genre = book_genres_df.groupby(COL_BOOK_ID)[COL_GENRE_ID].first().reset_index()
    main_genre.columns = [COL_BOOK_ID, 'main_genre_id']
    df = df.merge(genre_counts, on=COL_BOOK_ID, how="left")
    df = df.merge(main_genre, on=COL_BOOK_ID, how="left")
    return df

def add_temporal_features(df):
    if COL_TIMESTAMP in df.columns:
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
        df['year'] = df[COL_TIMESTAMP].dt.year
        df['month'] = df[COL_TIMESTAMP].dt.month
        df['day'] = df[COL_TIMESTAMP].dt.day
        df['dayofweek'] = df[COL_TIMESTAMP].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['hour'] = df[COL_TIMESTAMP].dt.hour
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def add_text_features(df, train_df, desc_df):
    # TF-IDF
    vectorizer_path = f"{WORKING_DIR}/tfidf_vectorizer.pkl"
    train_books = train_df[COL_BOOK_ID].unique()
    train_desc = desc_df[desc_df[COL_BOOK_ID].isin(train_books)].copy()
    train_desc[COL_DESCRIPTION] = train_desc[COL_DESCRIPTION].fillna("")

    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE
        )
        vectorizer.fit(train_desc[COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)

    desc_map = dict(zip(desc_df[COL_BOOK_ID], desc_df[COL_DESCRIPTION].fillna("")))
    df_desc = df[COL_BOOK_ID].map(desc_map).fillna("")
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
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        model.to(BERT_DEVICE)
        model.eval()

        desc_clean = desc_df[[COL_BOOK_ID, COL_DESCRIPTION]].copy()
        desc_clean[COL_DESCRIPTION] = desc_clean[COL_DESCRIPTION].fillna("")
        unique_books = desc_clean.drop_duplicates(COL_BOOK_ID)
        book_ids = unique_books[COL_BOOK_ID].values
        descs = unique_books[COL_DESCRIPTION].tolist()

        embeddings_dict = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(descs), BERT_BATCH_SIZE), desc="BERT batches"):
                batch_desc = descs[i:i+BERT_BATCH_SIZE]
                batch_ids = book_ids[i:i+BERT_BATCH_SIZE]
                inputs = tokenizer(batch_desc, padding=True, truncation=True, max_length=BERT_MAX_LENGTH, return_tensors="pt")
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
    bert_array = np.zeros((len(df), BERT_EMBEDDING_DIM))
    for i, bid in enumerate(df[COL_BOOK_ID]):
        if bid in embeddings_dict:
            bert_array[i] = embeddings_dict[bid]

    bert_names = [f"bert_{i}" for i in range(BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(bert_array, columns=bert_names, index=df.index)
    return pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

def handle_missing(df, train_df):
    global_mean = train_df[COL_TARGET].mean() if len(train_df) > 0 else 5.0
    fill_vals = {
        COL_AGE: df[COL_AGE].median() if COL_AGE in df.columns else 30,
        COL_AVG_RATING: global_mean,
        F_USER_MEAN_RATING: global_mean,
        F_BOOK_MEAN_RATING: global_mean,
        F_AUTHOR_MEAN_RATING: global_mean,
        F_USER_RATINGS_COUNT: 0,
        F_BOOK_RATINGS_COUNT: 0,
        F_BOOK_GENRES_COUNT: 0,
    }
    for col, val in fill_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    cat_cols = [COL_GENDER, COL_LANGUAGE, COL_PUBLISHER, 'main_genre_id']
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
    df = add_text_features(df, df[df[COL_SOURCE] == VAL_SOURCE_TRAIN], book_desc_df)
    df = add_bert_features(df, book_desc_df)
    
    # Сохраняем обработанные признаки
    processed_path = f"{WORKING_DIR}/processed_features.parquet"
    df.to_parquet(processed_path, index=False)
    print(f"✅ Признаки сохранены: {processed_path}")
    
    # Разделение
    train_set = df[df[COL_SOURCE] == VAL_SOURCE_TRAIN].copy()
    test_set = df[df[COL_SOURCE] == VAL_SOURCE_TEST].copy()
    
    # Временное разделение
    train_set = train_set.sort_values(COL_TIMESTAMP)
    split_idx = int(len(train_set) * TEMPORAL_SPLIT_RATIO)
    train_split = train_set.iloc[:split_idx].copy()
    val_split = train_set.iloc[split_idx:].copy()
    
    # Агрегаты + обработка пропусков
    train_final = add_basic_aggregates(train_split.copy(), train_split)
    val_final = add_basic_aggregates(val_split.copy(), train_split)
    train_final = handle_missing(train_final, train_split)
    val_final = handle_missing(val_final, train_split)
    
    # Признаки
    exclude_cols = [COL_SOURCE, COL_TARGET, COL_TIMESTAMP, COL_HAS_READ, 'title', 'author_name']
    features = [c for c in train_final.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_final[c])]
    cat_features = [COL_GENDER, COL_LANGUAGE, COL_PUBLISHER, 'main_genre_id']
    for cf in cat_features:
        if cf in train_final.columns and cf not in features:
            features.append(cf)
    
    X_train = train_final[features]
    y_train = train_final[COL_TARGET]
    X_val = val_final[features]
    y_val = val_final[COL_TARGET]
    cat_indices = [features.index(c) for c in cat_features if c in features]
    
    # Обучение ансамбля
    models = []
    weights = []
    
    for seed in SEEDS:
        print(f"\n🌱 Обучение модели с seed = {seed}")
        params = BASE_LGB_PARAMS.copy()
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
        
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        weight = 1.0 / (rmse + 1e-6)
        models.append(model)
        weights.append(weight)
        print(f"   → RMSE: {rmse:.4f}, вес: {weight:.2f}")
    
    # Предсказания на тесте
    test_agg = add_basic_aggregates(test_set.copy(), train_set)
    test_final = handle_missing(test_agg, train_set)
    X_test = test_final[features]
    
    preds = np.zeros(len(X_test))
    for model, w in zip(models, weights):
        preds += w * model.predict(X_test)
    preds /= sum(weights)
    preds = np.clip(preds, 0, 10)
    
    # Сохранение
    submission = test_set[[COL_USER_ID, COL_BOOK_ID]].copy()
    submission[COL_PREDICTION] = preds
    submission.to_csv(f"{WORKING_DIR}/submission.csv", index=False)
    
    print(" ГОТОВО!")

if __name__ == "__main__":
    main()
