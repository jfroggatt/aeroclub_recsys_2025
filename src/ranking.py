import datetime
import gc
import lightgbm as lgb
import polars as pl


def prepare_ranking_data(df):
    # Sort the data to keep flights related to a search (ranker_id) together
    sort_indices = df['ranker_id'].arg_sort()
    X = df[sort_indices]
    y = X['selected'].to_numpy()

    # Get search group sizes (number of flights) for chunked ranking
    group_sizes = (
        X
        .group_by('ranker_id', maintain_order=True)
        .len()
        .select('len')
        .to_numpy()
        .flatten()
    )

    return X.drop(['ranker_id','selected']), y, group_sizes, sort_indices


def train_lgb_ranker(X, y, group_sizes):
    """
    Train LGBRanker with proper group structure
    """
    categorical_features = [col for col in X.columns if not X[col].dtype.is_numeric()]
    X = X.with_columns([pl.col(col).cast(pl.Categorical) for col in categorical_features])

    # Convert to pandas for LightGBM compatibility
    X = X.to_pandas()

    train_data = lgb.Dataset(
        X, label=y,
        group=group_sizes,  # This tells LGB how many samples per search
        categorical_feature=categorical_features
    )

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'num_leaves': 255,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'num_threads': -1,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        # callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        callbacks=[lgb.log_evaluation(100)]
    )

    return model


def train_segment_based_ranker(df):
    segment_models = {}
    segments = df['customer_segment'].unique()

    # Train segment specific models
    for segment in segments:
        segment_df = df.filter(pl.col('customer_segment') == segment)
        print(f"{datetime.datetime.now()}: Training segment {segment} with {len(segment_df)} rows")

        # Prepare Train data for ranking
        X, y, group_sizes, _ = prepare_ranking_data(segment_df)

        # Free up memory
        del segment_df
        gc.collect()

        # Train model
        model = train_lgb_ranker(X, y, group_sizes)

        segment_models[segment] = model

    # Train a model for all segments as a fallback for poorly performing segments
    print(f"{datetime.datetime.now()}: Training on all segments with {len(df)} rows")
    X, y, group_sizes, _ = prepare_ranking_data(df)
    segment_models[99] = train_lgb_ranker(X, y, group_sizes)

    return segment_models
