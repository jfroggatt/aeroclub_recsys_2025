import datetime
import numpy as np
import polars as pl
from src.ranking import prepare_ranking_data, train_lgb_ranker


def calculate_hit_rate_at_k(y_true, y_pred, groups, ranker_ids=None, k=3, min_size=None):
    """
    More detailed evaluation with additional metrics
    """
    hits = 0
    total_searches = 0
    hit_details = []
    start_idx = 0

    for i, group_size in enumerate(groups):
        # Skip groups smaller than min_size
        if min_size is not None and group_size < min_size:
            start_idx += group_size
            continue

        group_true = y_true[start_idx:start_idx + group_size]
        group_pred = y_pred[start_idx:start_idx + group_size]

        # Actual selections
        actual_selections = np.where(group_true == 1)[0]
        num_actual_selections = len(actual_selections)

        # Top-k predictions
        top_k_indices = np.argsort(group_pred)[-k:]

        # Calculate hit
        hit = len(set(actual_selections).intersection(set(top_k_indices))) > 0

        if hit:
            hits += 1

        # Store details for analysis
        search_id = ranker_ids[i] if ranker_ids is not None else i
        hit_details.append({
            'search_id': search_id,
            'group_size': group_size,
            'num_actual_selections': num_actual_selections,
            'hit': hit,
            'top_k_scores': group_pred[top_k_indices].tolist(),
            'actual_selections_scores': group_pred[actual_selections].tolist() if len(actual_selections) > 0 else []
        })

        total_searches += 1
        start_idx += group_size

    hit_rate = hits / total_searches

    return hit_rate, hit_details


def test_lgb_ranker(model, df, k=3, min_size=None):
    # Get X, y, and flight search groups
    X_test, y_test, groups_test, sort_indices = prepare_ranking_data(df)

    # Make sure categorical features are cast as categorical
    categorical_features = [col for col in X_test.columns if not X_test[col].dtype.is_numeric()]
    X_test = X_test.with_columns([pl.col(col).cast(pl.Categorical) for col in categorical_features])

    # Convert to pandas for LightGBM compatibility
    X_test = X_test.to_pandas()

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("Calculating Hit Rate @ 3...")
    hit_rate, details = calculate_hit_rate_at_k(
        y_test,
        y_pred,
        groups_test,
        ranker_ids=df['ranker_id'].unique().sort(),
        k=k,
        min_size=min_size
    )
    print(f"Overall Hit Rate @ 3: {hit_rate:.4f}")

    # re-sort y_pred back to original order
    sort_map = np.empty_like(sort_indices)
    sort_map[sort_indices] = np.arange(len(sort_indices))
    y_pred = y_pred[sort_map]

    return hit_rate, details, y_pred


def test_segment_based_ranker(models, df, k=3, min_size=None):
    segment_results = {}
    segments = df['customer_segment'].unique().sort()

    # Get row indices to track original positions
    df_indexed = df.with_row_index('original_index')

    # Initialize an array for predictions
    predictions = [None] * len(df)

    for segment in segments:
        segment_df = df_indexed.filter(pl.col('customer_segment') == segment)
        print(f"\n{datetime.datetime.now()}: Testing segment {segment} with {len(segment_df)} rows")

        # Skip empty segments
        if len(segment_df) == 0:
            continue

        # Save row indices for this segment
        segment_indices = segment_df['original_index'].to_list()

        hit_rate, details, y_pred = test_lgb_ranker(models[segment], segment_df.drop('original_index'), k, min_size)
        segment_results[segment] = (hit_rate, details, y_pred)

        # Map predictions to original positions
        for i, pred in zip(segment_indices, y_pred):
            predictions[i] = pred

    # Add predictions to original dataframe
    df_with_predictions = df_indexed.with_columns(pl.Series('y_pred', predictions)).drop('original_index')

    return segment_results, df_with_predictions
