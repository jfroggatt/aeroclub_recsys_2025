import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances


def remove_outliers(df, contamination=0.05):
    # Get numeric columns only
    numeric_cols = [col for col, dtype in df.schema.items() if dtype.is_numeric()]
    df_numeric = df.select(numeric_cols)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(df_numeric.to_pandas())
    mask = outlier_labels == 1

    # Get outlier indices
    outlier_indices = np.where(~mask)[0]

    cleaned_df = df.filter(pl.Series('mask', mask))
    print(f"Removed {len(df) - len(cleaned_df):,} outliers ({(len(df) - len(cleaned_df))/len(df)*100:.1f}%)")

    return cleaned_df, outlier_indices


def encode_features(df, saved_encoders=None, save_encoders=False):
    """
    Encode categorical features with option to save/load encoders
    """
    # Get categorical columns
    categorical_cols = [col for col, dtype in df.schema.items()
                       if not dtype.is_numeric() and col not in ['id', 'ranker_id', 'request_date']]

    encoded_df = df.clone()
    encoders = saved_encoders if saved_encoders else {}

    for col in categorical_cols:
        if col == 'frequent_flyer':
            # Handle string format: "S7/SU/UT" or "S7" or ""
            if save_encoders or 'frequent_flyer' not in encoders:
                # Training phase - determine top programs
                all_programs = []
                for ff_string in df[col].to_list():
                    if ff_string and ff_string != '':
                        programs = ff_string.split('/')
                        all_programs.extend(programs)

                from collections import Counter
                top_programs = Counter(all_programs).most_common(10)
                top_program_names = [prog[0] for prog in top_programs]
                encoders['frequent_flyer'] = {
                    'top_programs': top_program_names,
                    'program_counts': dict(top_programs)
                }

            # Apply encoding using saved/created encoder
            top_programs = encoders['frequent_flyer']['top_programs']

            # Create binary features for top programs
            for program in top_programs:
                encoded_df = encoded_df.with_columns([
                    pl.col(col).map_elements(
                        lambda x: 1 if x and program in str(x).split('/') else 0,
                        return_dtype=pl.Int8
                    ).alias(f'ff_{program}')
                ])

            # Add count of total programs
            encoded_df = encoded_df.with_columns([
                pl.col(col).map_elements(
                    lambda x: len(str(x).split('/')) if x and x != '' else 0,
                    return_dtype=pl.Int8
                ).alias('ff_program_count')
            ])

        else:
            # Handle other categorical columns with frequency encoding
            if save_encoders or col not in encoders:
                # Training phase - create frequency mapping
                value_counts = df[col].value_counts()
                freq_map = {row[0]: row[1] for row in value_counts.rows()}
                encoders[col] = {'freq_map': freq_map}

            # Apply frequency encoding
            freq_map = encoders[col]['freq_map']
            encoded_df = encoded_df.with_columns([
                pl.col(col).map_elements(
                    lambda x: freq_map.get(x, 0),  # Use 0 for unseen values
                    return_dtype=pl.Int32
                ).alias(f'{col}_frequency')
            ])

    # Remove original categorical columns
    final_df = encoded_df.select(pl.exclude(categorical_cols))

    return final_df.fill_null(0), encoders


def generate_clusters(features, n_clusters=10):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(features)

    # Create pseudo-centroids for use in prediction, since AgglomerativeClustering doesn't have centroids
    unique_segments = np.unique(labels)
    centroids = []
    for segment_id in unique_segments:
        segment_mask = labels == segment_id
        centroid = features[segment_mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    score = silhouette_score(features, labels)

    return {
        "labels": labels,
        "centroids": centroids,
        "model": agg,
        "silhouette": score,
        "n_clusters": n_clusters
    }


def gen_optimum_num_clusters(features):
    best_agg_score = -1
    best_agg_n = 0
    best_centroids = None

    for n_clusters in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(features)

        score = silhouette_score(features, labels)
        if score > best_agg_score:
            best_agg_score = score
            best_agg_n = n_clusters

            # Create pseudo-centroids for use in prediction, since AgglomerativeClustering doesn't have centroids
            unique_segments = np.unique(labels)
            centroids = []
            for segment_id in unique_segments:
                segment_mask = labels == segment_id
                centroid = features[segment_mask].mean(axis=0)
                centroids.append(centroid)
            best_centroids = np.array(centroids)

    best_agg = AgglomerativeClustering(n_clusters=best_agg_n, linkage='ward')
    agg_labels = best_agg.fit_predict(features)

    print(f"Best number of clusters: {best_agg_n}, with Silhouette score: {best_agg_score:.4f}")

    return {
        'labels': agg_labels,
        'centroids': best_centroids,
        'model': best_agg,
        'silhouette': silhouette_score(features, agg_labels),
        'n_clusters': len(set(agg_labels))
    }


def predict_cluster_with_centroids(X, centroids):
    """
    Predict clusters by assigning to nearest centroid
    """
    # Calculate distances to all centroids
    distances = euclidean_distances(X, centroids)

    # Assign to nearest centroid
    predictions = np.argmin(distances, axis=1)

    # Get minimum distances as confidence measure
    min_distances = np.min(distances, axis=1)

    return predictions, min_distances


def analyze_clusters(original_df, cluster_labels, outlier_indices=None):
    """Analyze the final clusters"""
    # Add cluster labels to original dataframe for analysis
    if original_df is not None:

        # Check if sizes match
        if len(cluster_labels) != (len(original_df) - (len(outlier_indices) if outlier_indices is not None else 0)):
            print(f"Warning: Cluster labels size ({len(cluster_labels)}) doesn't match DataFrame size ({len(original_df)})")
            print("This likely happened because outliers were removed during clustering and outlier indices not provided or incorrect.")

        else:
            if outlier_indices is not None and len(outlier_indices) > 0:
                mask = ~pl.Series(range(len(original_df))).is_in(outlier_indices)
                analysis_df = original_df.filter(mask)
            else:
                analysis_df = original_df.head(len(cluster_labels))

            # Now add cluster labels safely
            analysis_df = analysis_df.with_columns(pl.Series('cluster', cluster_labels))

            # Cluster profiling
            print("\n📊 CLUSTER PROFILES:")
            print("=" * 50)

            cluster_profiles = analysis_df.group_by('cluster').agg([
                pl.col('total_searches').mean().alias('avg_searches'),
                pl.col('is_vip').mean().alias('vip_rate'),
                pl.col('roundtrip_preference').mean().alias('roundtrip_rate'),
                pl.col('avg_booking_lead_days').mean().alias('avg_lead_days'),
                pl.col('unique_carriers_used').mean().alias('avg_carriers'),
                pl.len().alias('size')
            ]).sort('cluster')

            print(cluster_profiles)