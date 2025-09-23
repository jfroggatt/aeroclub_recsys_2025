import polars as pl
import re
from sklearn.model_selection import train_test_split


def camel_to_snake(name):
    """Convert camelCase or PascalCase to snake_case"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def convert_columns_to_snake_case(df):
    """Convert all column names in a polars DataFrame to snake_case"""
    return df.rename({col: camel_to_snake(col) for col in df.columns})


def stratified_flight_count_split(df, test_size=0.2, random_state=42):
    # Calculate flights per search
    search_flight_counts = (
        df
        .group_by('ranker_id')
        .agg(pl.len().alias('flight_count'))
    )

    # Create stratification bins
    search_flight_counts = search_flight_counts.with_columns(
        pl.when(pl.col('flight_count') <= 50).then(pl.lit('very_small'))     # Most searches
        .when(pl.col('flight_count') <= 200).then(pl.lit('small'))
        .when(pl.col('flight_count') <= 500).then(pl.lit('medium'))
        .when(pl.col('flight_count') <= 1000).then(pl.lit('large'))
        .otherwise(pl.lit('very_large'))                                     # High-flight searches
        .alias('size_category')
    )

    # Print distribution for verification
    size_dist = search_flight_counts.group_by('size_category').len().sort('size_category')
    print("Search size distribution:")
    for row in size_dist.iter_rows(named=True):
        print(f"  {row['size_category']}: {row['len']:,} searches")

    # Split within each size category
    train_searches = []
    test_searches = []

    for category in ['very_small', 'small', 'medium', 'large', 'very_large']:
        category_searches = (
            search_flight_counts
            .filter(pl.col('size_category') == category)
            .select('ranker_id')
            .to_pandas()['ranker_id'].tolist()
        )

        if len(category_searches) > 1:
            train_cat, test_cat = train_test_split(
                category_searches,
                test_size=test_size,
                random_state=random_state,
                stratify=None  # No further stratification within category
            )
            train_searches.extend(train_cat)
            test_searches.extend(test_cat)
        elif len(category_searches) == 1:
            # Single search goes to train
            train_searches.extend(category_searches)

    # Filter original data
    train_data = df.filter(pl.col('ranker_id').is_in(train_searches))
    test_data = df.filter(pl.col('ranker_id').is_in(test_searches))

    # Verification: Check distribution preservation
    print(f"\nSplit results:")
    print(f"Train searches: {len(train_searches):,}")
    print(f"Test searches: {len(test_searches):,}")
    print(f"Train rows: {train_data.height:,}")
    print(f"Test rows: {test_data.height:,}")

    # Verify stratification worked
    train_size_dist = (
        train_data
        .group_by('ranker_id')
        .agg(pl.len().alias('flight_count'))
        .with_columns(
            pl.when(pl.col('flight_count') <= 50).then(pl.lit('very_small'))
            .when(pl.col('flight_count') <= 200).then(pl.lit('small'))
            .when(pl.col('flight_count') <= 500).then(pl.lit('medium'))
            .when(pl.col('flight_count') <= 1000).then(pl.lit('large'))
            .otherwise(pl.lit('very_large'))
            .alias('size_category')
        )
        .group_by('size_category')
        .len()
        .sort('size_category')
    )

    test_size_dist = (
        test_data
        .group_by('ranker_id')
        .agg(pl.len().alias('flight_count'))
        .with_columns(
            pl.when(pl.col('flight_count') <= 50).then(pl.lit('very_small'))
            .when(pl.col('flight_count') <= 200).then(pl.lit('small'))
            .when(pl.col('flight_count') <= 500).then(pl.lit('medium'))
            .when(pl.col('flight_count') <= 1000).then(pl.lit('large'))
            .otherwise(pl.lit('very_large'))
            .alias('size_category')
        )
        .group_by('size_category')
        .len()
        .sort('size_category')
    )

    print(f"\nTrain set distribution:")
    for row in train_size_dist.iter_rows(named=True):
        print(f"  {row['size_category']}: {row['len']:,} searches")

    print(f"\nTest set distribution:")
    for row in test_size_dist.iter_rows(named=True):
        print(f"  {row['size_category']}: {row['len']:,} searches")

    return train_data, test_data


def parse_duration_to_minutes(duration_col: str) -> pl.Expr:
    """Parse duration string to minutes (handles format like '2.04:20' - D.HH:MM)."""
    return (
        pl.when(pl.col(duration_col).is_not_null() & (pl.col(duration_col) != ""))
        .then(
            pl.when(pl.col(duration_col).str.contains(r'\.'))
            .then(
                # Format: D.HH:MM:SS (e.g., "1.00:30:00", "2.09:45:00")
                pl.col(duration_col).str.extract(r'^(\d+)\.(\d{2}):(\d{2}):(\d{2})$', 1).cast(pl.Int32, strict=False).fill_null(0) * 1440 +  # Days
                pl.col(duration_col).str.extract(r'^(\d+)\.(\d{2}):(\d{2}):(\d{2})$', 2).cast(pl.Int32, strict=False).fill_null(0) * 60 +   # Hours
                pl.col(duration_col).str.extract(r'^(\d+)\.(\d{2}):(\d{2}):(\d{2})$', 3).cast(pl.Int32, strict=False).fill_null(0) +        # Minutes
                (pl.col(duration_col).str.extract(r'^(\d+)\.(\d{2}):(\d{2}):(\d{2})$', 4).cast(pl.Int32, strict=False).fill_null(0) / 60).round(0).cast(pl.Int32, strict=False)  # Seconds
            )
            .otherwise(
                # Format: HH:MM:SS (e.g., "07:25:00", "17:55:00")
                pl.col(duration_col).str.extract(r'^(\d{2}):(\d{2}):(\d{2})$', 1).cast(pl.Int32, strict=False).fill_null(0) * 60 +   # Hours
                pl.col(duration_col).str.extract(r'^(\d{2}):(\d{2}):(\d{2})$', 2).cast(pl.Int32, strict=False).fill_null(0) +        # Minutes
                (pl.col(duration_col).str.extract(r'^(\d{2}):(\d{2}):(\d{2})$', 3).cast(pl.Int32, strict=False).fill_null(0) / 60).round(0).cast(pl.Int32, strict=False)  # Seconds
            )
        )
        .otherwise(0)
    )