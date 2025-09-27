import polars as pl
import polars.selectors as cs
from src.constants import CUSTOMER_FEATURES, POLARS_INDEX_COL, UNNEEDED_FEATURES, UNNEEDED_FEATURES_REGEX, MAJOR_HUBS
from typing import Dict, List
from src.utils import parse_duration_to_minutes


# function to get column groups for the flight segments features that will be used in engineering the new features
def get_column_groups(df: pl.DataFrame) -> Dict[str, List[str]]:
    columns = df.columns
    return {
        'leg0_duration': [col for col in columns if col.startswith('legs0_segments') and col.endswith('_duration')],
        'leg1_duration': [col for col in columns if col.startswith('legs1_segments') and col.endswith('_duration')],
        'mkt_carrier': [col for col in columns if col.endswith('marketingCarrier_code')],
        'op_carrier': [col for col in columns if col.endswith('operatingCarrier_code')],
        'aircraft': [col for col in columns if col.endswith('aircraft_code')],
        'airport': [col for col in columns if 'airport_iata' in col or 'airport_city_iata' in col],
        'seats_avail': [col for col in columns if col.endswith('seatsAvailable')],
        'baggage_type': [col for col in columns if 'baggageAllowance_weightMeasurementType' in col],
        'baggage_allowance': [col for col in columns if 'baggageAllowance_quantity' in col],
        'cabin_class': [col for col in columns if 'cabinClass' in col]
    }


def create_basic_features(col_groups: Dict[str, List[str]]) -> List[pl.Expr]:
    """Create basic customer and route characteristics features."""
    return [
        pl.col('isAccess3D').cast(pl.Int32).alias('is_access3D'),

        # Normalized frequentFlyer program, addressing null values as null strings, and translating UT program
        # pl.col('frequentFlyer').str.replace('- ЮТэйр ЗАО', 'UT').fill_null('').alias('frequent_flyer'),

        # Route characteristics
        pl.col('legs1_departureAt').is_not_null().cast(pl.Int8).alias('is_roundtrip'),
        pl.col('searchRoute').str.slice(0, 3).alias('route_origin'),

        # Hub features
        pl.col('searchRoute').str.slice(0, 3).is_in(MAJOR_HUBS).cast(pl.Int32).alias('origin_is_major_hub'),
        pl.col('searchRoute').str.slice(3, 3).is_in(MAJOR_HUBS).cast(pl.Int32).alias('destination_is_major_hub'),
        # pl.max_horizontal([
        #     pl.col(col).is_in(MAJOR_HUBS) for col in col_groups['airport']
        # ]).cast(pl.Int32).alias('includes_major_hub'),
    ]


def create_segment_features(col_groups: Dict[str, List[str]]) -> List[pl.Expr]:
    """Create flight segment related features."""
    return [
        # Number of segments per leg
        pl.concat_list([pl.col(col) for col in col_groups['leg0_duration']])
        .list.drop_nulls().list.len().alias('leg0_num_segments'),

        pl.concat_list([pl.col(col) for col in col_groups['leg1_duration']])
        .list.drop_nulls().list.len().alias('leg1_num_segments'),

        # Total segments
        pl.sum_horizontal([pl.col(col).is_not_null() for col in col_groups['aircraft']])
        .alias('total_segments'),

        # Flight time in minutes (sum of segment durations)
        pl.sum_horizontal([
            parse_duration_to_minutes(col)
            for col in col_groups['leg0_duration']
        ]).alias('leg0_flight_time_min'),

        pl.sum_horizontal([
            parse_duration_to_minutes(col)
            for col in col_groups['leg1_duration']
        ]).alias('leg1_flight_time_min'),
    ]


def create_time_features() -> List[pl.Expr]:
    """Create time-based features."""
    return [
        # Booking lead time
        ((pl.col('legs0_departureAt').str.to_datetime() -
            pl.col('requestDate').cast(pl.Datetime)) / pl.duration(days=1)).cast(pl.Int32).alias('booking_lead_days'),

        # Trip duration features
        parse_duration_to_minutes('legs0_duration').alias('leg0_duration_minutes'),
        parse_duration_to_minutes('legs1_duration').alias('leg1_duration_minutes'),
        (parse_duration_to_minutes('legs0_duration') + parse_duration_to_minutes('legs1_duration')).alias('trip_duration_minutes'),

        # Departure/arrival hours
        pl.col('legs0_departureAt').str.to_datetime().dt.hour().alias('leg0_departure_hour'),
        pl.col('legs0_departureAt').str.to_datetime().dt.weekday().alias('leg0_departure_weekday'),
        pl.col('legs0_arrivalAt').str.to_datetime().dt.hour().alias('leg0_arrival_hour'),
        pl.col('legs0_arrivalAt').str.to_datetime().dt.weekday().alias('leg0_arrival_weekday'),
        pl.col('legs1_departureAt').str.to_datetime().dt.hour().alias('leg1_departure_hour'),
        pl.col('legs1_arrivalAt').str.to_datetime().dt.hour().alias('leg1_arrival_hour'),
        pl.col('legs1_departureAt').str.to_datetime().dt.weekday().alias('leg1_departure_weekday'),
        pl.col('legs1_arrivalAt').str.to_datetime().dt.weekday().alias('leg1_arrival_weekday'),
    ]


def create_carrier_features(col_groups: Dict[str, List[str]]) -> List[pl.Expr]:
    """Create carrier and aircraft features."""
    # Create combined carrier column group for carrier features
    all_carrier_cols = col_groups['mkt_carrier'] + col_groups['op_carrier']

    # Create frequent flyer matching expressions
    ff_matches_mkt = [
        pl.when(pl.col(col).is_not_null() & (pl.col(col) != ''))
        .then(pl.col('frequent_flyer').str.contains(pl.col(col)))
        .otherwise(False)
        for col in col_groups['mkt_carrier']
    ]
    ff_matches_op = [
        pl.when(pl.col(col).is_not_null() & (pl.col(col) != ''))
        .then(pl.col('frequent_flyer').str.contains(pl.col(col)))
        .otherwise(False)
        for col in col_groups['op_carrier']
    ]

    return [
        # Carrier features
        pl.concat_list([pl.col(col) for col in all_carrier_cols])
        .list.drop_nulls().list.unique().list.len().alias('unique_carriers'),

        pl.coalesce([pl.col(col) for col in all_carrier_cols]).alias('primary_carrier'),
        pl.col('legs0_segments0_marketingCarrier_code').alias('marketing_carrier'),

        # Frequent flyer matching (check if FF programs match carriers)
        # pl.max_horizontal(ff_matches_mkt).cast(pl.Int32).alias('has_mkt_ff'),
        # pl.max_horizontal(ff_matches_op).cast(pl.Int32).alias('has_op_ff'),

        # Aircraft features
        pl.concat_list([pl.col(col) for col in col_groups['aircraft']])
        .list.drop_nulls().list.unique().list.len().alias('aircraft_diversity'),

        pl.coalesce([pl.col(col) for col in col_groups['aircraft']]).alias('primary_aircraft'),
    ]


def create_service_features(col_groups: Dict[str, List[str]]) -> List[pl.Expr]:
    """Create service-related features (fees, cabin, seats, baggage)."""
    return [
        # Cancellation/exchange fees
        (
            ((pl.col('miniRules0_monetaryAmount') > 0) & pl.col('miniRules0_monetaryAmount').is_not_null()) |
            ((pl.col('miniRules0_percentage') > 0) & pl.col('miniRules0_percentage').is_not_null())
        ).cast(pl.Int32).alias('has_cancellation_fee'),

        (
            ((pl.col('miniRules1_monetaryAmount') > 0) & pl.col('miniRules1_monetaryAmount').is_not_null()) |
            ((pl.col('miniRules1_percentage') > 0) & pl.col('miniRules1_percentage').is_not_null())
        ).cast(pl.Int32).alias('has_exchange_fee'),

        # Cabin class features
        pl.max_horizontal([pl.col(col) for col in col_groups['cabin_class']]).alias('max_cabin_class'),
        pl.mean_horizontal([pl.col(col) for col in col_groups['cabin_class']]).alias('avg_cabin_class'),

        # Seat availability (using for understanding popularity/scarcity)
        pl.min_horizontal([pl.col(col) for col in col_groups['seats_avail']]).alias('min_seats_available'),
        pl.sum_horizontal([pl.col(col).fill_null(0) for col in col_groups['seats_avail']]).alias('total_seats_available'),

        # Baggage allowance
        pl.min_horizontal([
            pl.when(pl.col(type_col) == 0).then(pl.col(qty_col)).otherwise(pl.lit(None))
            for type_col, qty_col in zip(col_groups['baggage_type'], col_groups['baggage_allowance'])
        ]).fill_null(0).alias("baggage_allowance_quantity"),

        pl.min_horizontal([
            pl.when(pl.col(type_col) == 1).then(pl.col(qty_col)).otherwise(pl.lit(None))
            for type_col, qty_col in zip(col_groups['baggage_type'], col_groups['baggage_allowance'])
        ]).fill_null(0).alias("baggage_allowance_weight")
    ]


def create_window_based_flight_features(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    return lazy_df.with_columns([
        # calculate price percentile over search session
        ((pl.col('totalPrice').rank(method='min').over('ranker_id') - 1) /
        (pl.col('totalPrice').count().over('ranker_id') - 1) * 100)
        .fill_null(50.0).alias('price_percentile'),

        # Price features (always relative to search session)
        (pl.col('totalPrice').rank(method='ordinal').over('ranker_id') /
         pl.col('totalPrice').count().over('ranker_id')).alias('price_rank_pct'),
        (pl.col('totalPrice') / pl.col('totalPrice').min().over('ranker_id')).alias('price_ratio_to_min'),
    ]).with_columns([

        # Route popularity features
        pl.len().over('searchRoute').alias('route_popularity')
    ]).with_columns([
        pl.col('route_popularity').log1p().alias('route_popularity_log')
    ])


def create_derived_flight_features() -> List[pl.Expr]:
    return [
        pl.col('leg0_departure_hour').is_between(6, 22).cast(pl.Int8).alias('is_daytime'),
        (pl.col('leg0_departure_weekday') >= 5).cast(pl.Int8).alias('is_weekend'),
        (pl.col('leg0_departure_hour').is_between(6, 22) &
         (~pl.col('leg0_arrival_hour').is_between(6, 22)))
        .cast(pl.Int8).alias('is_redeye'),
        (pl.col('total_segments') > 1).cast(pl.Int8).alias('has_connections'),
    ]

def extract_flight_features(df: pl.DataFrame) -> pl.DataFrame:
    """ Create flight-level features"""
    # Check if already processed
    if df.height > 0 and 'total_duration_mins' in df.columns:
        return df


    # Get column groups
    col_groups = get_column_groups(df)

    # Start with lazy frame, dropping unnecessary columns
    lazy_df = df.drop(CUSTOMER_FEATURES + POLARS_INDEX_COL, strict=False).lazy()
    # lazy_df = create_window_based_flight_features(lazy_df)

    # Apply all feature groups
    lazy_df = lazy_df.with_columns([
        *create_basic_features(col_groups),
        *create_segment_features(col_groups),
        *create_time_features(),
        *create_service_features(col_groups)
    ])

    # Add carrier features (requires frequent_flyer from create_basic_features() step)
    lazy_df = lazy_df.with_columns(create_carrier_features(col_groups))

    # Add derived features
    lazy_df = lazy_df.with_columns(create_derived_flight_features())

    # Add window based derived flight features
    lazy_df = create_window_based_flight_features(lazy_df)

    # Materialize to generate new features, drop unused features, and fill nulls
    return (lazy_df
            .collect()
            .select(~cs.matches(UNNEEDED_FEATURES_REGEX) & ~cs.by_name(UNNEEDED_FEATURES))
            .fill_null(0)
            )


def interactive_features()  -> List[pl.Expr]:
    return [
        (pl.col('is_daytime') * (1.0 - pl.col('night_flight_preference').fill_null(0.5))).alias('daytime_alignment'),
        (pl.col('is_weekend') * pl.col('weekend_travel_rate').fill_null(0.5)).alias('weekend_alignment'),
        (pl.col('leg0_departure_weekday') == pl.col('weekday_preference')).cast(pl.Int8)
            .alias('departure_weekday_match'),
        (pl.col('leg1_departure_weekday') == pl.col('return_weekday_preference')).cast(pl.Int8)
            .alias('return_departure_weekday_match'),
        (pl.col('is_redeye') * pl.col('redeye_flight_preference').fill_null(0.5)).alias('redeye_alignment'),
        (pl.col('price_rank_pct') * pl.col('price_position_preference').fill_null(0.5)).alias('price_preference_match'),
        (pl.col('primary_carrier') == pl.col('most_common_carrier').fill_null('unknown')).cast(pl.Int8).alias('carrier_loyalty_match'),
        pl.col('frequent_flyer')
            .str.contains(pl.col('marketing_carrier').fill_null(False))
            .cast(pl.Int8).alias('carrier_ff_match')
    ]

def prepare_combined_data(flight_features, cust_features):
    if not isinstance(cust_features, pl.LazyFrame):
        cust_features = cust_features.lazy()

    # Join the customer features with the flight features, removing outlier profiles
    flight_features = (
        flight_features.lazy()
        .join(cust_features, on='profileId', how='inner').drop(pl.col('profileId'))
        .with_columns([
            *interactive_features(),  # Additional interactive features between customer and flight search data sets
        ])
        .collect()
    )

    return flight_features
