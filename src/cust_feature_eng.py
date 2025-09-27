import polars as pl
from src.constants import MAJOR_HUBS
from src.utils import parse_duration_to_minutes
from typing import List


def create_customer_aggregation_features() -> List[pl.Expr]:
    """Create customer aggregation expressions for basic attributes and search behavior."""
    return [
        # Basic customer attributes (take first non-null value per customer, since they should be the same)
        pl.col('companyID').drop_nulls().first().alias('company_id'),
        pl.col('sex').drop_nulls().first().cast(pl.Int8).alias('sex'),
        pl.col('nationality').drop_nulls().first().alias('nationality'),
        pl.col('frequentFlyer').drop_nulls().first().str.replace('- ЮТэйр ЗАО', 'UT').fill_null('')
            .alias('frequent_flyer'),
        pl.col('isVip').drop_nulls().first().cast(pl.Int8).alias('is_vip'),
        pl.col('bySelf').drop_nulls().first().cast(pl.Int8).alias('by_self'),
        pl.col('corporateTariffCode').is_not_null().cast(pl.Int8).max().alias('has_corp_codes'),

        # Search behavior metrics
        pl.len().alias('total_flights_searched'),
        pl.col('ranker_id').n_unique().alias('total_search_sessions'),
        (pl.len() / pl.col('ranker_id').n_unique()).alias('avg_searches_per_session'),
        pl.col('searchRoute').drop_nulls().n_unique().alias('unique_routes_searched'),
    ]


def create_booking_lead_time_features() -> List[pl.Expr]:
    """Create booking lead time statistics."""
    # Calculate booking lead time in days
    booking_lead_expr = (
        (pl.col('legs0_departureAt').str.to_datetime() -
         pl.col('requestDate').cast(pl.Datetime)) / pl.duration(days=1)
    ).filter(pl.col('selected') == 1).cast(pl.Int32)

    return [
        booking_lead_expr.min().alias('min_booking_lead_days'),
        booking_lead_expr.max().alias('max_booking_lead_days'),
        booking_lead_expr.mean().alias('avg_booking_lead_days'),
        booking_lead_expr.median().alias('median_booking_lead_days'),
    ]


def create_travel_preference_features() -> List[pl.Expr]:
    """Create travel preference features for selected flights."""
    return [
        # departure airport
        pl.col('legs0_segments0_departureFrom_airport_iata').filter(pl.col('selected') == 1)
            .drop_nulls().mode().first().alias('most_common_departure_airport'),
        pl.col('legs0_segments0_departureFrom_airport_iata').filter(pl.col('selected') == 1)
            .drop_nulls().n_unique().alias('unique_departure_airports'),

        # marketing carrier
        pl.col('legs0_segments0_marketingCarrier_code').filter(pl.col('selected') == 1)
            .drop_nulls().mode().first().alias('most_common_carrier'),
        pl.col('legs0_segments0_marketingCarrier_code').filter(pl.col('selected') == 1)
            .drop_nulls().n_unique().alias('unique_carriers_used'),

        # Round trip preference (noted by legs1* not null)
        pl.col('legs1_departureAt').filter(pl.col('selected') == 1).is_not_null().mean().alias('roundtrip_preference'),
    ]


def create_cabin_class_features(cabin_class_cols: List[str]) -> List[pl.Expr]:
    """Create cabin class preference statistics for selected flights."""
    return [
        # Cabin class statistics across all segments
        pl.min_horizontal([pl.col(col) for col in cabin_class_cols]).filter(pl.col('selected') == 1)
            .min().alias('min_cabin_class'),
        pl.max_horizontal([pl.col(col) for col in cabin_class_cols]).filter(pl.col('selected') == 1)
            .max().alias('max_cabin_class'),
        pl.mean_horizontal([pl.col(col) for col in cabin_class_cols]).filter(pl.col('selected') == 1)
            .mean().alias('avg_cabin_class'),
    ]


def create_temporal_preference_features() -> List[pl.Expr]:
    """Create temporal preference features for departure patterns for selected flights."""
    return [
        # Weekday preference (most common day of week for departures)
        pl.col('legs0_departureAt').filter(pl.col('selected') == 1).str.to_datetime().dt.weekday()
            .mode().first().alias('weekday_preference'),

        # Return weekday preference (most common day of week for return flights), use -1 when no return flights
        pl.col('legs1_departureAt').filter((pl.col('selected') == 1 ) & (pl.col('legs1_departureAt').is_not_null()))
            .str.to_datetime().dt.weekday()
            .mode().first().fill_null(-1).alias('return_weekday_preference'),

        # Weekend travel rate (percentage of weekend departures - 5=Sat, 6=Sun)
        pl.col('legs0_departureAt').filter(pl.col('selected') == 1)
            .str.to_datetime().dt.weekday()
            .map_elements(lambda x: 1 if x >= 5 else 0, return_dtype=pl.Int8)
            .mean().alias('weekend_travel_rate'),

        # Return weekend travel rate (percentage of weekend departures - 5=Sat, 6=Sun)
        pl.col('legs1_departureAt').filter((pl.col('selected') == 1) & (pl.col('legs1_departureAt').is_not_null()))
            .str.to_datetime().dt.weekday()
            .map_elements(lambda x: 1 if x >= 5 else 0, return_dtype=pl.Int8)
            .mean().alias('return_weekend_travel_rate'),

        # Time of day variance (how consistent are departure times)
        pl.col('legs0_departureAt').filter(pl.col('selected') == 1).str.to_datetime().dt.hour()
            .std().alias('time_of_day_variance'),

        # Night flight preference (flights departing 22:00-06:00)
        pl.col('legs0_departureAt').filter(pl.col('selected') == 1).str.to_datetime().dt.hour()
            .map_elements(lambda x: 1 if (x >= 22 or x < 6) else 0, return_dtype=pl.Int8)
            .mean().alias('night_flight_preference'),

        # Redeye flight preference (overnight flights - depart late, arrive early)
        pl.when(pl.col('selected') == 1)
            .then(
                (pl.col('legs0_departureAt').str.to_datetime().dt.hour() >= 22) |
                (pl.col('legs0_departureAt').str.to_datetime().dt.hour() <= 5) |
                ((pl.col('legs0_departureAt').str.to_datetime().dt.hour() >= 18) &
                 (pl.col('legs0_arrivalAt').str.to_datetime().dt.hour() <= 8))
            )
            .otherwise(None).cast(pl.Int8).mean()
            .alias('redeye_flight_preference'),
]


def create_route_specific_features() -> List[pl.Expr]:
    """Create features related to route preferences and characteristics."""
    return [
        # Route loyalty
        (1 - (pl.col('searchRoute').n_unique() / pl.len().clip(1, None))).alias('route_loyalty'),

        # Hub preference (preference for major hub airports)
        (
            pl.col('legs0_segments0_departureFrom_airport_iata').filter(pl.col('selected') == 1).is_in(MAJOR_HUBS) |
            pl.coalesce([
                pl.col('legs0_segments2_arrivalTo_airport_iata'),
                pl.col('legs0_segments1_arrivalTo_airport_iata'),
                pl.col('legs0_segments0_arrivalTo_airport_iata')
            ]).filter(pl.col('selected') == 1).is_in(MAJOR_HUBS)
        ).cast(pl.Int8).mean().alias('hub_preference'),

        # Short haul preference
        ((1 - (pl.col('leg0_duration_minutes').filter(pl.col('selected') == 1).mean() / 180)) * 0.7 +
            pl.when(pl.col('leg0_duration_minutes') <= 180)
                .then(1)
                .otherwise(0)
                .filter(pl.col('selected') == 1)
                .mean() * 0.3)
        .clip(0, 1).alias('short_haul_preference'),

        # Connection tolerance (average segments)
        pl.sum_horizontal([
            pl.col(f'legs0_segments{i}_departureFrom_airport_iata')
            .is_not_null().cast(pl.Int8)
            for i in range(4)  # assuming max 4 segments
        ]).filter(pl.col('selected') == 1).mean().alias('connection_tolerance'),

        # Preference for longer vs shorter flights (duration quartile preference)
        pl.when(pl.col('leg0_duration_minutes') <= pl.col('leg0_duration_q25'))
            .then(1)  # Short flights
            .when(pl.col('leg0_duration_minutes') <= pl.col('leg0_duration_q50'))
            .then(2)  # Medium-short flights
            .when(pl.col('leg0_duration_minutes') <= pl.col('leg0_duration_q75'))
            .then(3)  # Medium-long flights
            .otherwise(4)  # Long flights
            .filter(pl.col('selected') == 1)
            .mode().first().alias('preferred_duration_quartile')
    ]


def create_price_sensitivity_features() -> List[pl.Expr]:
    """Create features related to price sensitivity and patterns."""
    return [
        # Basic correlation between price and duration
        pl.corr(
            pl.col('totalPrice').filter(pl.col('selected') == 1),
            pl.col('trip_duration_minutes').filter(pl.col('selected') == 1)
        ).fill_null(0).alias('price_to_duration_sensitivity'),

        # Price per minute metric (average across all flights)
        (pl.col('totalPrice') / pl.col('trip_duration_minutes').clip(1, None))
            .filter(pl.col('selected') == 1).mean()
            .alias('avg_price_per_minute'),

        # Consistency of price-per-minute (lower std = more consistent valuation)
        (pl.col('totalPrice') / pl.col('trip_duration_minutes').clip(1, None))
            .filter(pl.col('selected') == 1).std().fill_null(0)
            .alias('price_per_minute_variance'),

        # Price position within searches (percentile rank)
        pl.col('price_percentile').filter(pl.col('selected') == 1).mean().alias('price_position_preference'),

        # Premium economy preference (assuming cabin class 2 is premium economy)
        pl.mean_horizontal([
            pl.col(f'legs0_segments{i}_cabinClass').filter(pl.col('selected') == 1) == 2
            for i in range(4)
        ]).mean().alias('premium_economy_preference'),

        # Consistent price tier
        (1 - (
            pl.when(pl.col('totalPrice') <= pl.col('price_q25'))
                .then(1)  # Budget tier
                .when(pl.col('totalPrice') <= pl.col('price_q50'))
                .then(2)  # Economy tier
                .when(pl.col('totalPrice') <= pl.col('price_q75'))
                .then(3)  # Premium tier
                .otherwise(4)  # Luxury tier
                .filter(pl.col('selected') == 1)
                .std()
                .fill_null(0)  # Handle null case
            / 3
        ).clip(0, 1)).alias('consistent_price_tier'),

        # Most common price tier
        pl.when(pl.col('totalPrice') <= pl.col('price_q25'))
            .then(1)  # Budget tier
            .when(pl.col('totalPrice') <= pl.col('price_q50'))
            .then(2)  # Economy tier
            .when(pl.col('totalPrice') <= pl.col('price_q75'))
            .then(3)  # Premium tier
            .otherwise(4)  # Luxury tier
            .filter(pl.col('selected') == 1)
            .mode().first().alias('preferred_price_tier'),
    ]


def create_service_preference_features() -> List[pl.Expr]:
    """Create features related to service preferences."""
    # First get all the relevant column names for type and quantity
    type_cols = [
        f'legs{leg}_segments{seg}_baggageAllowance_weightMeasurementType'
        for leg in range(2) for seg in range(4)
    ]
    qty_cols = [
        f'legs{leg}_segments{seg}_baggageAllowance_quantity'
        for leg in range(2) for seg in range(4)
    ]

    return [
        # Baggage quantity preference (average of minimum bags allowed per flight option)
        pl.min_horizontal([
            pl.when(pl.col(type_col) == 0)
                .then(pl.col(qty_col))
                .otherwise(pl.lit(None))
            for type_col, qty_col in zip(type_cols, qty_cols)
        ]).filter(pl.col('selected') == 1).mean().fill_null(0).alias('baggage_qty_preference'),

        # Baggage weight preference (average of minimum weight allowed per flight option)
        pl.min_horizontal([
            pl.when(pl.col(type_col) == 1)
                .then(pl.col(qty_col))
                .otherwise(pl.lit(None))
            for type_col, qty_col in zip(type_cols, qty_cols)
        ]).filter(pl.col('selected') == 1).mean().fill_null(0).fill_nan(0).alias('baggage_weight_preference'),

        # Loyalty program utilization
        pl.when(pl.col('selected') == 1)
            .then(
                pl.col('frequentFlyer')
                    .str.contains(pl.col('legs0_segments0_marketingCarrier_code'))
                    .fill_null(False)
            )
            .otherwise(pl.lit(None))
            .mean()
            .fill_null(0)
            .alias('loyalty_program_utilization')
    ]


def create_derived_metrics() -> List[pl.Expr]:
    """Create derived metrics from combinations of features."""
    return [
        # Convenience priority score (higher = more emphasis on convenient times)
        ((1 - pl.col('time_of_day_variance')) * 10 +
            pl.col('price_to_duration_sensitivity') * 5
        ).alias('convenience_priority_score'),

        # Loyalty vs price index (higher = more loyal, less price sensitive)
        (pl.col('loyalty_program_utilization') * 10 -
            pl.col('price_position_preference') / 10
        ).alias('loyalty_vs_price_index'),

        # Planning consistency score (inverse of lead time variance)
        (1 / (pl.col('max_booking_lead_days') - pl.col('min_booking_lead_days') + 1)
         ).alias('planning_consistency_score'),

        # Luxury index (combination of cabin class and price tier)
        (pl.col('avg_cabin_class') * 20 +
            pl.col('price_position_preference') / 2
        ).alias('luxury_index'),

        # Create advanced features
        (pl.col('total_flights_searched') / pl.col('unique_routes_searched').clip(1)).alias('search_intensity_per_route'),
        (pl.col('max_booking_lead_days') - pl.col('min_booking_lead_days')).alias('lead_time_variance'),
        (pl.col('avg_booking_lead_days') / pl.col('median_booking_lead_days').clip(1)).alias('lead_time_skew'),
        (pl.col('unique_carriers_used') / pl.col('total_flights_searched').clip(1)).alias('carrier_diversity'),
        (pl.col('unique_departure_airports') / pl.col('total_flights_searched').clip(1)).alias('airport_diversity'),
        (pl.col('max_cabin_class') - pl.col('min_cabin_class')).alias('cabin_class_range'),
        (pl.col('is_vip').cast(pl.Int8) * 2 + pl.col('has_corp_codes').is_not_null().cast(pl.Int8)
         ).alias('customer_tier'),
    ]


def add_trip_duration(df: pl.DataFrame) -> pl.DataFrame:
    """Add trip duration to the data set, for reference."""
    leg0_duration_minutes = parse_duration_to_minutes('legs0_duration')
    leg1_duration_minutes = parse_duration_to_minutes('legs1_duration')
    trip_duration_minutes = leg0_duration_minutes + leg1_duration_minutes

    return df.with_columns([
        leg0_duration_minutes.alias('leg0_duration_minutes'),
        leg1_duration_minutes.alias('leg1_duration_minutes'),
        trip_duration_minutes.alias('trip_duration_minutes'),
    ])


def create_windows_based_features(df) -> pl.DataFrame:
    """Add window-based features for price and duration percentiles/quartiles for reference"""
    return (df.with_columns([
        # calculate price percentile over search session
        ((pl.col('totalPrice').rank(method='min').over('ranker_id') - 1) /
            (pl.col('totalPrice').count().over('ranker_id') - 1) * 100
        ).fill_null(50.0).alias('price_percentile'),

        # calculate price quartiles over profileId
        pl.col('totalPrice').quantile(0.25).over('profileId').alias('price_q25'),
        pl.col('totalPrice').quantile(0.50).over('profileId').alias('price_q50'),
        pl.col('totalPrice').quantile(0.75).over('profileId').alias('price_q75'),

        # calculate leg0_duration quartiles over profileId
        pl.col('leg0_duration_minutes').quantile(0.25).over('profileId').alias('leg0_duration_q25'),
        pl.col('leg0_duration_minutes').quantile(0.50).over('profileId').alias('leg0_duration_q50'),
        pl.col('leg0_duration_minutes').quantile(0.75).over('profileId').alias('leg0_duration_q75'),
    ]))


def create_interaction_features() -> List[pl.Expr]:
    # Create customer/business interaction features
    return [
        # Create VIP interactions
        (pl.col('search_intensity_per_route') * pl.col('is_vip')).alias('vip_search_intensity'),
        (pl.col('carrier_diversity') * pl.col('is_vip')).alias('vip_carrier_diversity'),
        (pl.col('avg_cabin_class') * pl.col('is_vip')).alias('vip_cabin_preference'),

        # Create corporate interactions
        (pl.col('total_flights_searched') * pl.col('has_corp_codes')).alias('corp_search_volume'),
        (pl.col('roundtrip_preference') * pl.col('has_corp_codes')).alias('corp_roundtrip_pref'),
        (pl.col('lead_time_variance') * pl.col('has_corp_codes')).alias('corp_planning_variance'),
    ]

def extract_customer_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract customer features for clustering analysis.
    Aggregates by profileId to create customer-level features.
    """
    # Check if already processed
    if df.height > 0 and 'total_flights_searched' in df.columns:
        return df

    # Get cabin class columns
    cabin_class_cols = [col for col in df.columns if col.startswith('legs') and col.endswith('_cabinClass')]

    # Create lazy frame and group by profileId
    lazy_df = create_windows_based_features(add_trip_duration(df)).lazy().group_by('profileId')

    # Apply customer feature groups
    lazy_df = lazy_df.agg([
        *create_customer_aggregation_features(),
        *create_booking_lead_time_features(),
        *create_travel_preference_features(),
        *create_cabin_class_features(cabin_class_cols),
        *create_temporal_preference_features(),
        *create_route_specific_features(),
        *create_price_sensitivity_features(),
        *create_service_preference_features()
    ])

    # Add the derived metrics that depend on the generated features
    lazy_df = lazy_df.with_columns(create_derived_metrics())

    # Add interactive features
    lazy_df = lazy_df.with_columns(create_interaction_features())

    # print(f"Generated {len(enhanced_features.columns)} customer features for {len(enhanced_features)} customers")
    return lazy_df.collect().fill_null(0).fill_nan(0)