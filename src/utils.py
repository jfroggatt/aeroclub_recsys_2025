import polars as pl
import re


def camel_to_snake(name):
    """Convert camelCase or PascalCase to snake_case"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def convert_columns_to_snake_case(df):
    """Convert all column names in a polars DataFrame to snake_case"""
    return df.rename({col: camel_to_snake(col) for col in df.columns})


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