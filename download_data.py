"""
download_data.py
----------------
One-time script: downloads private TikTok data from Supabase (PostgreSQL)
into tiktok_data.db (SQLite) so the notebook can run without a live DB connection.

Set credentials as environment variables first — see README.md for instructions.

Usage
-----
Windows (PowerShell):
    $env:PG_HOST     = "aws-1-eu-west-1.pooler.supabase.com"
    $env:PG_PORT     = "5432"
    $env:PG_DBNAME   = "postgres"
    $env:PG_USER     = "postgres.mlmlcilyoqvbvgljsjtv"
    $env:PG_PASSWORD = "<your-password>"
    python download_data.py

macOS / Linux:
    export PG_HOST="aws-1-eu-west-1.pooler.supabase.com"
    export PG_PORT="5432"
    export PG_DBNAME="postgres"
    export PG_USER="postgres.mlmlcilyoqvbvgljsjtv"
    export PG_PASSWORD="<your-password>"
    python download_data.py

Output: tiktok_data.db (SQLite — listed in .gitignore, do not commit)
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

SQLITE_PATH = "tiktok_data.db"
TABLES = ["videos", "video_snapshots", "video_hashtags", "hashtags"]


def get_pg_engine():
    required = ["PG_HOST", "PG_PORT", "PG_DBNAME", "PG_USER", "PG_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("See README.md for setup instructions.")
        sys.exit(1)
    url = (
        f"postgresql+psycopg2://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}"
        f"@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DBNAME']}"
    )
    return create_engine(url)


def get_sqlite_engine():
    return create_engine(f"sqlite:///{SQLITE_PATH}")


def download_tables(pg_engine, sqlite_engine):
    for table in TABLES:
        print(f"Downloading: {table} ...", end=" ", flush=True)
        df = pd.read_sql(f"SELECT * FROM {table}", pg_engine)
        # SQLite has no UUID type — cast any UUID columns to strings
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                first_val = df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else None
                if hasattr(first_val, 'hex'):  # uuid.UUID objects have .hex
                    df[col] = df[col].astype(str)
        df.to_sql(table, sqlite_engine, if_exists="replace", index=False)
        print(f"{len(df):,} rows")
    print(f"\nDone. Data saved to: {SQLITE_PATH}")


if __name__ == "__main__":
    download_tables(get_pg_engine(), get_sqlite_engine())
