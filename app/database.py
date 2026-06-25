from pathlib import Path
import sqlite3

import pandas as pd
from fastapi import HTTPException


def load_sqlite_table(db_path: str, table_name: str) -> pd.DataFrame:
    path = Path(db_path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail="SQLite database file not found.")

    safe_table = table_name.replace('"', "")
    query = f'SELECT * FROM "{safe_table}"'

    try:
        with sqlite3.connect(path) as connection:
            return pd.read_sql_query(query, connection)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not load SQLite table.") from error
