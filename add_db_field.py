#!/usr/bin/env python3
import argparse
import os
import sqlite3
import sys

def get_db_path(db_arg: str | None) -> str:
    if db_arg:
        return db_arg
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "database", "audio_data.db")

def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return any(row[1] == column for row in cur.fetchall())

def validate_type(t: str) -> str:
    t = t.strip().upper()
    allowed = {"TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"}
    if t not in allowed:
        raise argparse.ArgumentTypeError(f"type must be one of {sorted(allowed)}")
    return t

def main():
    p = argparse.ArgumentParser(description="Add a column to the SQLite DB (idempotent)")
    p.add_argument("table", help="table name")
    p.add_argument("column", help="new column name")
    p.add_argument("type", nargs="?", default="TEXT", type=validate_type, help="column type (default: TEXT)")
    p.add_argument("--db", dest="db", default=None, help="path to database (default: database/audio_data.db)")
    p.add_argument("--default", dest="default", default=None, help="default value for existing/new rows")
    p.add_argument("--not-null", dest="not_null", action="store_true", help="mark column NOT NULL (requires --default)")
    args = p.parse_args()

    db_path = get_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            if column_exists(conn, args.table, args.column):
                print(f"Column exists: {args.table}.{args.column}")
                return
            ddl = f'ALTER TABLE "{args.table}" ADD COLUMN "{args.column}" {args.type}'
            if args.not_null:
                if args.default is None:
                    print("--not-null requires --default", file=sys.stderr)
                    sys.exit(2)
                ddl += " NOT NULL"
            if args.default is not None:
                # Use literal for DEFAULT; quote TEXT, leave numbers unquoted
                try:
                    float(args.default)
                    default_lit = args.default
                except ValueError:
                    default_lit = "'" + args.default.replace("'", "''") + "'"
                ddl += f" DEFAULT {default_lit}"
            conn.execute("BEGIN")
            conn.execute(ddl)
            if args.default is not None and not args.not_null:
                conn.execute(
                    f'UPDATE "{args.table}" SET "{args.column}" = ? WHERE "{args.column}" IS NULL',
                    (args.default,)
                )
            conn.execute("COMMIT")
            print(f"Added column: {args.table}.{args.column} ({args.type})")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
