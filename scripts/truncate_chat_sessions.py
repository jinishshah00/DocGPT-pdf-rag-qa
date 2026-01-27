#!/usr/bin/env python3
"""
Safe truncation script for the chat_sessions table (and related messages).

Usage:
  python scripts/truncate_chat_sessions.py --yes
  python scripts/truncate_chat_sessions.py --dry-run

This script will detect the project's DATABASE_URL from backend.config and
if it's a SQLite file it will back up the file before deleting rows.
"""
import argparse
import os
import shutil
import time
from urllib.parse import urlparse

from backend.config import DATABASE_URL
from backend.db import SessionLocal, engine
from backend.models import ChatSession, Message


def is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite:")


def sqlite_file_from_url(url: str) -> str | None:
    # url forms: sqlite:///./app.db or sqlite:////absolute/path/app.db
    if not is_sqlite_url(url):
        return None
    path = url.replace("sqlite:///", "")
    # If it began with 4 slashes, it was absolute
    if url.startswith("sqlite:////"):
        path = "/" + url.replace("sqlite:////", "")
    return os.path.abspath(path)


def backup_sqlite(path: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = f"{path}.bak.{ts}"
    shutil.copy2(path, dest)
    return dest


def truncate(session, dry_run: bool = False):
    # Collect session ids
    ids = [s.id for s in session.query(ChatSession.id).all()]
    print(f"Found {len(ids)} chat_sessions")
    if dry_run:
        return
    # delete messages referencing these sessions
    deleted_msgs = session.query(Message).delete()
    print(f"Deleted {deleted_msgs} messages (all messages)")
    deleted_sessions = session.query(ChatSession).delete()
    print(f"Deleted {deleted_sessions} chat_sessions")
    session.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Proceed with truncation")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify DB; just report")
    args = parser.parse_args()

    print(f"Using DATABASE_URL={DATABASE_URL}")

    sqlite_path = sqlite_file_from_url(DATABASE_URL)
    if sqlite_path:
        print(f"Detected SQLite DB at: {sqlite_path}")
        if not os.path.exists(sqlite_path):
            print("Database file not found. Aborting.")
            return
        if args.dry_run:
            print("Dry run: will not backup or modify DB.")
        else:
            bak = backup_sqlite(sqlite_path)
            print(f"Backed up database to {bak}")
    else:
        print("Non-sqlite DATABASE_URL detected. Proceeding without file backup.")

    if not args.yes and not args.dry_run:
        confirm = input("Type 'TRUNCATE' to proceed: ")
        if confirm != "TRUNCATE":
            print("Aborted by user.")
            return

    db = SessionLocal()
    try:
        truncate(db, dry_run=args.dry_run)
    finally:
        db.close()


if __name__ == "__main__":
    main()
