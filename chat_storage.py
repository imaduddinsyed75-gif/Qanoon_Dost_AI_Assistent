"""SQLite persistence for RightsGuide chat conversations."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path


DATABASE_PATH = Path(__file__).with_name("rights_guide_chats.sqlite3")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            messages TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    return connection


def load_conversations() -> OrderedDict:
    """Load persisted conversations in oldest-to-newest activity order."""

    conversations = OrderedDict()
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT chat_id, title, messages
            FROM conversations
            WHERE messages != '[]'
            ORDER BY updated_at ASC
            """
        ).fetchall()
    finally:
        connection.close()

    for chat_id, title, messages_json in rows:
        try:
            messages = json.loads(messages_json)
        except json.JSONDecodeError:
            continue
        if isinstance(messages, list) and messages:
            conversations[chat_id] = {"title": title, "messages": messages}
    return conversations


def save_conversation(chat_id, conversation) -> None:
    """Persist one non-empty conversation and its messages."""

    messages = conversation.get("messages", [])
    if not messages:
        return

    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO conversations (chat_id, title, messages, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                title = excluded.title,
                messages = excluded.messages,
                updated_at = excluded.updated_at
            """,
            (
                chat_id,
                conversation.get("title") or "Conversation",
                json.dumps(messages, ensure_ascii=False),
                time.time_ns(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def delete_conversation(chat_id: str) -> None:
    """Delete one persisted conversation."""

    connection = _connect()
    try:
        connection.execute(
            "DELETE FROM conversations WHERE chat_id = ?",
            (chat_id,),
        )
        connection.commit()
    finally:
        connection.close()
