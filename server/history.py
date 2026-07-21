"""
对话历史持久化。用一个本地 sqlite 文件存下每一场对话和其中的每条消息，
重启后仍能查阅。单人本地应用，并发压力小，所以每次操作各开一个连接，
省去跨线程共享连接的麻烦（/chat 在子线程里写库）。
"""

import sqlite3
from pathlib import Path

_DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "history.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表，已存在就跳过。应用启动时调一次即可。"""
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT    NOT NULL DEFAULT '新对话',
            created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT    NOT NULL,
            content         TEXT    NOT NULL,
            position        INTEGER NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        """
    )
    conn.commit()
    conn.close()


def create_conversation(title: str | None = None) -> int:
    """开一场新对话，返回它的编号。标题留空先记成默认，等第一条提问再补。"""
    conn = _connect()
    cur = conn.execute("INSERT INTO conversations (title) VALUES (?)", (title or "新对话",))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def message_count(conversation_id: int) -> int:
    """某场对话已有多少条消息。"""
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return row["n"]


def conversation_exists(conversation_id: int) -> bool:
    """某个对话编号是否还在（前端带来的编号可能已被别处删掉）。"""
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return row is not None


def append_message(conversation_id: int, role: str, content: str) -> None:
    """往某场对话追加一条消息，位置自动接在最后一条后面。"""
    conn = _connect()
    row = conn.execute(
        "SELECT COALESCE(MAX(position), -1) AS p FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    pos = row["p"] + 1
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, position) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, pos),
    )
    conn.commit()
    conn.close()


def set_title(conversation_id: int, title: str) -> None:
    """改某场对话的标题（供重命名接口用）。"""
    conn = _connect()
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    conn.commit()
    conn.close()


def set_title_if_default(conversation_id: int, title: str) -> None:
    """只有标题还是默认「新对话」时才替换，不覆盖用户手动改过的标题。"""
    conn = _connect()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND title = '新对话'",
        (title, conversation_id),
    )
    conn.commit()
    conn.close()


def list_conversations() -> list[dict]:
    """列出所有对话，最近的排在前面，附带各自的消息条数。"""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.created_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS count
        FROM conversations c
        ORDER BY c.id DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_messages(conversation_id: int) -> list[dict]:
    """按顺序读出某场对话的全部消息。"""
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY position",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: int) -> None:
    """删掉某场对话，连同它下面的所有消息。"""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()