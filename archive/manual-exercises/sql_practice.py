import sqlite3

db = sqlite3.connect("practice.sqlite3")

try:
    db.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        ("chat_test", "user", "你好"),
    )

    # 故意写错表名
    db.execute(
        "INSERT INTO wrong_table (chat_id) VALUES (?)",
        ("chat_test",),
    )

    db.commit()

except Exception as e:
    db.rollback()
    print("出错了，已经回滚：", e)

finally:
    db.close()