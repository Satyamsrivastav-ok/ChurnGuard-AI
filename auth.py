import sqlite3
import hashlib

DATABASE = "users.db"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def signup(username, email, password):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users(username,email,password)
            VALUES(?,?,?)
            """,

            (
                username,
                email,
                hash_password(password)
            )
        )

        conn.commit()
        conn.close()

        return True

    except:

        conn.close()

        return False


def login(username, password):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(

        """
        SELECT * FROM users

        WHERE username=?

        AND password=?

        """,

        (
            username,
            hash_password(password)
        )

    )

    user = cursor.fetchone()

    conn.close()

    return user