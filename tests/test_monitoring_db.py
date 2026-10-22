from src.monitoring import db


def test_monitoring_is_disabled_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def must_not_connect():
        raise AssertionError("should not open a database connection")

    monkeypatch.setattr(db, "get_connection", must_not_connect)

    assert db.monitoring_enabled() is False
    db.init_db()
    assert db.log_conversation("pregunta", "respuesta", sources=[]) is None
    db.log_feedback(1, "up")


def test_monitoring_is_enabled_with_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    assert db.monitoring_enabled() is True
