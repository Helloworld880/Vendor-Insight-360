import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["APP_ENV"] = "test"

from api.main import app
from database.db import SessionLocal, engine, get_db_session, initialize_database, wait_for_database
from utils.redis_client import redis_client


@pytest.fixture(scope="session", autouse=True)
def bootstrap_integration_services() -> None:
    wait_for_database()
    initialize_database()
    redis_client.ping()
    redis_client.get_client().flushdb()


@pytest.fixture()
def db_session() -> Session:
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(current_session: Session, current_transaction) -> None:
        if current_transaction.nested and not current_transaction._parent.nested:
            current_session.begin_nested()

    try:
        yield session
    finally:
        event.remove(session, "after_transaction_end", restart_savepoint)
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session: Session):
    redis_client.get_client().flushdb()
    db_session.execute(text("TRUNCATE TABLE vendors RESTART IDENTITY CASCADE"))
    db_session.flush()

    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    redis_client.get_client().flushdb()
