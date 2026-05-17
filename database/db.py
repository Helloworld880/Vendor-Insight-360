from __future__ import annotations

import re
import time
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from database.models import Base, User
from utils.security import hash_password, verify_password


settings = get_settings()
engine_kwargs = {
    "future": True,
    "pool_pre_ping": True,
}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    **engine_kwargs,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
DATABASE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
IS_SQLITE = engine.dialect.name == "sqlite"


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def initialize_database() -> None:
    wait_for_database()
    Base.metadata.create_all(bind=engine)
    seed_default_admin()


def create_database_if_missing(database_name: str) -> None:
    if settings.database_url_override or IS_SQLITE:
        return
    if not DATABASE_NAME_PATTERN.fullmatch(database_name):
        raise ValueError(f"Invalid database name '{database_name}'.")
    admin_engine = create_engine(
        settings.admin_database_url,
        future=True,
        pool_pre_ping=True,
        isolation_level="AUTOCOMMIT",
    )
    try:
        with admin_engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :database_name"),
                {"database_name": database_name},
            ).scalar_one_or_none()
            if exists is None:
                connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    finally:
        admin_engine.dispose()


def wait_for_database(max_attempts: int = 10, delay_seconds: int = 2) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if not settings.database_url_override and not IS_SQLITE:
                create_database_if_missing(settings.active_postgres_db)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds)
    raise RuntimeError("Database did not become ready in time.") from last_error


def seed_default_admin() -> None:
    with SessionLocal() as session:
        admin = session.query(User).filter(User.username == settings.admin_username).one_or_none()
        if admin is None:
            admin = User(
                username=settings.admin_username,
                password_hash=hash_password(settings.admin_password),
                role=settings.admin_role,
                is_active=True,
            )
            session.add(admin)
            session.commit()
            return

        updated = False
        if not verify_password(settings.admin_password, admin.password_hash):
            admin.password_hash = hash_password(settings.admin_password)
            updated = True
        if admin.role != settings.admin_role:
            admin.role = settings.admin_role
            updated = True
        if not admin.is_active:
            admin.is_active = True
            updated = True
        if updated:
            session.commit()


def database_ready() -> bool:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
