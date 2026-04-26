from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from database.models import User, Vendor


def get_user_by_username(session: Session, username: str) -> User | None:
    return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def list_vendors(session: Session) -> list[Vendor]:
    statement = select(Vendor).order_by(Vendor.id.asc())
    return list(session.execute(statement).scalars().all())


def get_vendor(session: Session, vendor_id: int) -> Vendor | None:
    return session.get(Vendor, vendor_id)


def create_vendor(session: Session, payload: dict) -> Vendor:
    vendor = Vendor(**payload)
    session.add(vendor)
    session.commit()
    session.refresh(vendor)
    return vendor


def update_vendor(session: Session, vendor: Vendor, payload: dict) -> Vendor:
    for field, value in payload.items():
        setattr(vendor, field, value)
    session.commit()
    session.refresh(vendor)
    return vendor


def delete_vendor(session: Session, vendor: Vendor) -> None:
    session.delete(vendor)
    session.commit()


def vendor_performance_leaderboard(session: Session) -> list[Vendor]:
    statement = select(Vendor).order_by(desc(Vendor.performance_score), Vendor.id.asc())
    return list(session.execute(statement).scalars().all())
