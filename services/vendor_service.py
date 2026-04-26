from __future__ import annotations

import json
from typing import Any

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from database import queries
from database.models import Vendor


class VendorService:
    @staticmethod
    def serialize_vendor(vendor: Vendor) -> dict[str, Any]:
        return {
            "id": vendor.id,
            "name": vendor.name,
            "category": vendor.category,
            "status": vendor.status,
            "delivery_rate": vendor.delivery_rate,
            "quality_score": vendor.quality_score,
            "cost_efficiency": vendor.cost_efficiency,
            "on_time_rate": vendor.on_time_rate,
            "cost_variance": vendor.cost_variance,
            "reliability": vendor.reliability,
            "performance_score": vendor.performance_score,
            "risk_score": vendor.risk_score,
            "created_at": vendor.created_at.isoformat() if vendor.created_at else None,
            "updated_at": vendor.updated_at.isoformat() if vendor.updated_at else None,
        }

    def list_vendors(self, session: Session) -> list[dict[str, Any]]:
        try:
            return [self.serialize_vendor(vendor) for vendor in queries.list_vendors(session)]
        except SQLAlchemyError as exc:
            raise RuntimeError("Unable to list vendors.") from exc

    def get_vendor(self, session: Session, vendor_id: int) -> dict[str, Any] | None:
        try:
            vendor = queries.get_vendor(session, vendor_id)
            return self.serialize_vendor(vendor) if vendor else None
        except SQLAlchemyError as exc:
            raise RuntimeError("Unable to fetch vendor.") from exc

    def create_vendor(self, session: Session, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            vendor = queries.create_vendor(session, payload)
            return self.serialize_vendor(vendor)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("A vendor with this name already exists.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise RuntimeError("Unable to create vendor.") from exc

    def update_vendor(self, session: Session, vendor_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            vendor = queries.get_vendor(session, vendor_id)
            if vendor is None:
                return None
            vendor = queries.update_vendor(session, vendor, payload)
            return self.serialize_vendor(vendor)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("A vendor with this name already exists.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise RuntimeError("Unable to update vendor.") from exc

    def delete_vendor(self, session: Session, vendor_id: int) -> bool:
        try:
            vendor = queries.get_vendor(session, vendor_id)
            if vendor is None:
                return False
            queries.delete_vendor(session, vendor)
            return True
        except SQLAlchemyError as exc:
            session.rollback()
            raise RuntimeError("Unable to delete vendor.") from exc

    def performance_leaderboard(self, session: Session) -> list[dict[str, Any]]:
        try:
            rows = [self.serialize_vendor(vendor) for vendor in queries.vendor_performance_leaderboard(session)]
            for index, row in enumerate(rows, start=1):
                row["rank"] = index
            return rows
        except SQLAlchemyError as exc:
            raise RuntimeError("Unable to load vendor performance.") from exc

    @staticmethod
    def encode_cache_payload(payload: list[dict[str, Any]]) -> str:
        return json.dumps(payload)

    @staticmethod
    def decode_cache_payload(payload: str) -> list[dict[str, Any]]:
        return json.loads(payload)
