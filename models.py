from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    tier = db.Column(db.String(50), default="free", nullable=False)
    referral_code = db.Column(db.String(32), unique=True)
    referrals = db.Column(db.Integer, default=0)
    revenue = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    clients = db.relationship("Client", backref="owner", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "tier": self.tier,
            "referral_code": self.referral_code,
            "referrals": self.referrals,
            "revenue": self.revenue,
            "created_at": self.created_at.isoformat(),
        }


class Client(db.Model):
    __tablename__ = "clients"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    city = db.Column(db.String(120), nullable=False)
    last_seen = db.Column(db.Date, nullable=False)
    repeat = db.Column(db.Boolean, default=False, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    def to_dict(self):
        return {
            "client_id": self.id,
            "name": self.name,
            "city": self.city,
            "last_seen": self.last_seen.isoformat() if isinstance(self.last_seen, date) else self.last_seen,
            "repeat": self.repeat,
        }
