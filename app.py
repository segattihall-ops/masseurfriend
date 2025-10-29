from flask import Flask, jsonify, request, send_file, send_from_directory
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import random
import os
import secrets
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pytrends.request import TrendReq
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from flask_caching import Cache
import sys
from pathlib import Path

from models import db, User, Client, ForecastRun

BASE_DIR = Path(__file__).resolve().parent

bcrypt = Bcrypt()
jwt = JWTManager()
cache = Cache()
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("FORECAST_WORKERS", 2)))

app = Flask(__name__)
CORS(app, supports_credentials=True)

default_db_path = BASE_DIR / "forecastcity.db"
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{default_db_path}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-me")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=12)
app.config.setdefault("CACHE_TYPE", os.environ.get("CACHE_TYPE", "SimpleCache"))
app.config.setdefault("CACHE_DEFAULT_TIMEOUT", int(os.environ.get("CACHE_TIMEOUT", 3600)))

db.init_app(app)
bcrypt.init_app(app)
jwt.init_app(app)
cache.init_app(app)

INITIAL_USERS = [
    {
        "email": "admin@therankflow.com",
        "password": "admin123",
        "tier": "admin",
        "referral_code": "ADMIN123",
        "referrals": 12,
        "revenue": 4200,
    },
    {
        "email": "therapist1@email.com",
        "password": "abc123",
        "tier": "pro",
        "referral_code": "THERA1",
        "referrals": 2,
        "revenue": 650,
    },
    {
        "email": "newuser@email.com",
        "password": "testpass",
        "tier": "free",
        "referral_code": "FREENEW",
        "referrals": 0,
        "revenue": 0,
    },
]

INITIAL_CLIENTS = [
    {"name": "James", "city": "Atlanta", "last_seen": "2023-10-21", "repeat": True, "owner_email": "admin@therankflow.com"},
    {"name": "Liam", "city": "Boston", "last_seen": "2023-09-18", "repeat": False, "owner_email": "admin@therankflow.com"},
    {"name": "Mason", "city": "Dallas", "last_seen": "2023-10-01", "repeat": True, "owner_email": "admin@therankflow.com"},
]


def generate_referral_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(length))
        if not User.query.filter_by(referral_code=code).first():
            return code


def seed_initial_data() -> None:
    for user_data in INITIAL_USERS:
        if User.query.filter_by(email=user_data["email"]).first():
            continue
        password_hash = bcrypt.generate_password_hash(user_data["password"]).decode("utf-8")
        user = User(
            email=user_data["email"],
            password_hash=password_hash,
            tier=user_data["tier"],
            referral_code=user_data["referral_code"],
            referrals=user_data["referrals"],
            revenue=user_data["revenue"],
        )
        db.session.add(user)

    db.session.commit()

    for client_data in INITIAL_CLIENTS:
        owner = User.query.filter_by(email=client_data["owner_email"]).first()
        if not owner:
            continue
        exists = Client.query.filter_by(name=client_data["name"], user_id=owner.id).first()
        if exists:
            continue
        last_seen = datetime.datetime.strptime(client_data["last_seen"], "%Y-%m-%d").date()
        client = Client(
            name=client_data["name"],
            city=client_data["city"],
            last_seen=last_seen,
            repeat=client_data["repeat"],
            user_id=owner.id,
        )
        db.session.add(client)

    db.session.commit()

KEYWORDS = ["gay massage", "male massage therapist near me", "rentmasseur", "deep tissue gay massage"]
CIDADES_INFO = [
    {"cidade": "Atlanta", "estado": "GA", "regiao": "Sudeste", "concorrencia": 8.2, "lat": 33.7490, "lon": -84.3880},
    {"cidade": "Boston", "estado": "MA", "regiao": "Nordeste", "concorrencia": 9.1, "lat": 42.3601, "lon": -71.0589},
    {"cidade": "Chicago", "estado": "IL", "regiao": "Centro-Norte", "concorrencia": 6.5, "lat": 41.8781, "lon": -87.6298},
    {"cidade": "Dallas", "estado": "TX", "regiao": "Sul", "concorrencia": 7.8, "lat": 32.7767, "lon": -96.7970},
    {"cidade": "San Francisco", "estado": "CA", "regiao": "Oeste", "concorrencia": 9.3, "lat": 37.7749, "lon": -122.4194}
]

START_DATE = "2022-01-01"
END_DATE = datetime.date.today().isoformat()
LATEST_FORECAST_CACHE_KEY = "forecast:latest:id"


def get_latest_forecast_run() -> Optional[ForecastRun]:
    cached_id = cache.get(LATEST_FORECAST_CACHE_KEY)
    run: Optional[ForecastRun] = None
    if cached_id:
        run = ForecastRun.query.get(cached_id)
    if not run:
        run = ForecastRun.query.order_by(ForecastRun.created_at.desc()).first()
    if run:
        cache.set(LATEST_FORECAST_CACHE_KEY, run.id, timeout=app.config["CACHE_DEFAULT_TIMEOUT"])
    return run


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["ds"] = pd.to_datetime(cleaned["ds"])
    cleaned = cleaned.sort_values("ds").drop_duplicates(subset=["ds"])
    cleaned["y"] = pd.to_numeric(cleaned["y"], errors="coerce")
    cleaned["y"] = cleaned["y"].interpolate(method="linear")
    cleaned["y"] = cleaned["y"].fillna(method="bfill").fillna(method="ffill")

    if cleaned["y"].std(ddof=0) > 0:
        z_scores = np.abs((cleaned["y"] - cleaned["y"].mean()) / cleaned["y"].std(ddof=0))
        cleaned.loc[z_scores > 3, "y"] = np.nan
        cleaned["y"] = cleaned["y"].interpolate(method="linear")

    cleaned["y"] = cleaned["y"].clip(lower=0)
    cleaned = cleaned.dropna(subset=["y"])
    return cleaned


def detect_seasonality(series: pd.Series) -> Tuple[str, int]:
    if series.empty:
        return "add", 7

    series = series.asfreq(series.index.inferred_freq, method="ffill") if series.index.inferred_freq else series
    inferred_freq = series.index.inferred_freq
    seasonal_periods = 7

    if inferred_freq:
        if inferred_freq.startswith("W"):
            seasonal_periods = min(52, max(2, len(series) // 6))
        elif inferred_freq.startswith("M"):
            seasonal_periods = min(12, max(2, len(series) // 4))
        else:
            seasonal_periods = min(30, max(7, len(series) // 6))
    else:
        seasonal_periods = min(30, max(7, len(series) // 6))

    if seasonal_periods < 2:
        seasonal_periods = 2

    mean = series.mean()
    std = series.std(ddof=0)
    variation = std / mean if mean else 0
    seasonality = "mul" if variation > 0.5 else "add"
    return seasonality, int(seasonal_periods)


def evaluate_backtest(series: pd.Series, seasonality: str, seasonal_periods: int) -> Dict[str, Any]:
    if len(series) < max(seasonal_periods * 2, 10):
        return {}

    horizon = max(7, min(21, len(series) // 4))
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    if train.empty or test.empty:
        return {}

    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=seasonality,
            seasonal_periods=seasonal_periods,
        ).fit()
        forecast = model.forecast(horizon)
    except Exception:
        return {}

    comparison = pd.DataFrame({"actual": test, "predicted": forecast})
    comparison = comparison.dropna()
    if comparison.empty:
        return {}

    actual = comparison["actual"].clip(lower=1e-3)
    mape = float(np.mean(np.abs((comparison["predicted"] - actual) / actual)) * 100)
    rmse = float(np.sqrt(np.mean((comparison["predicted"] - actual) ** 2)))

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "horizon": int(horizon),
        "testCount": int(len(comparison)),
    }


def build_forecast(
    df: pd.DataFrame,
    cidade: str,
    keyword: str,
    info: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prepared = prepare_time_series(df)
    if prepared.empty:
        raise ValueError("Dataset vazio após preparação")

    prepared = prepared.set_index("ds")
    seasonality, seasonal_periods = detect_seasonality(prepared["y"])

    model = ExponentialSmoothing(
        prepared["y"],
        trend="add",
        seasonal=seasonality,
        seasonal_periods=seasonal_periods,
    ).fit()

    dias_futuros = 14
    previsao = model.forecast(dias_futuros)

    metrics = evaluate_backtest(prepared["y"], seasonality, seasonal_periods)
    last_observed = float(prepared["y"].iloc[-1])

    base_df = pd.DataFrame(
        {
            "ds": previsao.index,
            "yhat": previsao.values,
        }
    )

    rmse = metrics.get("rmse", 0)
    base_df["yhat_lower"] = np.clip(base_df["yhat"] - rmse, a_min=0, a_max=None)
    base_df["yhat_upper"] = base_df["yhat"] + rmse

    base_df["cidade"] = cidade
    base_df["keyword"] = keyword
    base_df["concorrencia"] = info["concorrencia"]
    base_df["regiao"] = info["regiao"]
    base_df["lat"] = info["lat"]
    base_df["lon"] = info["lon"]

    base_df["demand_stars"] = np.clip((base_df["yhat"] // 20).astype(int), 1, 5)
    base_df["spike_prob"] = np.clip(((base_df["yhat"] - last_observed) / (last_observed + 1e-3)) * 100, 0, 100).round(1)
    base_df["client_interest"] = (base_df["yhat"] * random.uniform(1.8, 2.6)).round()
    base_df["score"] = (base_df["yhat"] * (10 - info["concorrencia"]))

    metrics.update(
        {
            "keyword": keyword,
            "city": cidade,
            "seasonality": seasonality,
            "seasonalPeriods": seasonal_periods,
            "lastObserved": round(last_observed, 2),
            "forecastHorizon": dias_futuros,
        }
    )

    return base_df, metrics


def serialize_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    payload = []
    for record in df.to_dict(orient="records"):
        item = dict(record)
        if isinstance(item.get("ds"), (datetime.date, datetime.datetime, pd.Timestamp)):
            item["ds"] = pd.to_datetime(item["ds"]).date().isoformat()
        for key, value in list(item.items()):
            if isinstance(value, (np.floating, np.float64)):
                item[key] = float(value)
            elif isinstance(value, (np.integer, np.int64)):
                item[key] = int(value)
            elif isinstance(value, (np.bool_,)):
                item[key] = bool(value)
        payload.append(item)
    return payload


def summarise_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}

    mape_values = [m["mape"] for m in metrics if "mape" in m]
    rmse_values = [m["rmse"] for m in metrics if "rmse" in m]

    summary = {
        "combinations": metrics,
    }

    if mape_values:
        summary["globalMape"] = round(float(np.mean(mape_values)), 2)
        summary["bestMape"] = round(float(np.min(mape_values)), 2)
    if rmse_values:
        summary["globalRmse"] = round(float(np.mean(rmse_values)), 2)
        summary["bestRmse"] = round(float(np.min(rmse_values)), 2)

    return summary

# Ensure database schema and demo records exist when the app boots
with app.app_context():
    db.create_all()
    seed_initial_data()

# -----------------------------------------
# COLETA + PREVISÃO
# -----------------------------------------


@cache.memoize(timeout=int(os.environ.get("TRENDS_CACHE_TTL", 6 * 3600)))
def coletar_dados(keyword: str, cidade_sigla: str) -> Optional[pd.DataFrame]:
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe=f"{START_DATE} {END_DATE}", geo=cidade_sigla, gprop="")
    dados = pytrends.interest_over_time()
    if not dados.empty:
        dados = dados.reset_index()[["date", keyword]]
        dados.columns = ["ds", "y"]
        return dados
    return None


def generate_forecast_payload() -> Dict[str, Any]:
    previsoes: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []

    for keyword in KEYWORDS:
        for info in CIDADES_INFO:
            cidade = info["cidade"]
            estado = info["estado"]
            geo_code = f"US-{estado}"
            dados = coletar_dados(keyword, geo_code)
            if dados is None or dados.empty:
                continue
            try:
                pred, metric = build_forecast(dados, cidade, keyword, info)
            except ValueError:
                continue
            previsoes.append(pred)
            metrics.append(metric)

    if not previsoes:
        raise ValueError("Nenhuma previsão gerada com os parâmetros atuais.")

    previsoes_df = pd.concat(previsoes, ignore_index=True)
    previsoes_df["ds"] = pd.to_datetime(previsoes_df["ds"]).dt.date
    previsoes_df["score"] = previsoes_df["score"].round(2)
    previsoes_df["yhat"] = previsoes_df["yhat"].round(2)
    previsoes_df["yhat_lower"] = previsoes_df["yhat_lower"].round(2)
    previsoes_df["yhat_upper"] = previsoes_df["yhat_upper"].round(2)

    ranking_df = (
        previsoes_df.groupby(["cidade", "regiao", "lat", "lon"], as_index=False)["score"].mean()
        .sort_values(by="score", ascending=False)
    )

    top_city = ranking_df.iloc[0]["cidade"] if not ranking_df.empty else None

    ranking_payload = [
        {
            "cidade": row["cidade"],
            "regiao": row["regiao"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "score": round(float(row["score"]), 2),
        }
        for row in ranking_df.to_dict(orient="records")
    ]

    metrics_payload = summarise_metrics(metrics) or {}
    metrics_payload["combinationCount"] = len(metrics)

    return {
        "predictions": serialize_dataframe(previsoes_df),
        "ranking": ranking_payload,
        "metrics": metrics_payload,
        "keyword_count": len(KEYWORDS),
        "city_count": len(CIDADES_INFO),
        "top_city": top_city,
    }


def execute_forecast_job(run_id: int) -> None:
    with app.app_context():
        run = ForecastRun.query.get(run_id)
        if not run:
            return

        run.status = "running"
        db.session.commit()

        try:
            payload = generate_forecast_payload()
        except Exception as exc:  # noqa: BLE001
            run.status = "failed"
            run.metrics = {"error": str(exc)}
            db.session.commit()
            return

        run.status = "completed"
        run.keyword_count = payload["keyword_count"]
        run.city_count = payload["city_count"]
        run.top_city = payload["top_city"]
        combined_metrics = payload["metrics"]
        if isinstance(run.metrics, dict):
            combined_metrics = {**run.metrics, **payload["metrics"]}
        run.metrics = combined_metrics
        run.ranking = payload["ranking"]
        run.predictions = payload["predictions"]
        db.session.commit()

        cache.set(LATEST_FORECAST_CACHE_KEY, run.id, timeout=app.config["CACHE_DEFAULT_TIMEOUT"])
        cache.set(f"forecast:payload:{run.id}", run.to_dict(), timeout=app.config["CACHE_DEFAULT_TIMEOUT"])


# -----------------------------------------
# API: DASHBOARD, ADMIN, REFERRALS, AUTH, CRM, CHAT AI
# -----------------------------------------
@app.route("/api/run", methods=["POST"])
@jwt_required(optional=True)
def run_model():
    triggered_by = get_jwt_identity()

    run = ForecastRun(
        status="queued",
        keyword_count=len(KEYWORDS),
        city_count=len(CIDADES_INFO),
        metrics={"triggeredBy": triggered_by} if triggered_by else {},
    )
    db.session.add(run)
    db.session.commit()

    executor.submit(execute_forecast_job, run.id)

    return jsonify(
        {
            "status": "queued",
            "runId": run.id,
            "message": "Previsão em andamento. Consulte o status para acompanhar o progresso.",
        }
    ), 202


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"status": "error", "message": "Credenciais inválidas."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"status": "error", "message": "Credenciais inválidas."}), 401

    token = create_access_token(identity=user.email)
    return jsonify({"status": "ok", "token": token, "user": user.to_dict()})


@app.route("/api/data/summary")
def get_summary():
    run = get_latest_forecast_run()
    if not run or not run.ranking:
        return jsonify({"error": "Nenhuma previsão disponível."}), 404
    return jsonify(run.ranking)


@app.route("/api/data/predictions")
def get_predictions():
    run = get_latest_forecast_run()
    if not run or not run.predictions:
        return jsonify({"error": "Nenhuma previsão disponível."}), 404
    return jsonify(run.predictions)


@app.route("/api/forecasts/<int:run_id>")
@jwt_required(optional=True)
def get_forecast_run(run_id: int):
    run = ForecastRun.query.get(run_id)
    if not run:
        return jsonify({"error": "Previsão não encontrada."}), 404
    return jsonify(run.to_dict())


@app.route("/api/forecasts/latest")
def get_latest_forecast():
    run = get_latest_forecast_run()
    if not run:
        return jsonify({"error": "Nenhuma previsão disponível."}), 404
    return jsonify(run.to_dict())


@app.route("/api/users")
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


@app.route("/api/user/<email>")
def get_user_by_email(email):
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Usuário não encontrado."}), 404
    return jsonify(user.to_dict())


@app.route("/api/referral/<code>")
def referral_lookup(code):
    user = User.query.filter_by(referral_code=code).first()
    if not user:
        return jsonify({"error": "Código inválido."}), 404
    return jsonify(user.to_dict())


@app.route("/api/onboarding")
def onboarding_steps():
    return jsonify([
        "\U0001F4CD Ative sua localização para ver as cidades próximas.",
        "\U0001F4C8 Execute o modelo para obter oportunidades atuais.",
        "\U0001F680 Use seu link de indicação para convidar amigos.",
        "✅ Marque cidades favoritas para gerar alertas."
    ])


@app.route("/api/clients")
@jwt_required()
def get_clients():
    current_email = get_jwt_identity()
    user = User.query.filter_by(email=current_email).first()
    if not user:
        return jsonify({"error": "Usuário não encontrado."}), 404

    clients = Client.query.filter_by(user_id=user.id).all()
    return jsonify([client.to_dict() for client in clients])


@app.route("/api/register", methods=["POST"])
def register_user():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    tier = data.get("tier", "free")
    referral_code = data.get("referral_code")

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email já registrado"}), 409

    if referral_code:
        if User.query.filter_by(referral_code=referral_code).first():
            return jsonify({"error": "Código de indicação já em uso"}), 409
    else:
        referral_code = generate_referral_code()

    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    new_user = User(
        email=email,
        password_hash=password_hash,
        tier=tier,
        referral_code=referral_code,
        referrals=0,
        revenue=0,
    )

    db.session.add(new_user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        referral_code = generate_referral_code()
        new_user.referral_code = referral_code
        db.session.add(new_user)
        db.session.commit()

    return jsonify({"message": "Usuário registrado com sucesso", "user": new_user.to_dict()}), 201


@app.route("/api/chat", methods=["POST"])
def chat_concierge():
    data = request.json
    user_msg = data.get("message")
    return jsonify({"reply": f"Olá! Você perguntou: '{user_msg}'. Em breve, eu responderei com inteligência completa."})


@app.route("/api/notify/pushcut", methods=["POST"])
def send_pushcut():
    data = request.json
    print(f"\U0001F514 PUSHCUT: Enviando alerta para {data.get('city')} com score {data.get('score')}")
    return jsonify({"status": "ok"})


@app.route("/api/download/<file>")
def download(file):
    file_map = {
        "ranking": "ranking_semana.csv",
        "clientes": "clientes_previstos.csv",
        "faturamento": "faturamento_estimado.csv",
        "previsoes": "previsoes_completas.csv",
        "grafico": "ranking_oportunidade.png"
    }
    if file not in file_map:
        return "Arquivo não encontrado", 404
    target = BASE_DIR / file_map[file]
    if not target.exists():
        return "Arquivo não encontrado", 404
    return send_file(target, as_attachment=True)


@app.route("/")
def serve_index():
    """Serve the interactive dashboard."""
    return send_from_directory(str(BASE_DIR), "index.html")


# -----------------------------------------
# ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not isinstance(port, int) or port < 1 or port > 65535:
        print("Erro: Porta inválida ou não configurada corretamente.")
        sys.exit(1)
    try:
        print(f"Iniciando servidor Flask na porta {port}...")
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Erro ao iniciar o servidor: {e}")
        sys.exit(1)
