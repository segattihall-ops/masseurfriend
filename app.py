from flask import Flask, jsonify, request, send_file, send_from_directory
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import random
import os
import secrets
import string
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
import sys
from pathlib import Path

from models import db, User, Client

BASE_DIR = Path(__file__).resolve().parent

bcrypt = Bcrypt()
jwt = JWTManager()

app = Flask(__name__)
CORS(app, supports_credentials=True)

default_db_path = BASE_DIR / "forecastcity.db"
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{default_db_path}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-me")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=12)

db.init_app(app)
bcrypt.init_app(app)
jwt.init_app(app)

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
df_previsoes = None
ranking = None

# Ensure database schema and demo records exist when the app boots
with app.app_context():
    db.create_all()
    seed_initial_data()

# -----------------------------------------
# COLETA + PREVISÃO
# -----------------------------------------
def coletar_dados(keyword, cidade_sigla):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe=f'{START_DATE} {END_DATE}', geo=cidade_sigla, gprop='')
    dados = pytrends.interest_over_time()
    if not dados.empty:
        dados = dados.reset_index()[["date", keyword]]
        dados.columns = ["ds", "y"]
        return dados
    return None


def prever_spikes(df, cidade, keyword, dias_futuros=14):
    df = df.copy()
    df.set_index("ds", inplace=True)
    seasonal_periods = min(30, max(7, len(df) // 6))
    modelo = ExponentialSmoothing(df["y"], trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    ajuste = modelo.fit()
    previsao = ajuste.forecast(dias_futuros)
    forecast = pd.DataFrame({"ds": previsao.index, "yhat": previsao.values, "cidade": cidade, "keyword": keyword})
    return forecast


# -----------------------------------------
# API: DASHBOARD, ADMIN, REFERRALS, AUTH, CRM, CHAT AI
# -----------------------------------------
@app.route("/api/run", methods=["POST"])
def run_model():
    global df_previsoes, ranking
    previsoes = []
    for keyword in KEYWORDS:
        for info in CIDADES_INFO:
            cidade = info["cidade"]
            estado = info["estado"]
            geo_code = f'US-{estado}'
            dados = coletar_dados(keyword, geo_code)
            if dados is not None and not dados.empty:
                pred = prever_spikes(dados, cidade, keyword)
                pred["concorrencia"] = info["concorrencia"]
                pred["regiao"] = info["regiao"]
                pred["lat"] = info["lat"]
                pred["lon"] = info["lon"]
                pred["demand_stars"] = np.clip((pred["yhat"] // 20).astype(int), 1, 5)
                pred["spike_prob"] = np.clip((pred["yhat"] / 100 * 95), 0, 95).round(1)
                pred["client_interest"] = (pred["yhat"] * random.uniform(1.5, 2.5)).round()
                previsoes.append(pred)

    if previsoes:
        df_previsoes = pd.concat(previsoes)
        df_previsoes["score"] = df_previsoes["yhat"] * (10 - df_previsoes["concorrencia"])
        hoje = datetime.date.today()
        df_previsoes["ds"] = pd.to_datetime(df_previsoes["ds"]).dt.date
        ranking = df_previsoes[df_previsoes["ds"] >= hoje]
        ranking = ranking.groupby(["cidade", "regiao", "lat", "lon"])["score"].mean().reset_index()
        ranking = ranking.sort_values(by="score", ascending=False)
        top = ranking.iloc[0]
        print(f"\U0001F4F2 ALERT: {top['cidade']} alta demanda ({int(top['score'])} pts)")
        return jsonify({"status": "ok", "top_city": top["cidade"]})
    else:
        return jsonify({"status": "error", "message": "Nenhuma previsão gerada"})


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
    if ranking is None:
        return jsonify({"error": "Modelo ainda não foi executado."}), 400
    return ranking.to_dict(orient="records")


@app.route("/api/data/predictions")
def get_predictions():
    if df_previsoes is None:
        return jsonify({"error": "Modelo ainda não foi executado."}), 400
    return df_previsoes.to_dict(orient="records")


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
