from flask import Flask, jsonify, request, send_file
import pandas as pd
import numpy as np
import datetime
import random
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pytrends.request import TrendReq
from flask_cors import CORS
import sys

app = Flask(__name__)
CORS(app)

# Mocked users for onboarding + referral + admin control
USERS_DB = [
    {"email": "admin@therankflow.com", "password": "admin123", "tier": "admin", "referral_code": "ADMIN123", "referrals": 12, "revenue": 4200},
    {"email": "therapist1@email.com", "password": "abc123", "tier": "pro", "referral_code": "THERA1", "referrals": 2, "revenue": 650},
    {"email": "newuser@email.com", "password": "testpass", "tier": "free", "referral_code": "FREENEW", "referrals": 0, "revenue": 0},
]

CLIENTS_DB = [
    {"client_id": 1, "name": "James", "city": "Atlanta", "last_seen": "2023-10-21", "repeat": True},
    {"client_id": 2, "name": "Liam", "city": "Boston", "last_seen": "2023-09-18", "repeat": False},
    {"client_id": 3, "name": "Mason", "city": "Dallas", "last_seen": "2023-10-01", "repeat": True}
]

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
    data = request.json
    user = next((u for u in USERS_DB if u["email"] == data.get("email") and u["password"] == data.get("password")), None)
    if user:
        return jsonify({"status": "ok", "user": user})
    return jsonify({"status": "error", "message": "Credenciais inválidas."}), 401


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
    return jsonify(USERS_DB)


@app.route("/api/user/<email>")
def get_user_by_email(email):
    user = next((u for u in USERS_DB if u["email"] == email), None)
    return jsonify(user or {"error": "Usuário não encontrado."})


@app.route("/api/referral/<code>")
def referral_lookup(code):
    user = next((u for u in USERS_DB if u["referral_code"] == code), None)
    return jsonify(user or {"error": "Código inválido."})


@app.route("/api/onboarding")
def onboarding_steps():
    return jsonify([
        "\U0001F4CD Ative sua localização para ver as cidades próximas.",
        "\U0001F4C8 Execute o modelo para obter oportunidades atuais.",
        "\U0001F680 Use seu link de indicação para convidar amigos.",
        "✅ Marque cidades favoritas para gerar alertas."
    ])


@app.route("/api/clients")
def get_clients():
    return jsonify(CLIENTS_DB)


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
    return send_file(file_map[file], as_attachment=True)


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
