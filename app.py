from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Model ve scaler'ı yükle
with open("player_position_modelv1.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scalerv1.pkl", "rb") as f:
    scaler = pickle.load(f)

# Oyuncu verilerini içeren CSV dosyasını yükle
all_players = pd.read_csv("all_fifa_players.csv")

# Tüm sütun adlarını küçük harfe çevir
all_players.columns = all_players.columns.str.lower()

# Sütun isimlerini temizleme
all_players.columns = all_players.columns.str.lower().str.replace(" ", "_")

# Güncellenmiş sütun isimlerini kontrol et
print(all_players.columns.tolist())  # Yeni sütun adlarını gör
 
print(all_players["shot_power"].head())  # Çalışması gerekiyor 🚀

# Mevki önerme fonksiyonu
def suggest_position(player_features, current_position=None):
    # Mevki skorlarını hesapla
    skill_scores = {
        "ST": player_features[1] + player_features[4],
        "CM": player_features[2] + player_features[0],
        "CB": player_features[3] + player_features[0],
        "LW": player_features[5] + player_features[4],
        "RB": player_features[5] + player_features[3],
        "CDM": player_features[2] + player_features[3]
    }

    # Eğer mevcut pozisyon belirtilmişse, onu sil
    if current_position in skill_scores:
        del skill_scores[current_position]

    # En yüksek skorlu alternatifi döndür
    return max(skill_scores, key=skill_scores.get)

  
        

# overall_score sütununu dönüştürme fonksiyonu
def convert_to_float(value):
    try:
        # '+' işaretine göre ayır ve sayıları topla
        return sum(map(float, value.split('+')))
    except:
        # Eğer dönüştürme başarısız olursa NaN döndür
        return float('nan')

# overall_score sütununu dönüştür
all_players["overall_score"] = all_players["overall_score"].apply(convert_to_float)

# NaN değerlerini kaldır
all_players = all_players.dropna(subset=["overall_score"])

# Özellik mühendisliği fonksiyonu
def calculate_features(player):
    return {
        "Physical_Strength_diff": player["strength"],
        "Shooting_Skill_diff": player["finishing"] + player["shot_power"],
        "Passing_Skill_diff": player["short_passing"] + player["vision"],
        "Defensive_Skill_diff": player["standing_tackle"] + player["interceptions"],
        "Attacking_Skill_diff": player["attack_position"] + player["dribbling"],
        "Speed_diff": player["acceleration"] + player["sprint_speed"],
        "Z_Score": (player["overall_score"] - all_players["overall_score"].mean()) / all_players["overall_score"].std()
    }

# Ana sayfa
@app.route("/")
def home():
    return render_template("index.html")

# Oyuncu adı ile tahmin yapma
@app.route("/predict_by_name", methods=["POST"])
def predict_by_name():
    try:
        data = request.get_json()
        player_name = data.get("player_name")
        
        # Oyuncuyu veritabanında bul
        player = all_players[all_players["player"].str.lower() == player_name.lower()]
        
        if player.empty:
            return jsonify({"error": "Player not found"})

        print(player)  # Oyuncunun verisini terminalde göster

        # Özellikleri hesapla
        player_features = calculate_features(player.iloc[0])
        df = pd.DataFrame([player_features])
        
        # Model için ölçekleme
        X_scaled = scaler.transform(df)
        cluster = kmeans.predict(X_scaled)[0]
        suggested_position = suggest_position(df.iloc[0].values)
        
        return jsonify({"Cluster": int(cluster), "Suggested_Position": suggested_position})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Manuel veri girişi ile tahmin yapma
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        features = [
            "Physical_Strength_diff", "Shooting_Skill_diff", "Passing_Skill_diff",
            "Defensive_Skill_diff", "Attacking_Skill_diff", "Speed_diff", "Z_Score"
        ]

        X_scaled = scaler.transform(df[features])
        cluster = kmeans.predict(X_scaled)[0]
        suggested_position = suggest_position(df.iloc[0][features].values)

        return jsonify({"Cluster": int(cluster), "Suggested_Position": suggested_position})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    print(all_players["overall_score"].dtype)  # Sütunun veri tipini kontrol et