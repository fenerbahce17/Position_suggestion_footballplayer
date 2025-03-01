import requests

# 🔹 API adresi
url = "http://127.0.0.1:5000/predict"

# 🔹 Örnek oyuncu verisi
data = {
    "Physical_Strength_diff": 2.5,
    "Shooting_Skill_diff": 3.0,
    "Passing_Skill_diff": 1.8,
    "Defensive_Skill_diff": 2.2,
    "Attacking_Skill_diff": 2.9,
    "Speed_diff": 3.1,
    "Z_Score": 4.0
}

# 🔹 API'ye POST isteği gönder
response = requests.post(url, json=data)

# 🔹 Sonucu yazdır
print(response.json())
