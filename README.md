# SleepSense Flask API
**Team CC26-PSU230 | Coding Camp 2026 DBS Foundation**

REST API berbasis Flask untuk screening awal risiko stres, dideploy dengan Docker.
Menggunakan TensorFlow model + Gemini Flash via LangChain.

---

## Struktur Project

```
sleepsense_flask/
├── app/
│   └── main.py                   ← Flask API utama
├── models/
│   ├── sleepsense_model.keras    ← TF model (dari training)
│   ├── scaler_params.json        ← StandardScaler params
│   └── feature_meta.json         ← Feature metadata
├── notebooks/
│   └── SleepSense_Training.ipynb ← Training di Google Colab
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Endpoints

| Method | Endpoint   | Deskripsi |
|--------|------------|-----------|
| GET    | /health    | Health check |
| POST   | /predict   | Prediksi risiko dari model TF |
| POST   | /chat      | Respons empatik Gemini Flash |
| POST   | /analyze   | Predict + Chat dalam satu request |

---

## Request & Response

### POST /predict

**Request Body:**
```json
{
  "age": 22,
  "gender": "Male",
  "sleep_duration_hours": 5.5,
  "sleep_quality_score": 4.0,
  "daily_screen_time_hours": 8.0,
  "pre_sleep_screen_time_hours": 2.5,
  "physical_activity_minutes": 15,
  "caffeine_intake_cups": 4,
  "mental_fatigue_score": 7.5,
  "notifications_received_per_day": 120
}
```

**Response Body:**
```json
{
  "risk_label": "At Risk",
  "risk_level": "Tinggi",
  "risk_probability": 0.8234,
  "summary": "Pola tidur dan screen time Anda memerlukan perhatian segera.",
  "disclaimer": "Ini adalah screening awal, BUKAN diagnosis medis."
}
```

### POST /analyze (Predict + Gemini dalam satu request)

**Request Body:** sama dengan /predict

**Response Body:**
```json
{
  "prediction": {
    "risk_label": "At Risk",
    "risk_level": "Tinggi",
    "risk_probability": 0.8234,
    "summary": "...",
    "disclaimer": "..."
  },
  "advice": "Halo! Berdasarkan data tidurmu, kamu perlu memperhatikan..."
}
```

---

## Cara Menjalankan

### Step 1 - Training Model (Google Colab)

1. Buka `notebooks/SleepSense_Training.ipynb` di Colab
2. Jalankan semua cell
3. Download `sleepsense_flask_models.zip`
4. Extract dan letakkan file ke folder `models/`:
   ```
   models/sleepsense_model.keras
   models/scaler_params.json
   models/feature_meta.json
   ```

### Step 2 - Setup Environment

```bash
# Copy .env.example ke .env
cp .env.example .env

# Edit .env dan isi GEMINI_API_KEY
# Dapatkan API key gratis di: https://aistudio.google.com/apikey
nano .env
```

### Step 3 - Deploy dengan Docker

```bash
# Build image
docker build -t sleepsense-api .

# Jalankan container
docker compose up -d

# Cek status
docker compose ps
docker compose logs -f
```

### Step 4 - Test API

```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "gender": "Male",
    "sleep_duration_hours": 5.5,
    "sleep_quality_score": 4.0,
    "daily_screen_time_hours": 8.0,
    "pre_sleep_screen_time_hours": 2.5,
    "physical_activity_minutes": 15,
    "caffeine_intake_cups": 4,
    "mental_fatigue_score": 7.5,
    "notifications_received_per_day": 120
  }'

# Analyze (Predict + Gemini)
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "gender": "Male",
    "sleep_duration_hours": 5.5,
    "sleep_quality_score": 4.0,
    "daily_screen_time_hours": 8.0,
    "pre_sleep_screen_time_hours": 2.5,
    "physical_activity_minutes": 15,
    "caffeine_intake_cups": 4,
    "mental_fatigue_score": 7.5,
    "notifications_received_per_day": 120
  }'
```

---

## Docker Commands

```bash
# Build ulang setelah ada perubahan kode
docker compose up -d --build

# Stop container
docker compose down

# Lihat log real-time
docker compose logs -f sleepsense-api

# Masuk ke dalam container
docker exec -it sleepsense-api bash

# Restart container
docker compose restart
```

---

## Deploy ke Server (VPS/Cloud)

```bash
# Di server, clone repo
git clone https://github.com/kevinone0/sleepsense-flask.git
cd sleepsense-flask

# Upload model ke server (scp atau via GitHub LFS)
scp sleepsense_flask_models.zip user@server:/path/to/project/
unzip sleepsense_flask_models.zip -d models/

# Setup .env
cp .env.example .env
nano .env   # isi GEMINI_API_KEY

# Build dan jalankan
docker compose up -d --build

# Cek berjalan
curl http://YOUR_SERVER_IP:5000/health
```

---

## Mendapatkan Gemini API Key

1. Buka https://aistudio.google.com/apikey
2. Login dengan akun Google
3. Klik **Create API Key**
4. Copy key dan paste ke file `.env`

```
GEMINI_API_KEY=AIza...
```

> Output API bukan diagnosis medis — hanya screening awal.
