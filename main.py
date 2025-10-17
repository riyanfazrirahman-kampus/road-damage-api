# main.py

import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image # type: ignore

# --- KONFIGURASI ---
MODEL_PATH = "model/model.h5"
LABELS_PATH = "model/labels.json"
IMG_SIZE = (224, 224)

# --- INISIALISASI APLIKASI, MODEL & LABEL ---
app = FastAPI(
    title="Road Damage Detection API",
    description="API untuk mendeteksi jenis kerusakan jalan dari gambar menggunakan model CNN MobileNetV2.",
    version="1.0.0",
)

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi list label kosong
model = None
CLASS_LABELS = []


def load_model():
    """Memuat model Keras dari file .h5"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        model = None


def load_labels():
    """Memuat label dari file JSON"""
    global CLASS_LABELS
    try:
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
        # Urutkan label berdasarkan key numerik
        CLASS_LABELS = [labels[str(i)] for i in range(len(labels))]
        print(f"✅ Label berhasil dimuat: {CLASS_LABELS}")
    except Exception as e:
        print(f"❌ Gagal memuat label: {e}")
        CLASS_LABELS = []


@app.on_event("startup")
def startup_event():
    """Event yang dijalankan saat aplikasi start"""
    load_model()
    load_labels()


# --- FUNGSI BANTUAN ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Melakukan preprocessing pada gambar sesuai logika baru.
    - Membuka gambar dari bytes
    - Mengonversi ke RGB
    - Mengubah ukuran ke IMG_SIZE
    - Mengonversi ke array dan menskalakan pixel ke [0, 1]
    - Menambahkan dimensi batch
    """
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    # Konversi ke array dan skala ke [0, 1]
    img_array = tf_image.img_to_array(image) / 255.0

    # Tambahkan dimensi batch
    img_array_expanded = np.expand_dims(img_array, axis=0)

    return img_array_expanded


# --- API ENDPOINT ---
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Selamat datang di Road Damage Detection API. Silakan akses /docs untuk dokumentasi."
    }


@app.post("/predict", tags=["Prediction"])
async def predict_damage(file: UploadFile = File(...)):
    """
    Menerima file gambar, memprosesnya, dan mengembalikan prediksi kerusakan jalan.
    Mengembalikan 3 prediksi teratas dengan confidence tertinggi.
    """
    if model is None or not CLASS_LABELS:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model atau label tidak tersedia. Server mungkin gagal memuatnya."
            },
        )

    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400, content={"error": "File yang diunggah bukan gambar."}
        )

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        # Lakukan prediksi
        predictions = model.predict(processed_image)[0]

        # Urutkan hasil dari yang terbesar
        sorted_indices = np.argsort(predictions)[::-1]

        # Ambil prediksi teratas
        top_predictions = []
        for i in sorted_indices[:3]:
            label = CLASS_LABELS[i]
            confidence_percent = float(predictions[i]) * 100  # Kalikan dengan 100
            top_predictions.append(
                {
                    "class": label,
                    "confidence": round(confidence_percent, 2),  # Bulatkan ke 2 desimal
                }
            )

        return JSONResponse(status_code=200, content={"predictions": top_predictions})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"},
        )
