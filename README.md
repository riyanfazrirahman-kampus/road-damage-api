## Struktur Folder

```py
/road_damage_api
|-- /model
|   |-- model.h5         <-- Letakkan file model Anda di sini
|-- main.py              <-- Kode utama aplikasi FastAPI
|-- requirements.txt     <-- Daftar library yang diperlukan

```

## Instal Dependencies

```sh
pip install -r requirements.txt
```

## Jalankan Server

```sh
uvicorn main:app --reload
```
