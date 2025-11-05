# BNB Price Predictor

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12.1-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Prediksi harga BNB/USDT menggunakan Machine Learning dengan Linear Regression**

[Fitur](#-fitur) • [Instalasi](#-instalasi) • [Penggunaan](#-penggunaan) • [Catatan Penting](#-catatan-penting)

</div>

---
<h1 align="center">Tentang Proyek</h1>


Proyek ini adalah sistem prediksi harga cryptocurrency BNB (Binance Coin) menggunakan algoritma **Linear Regression**. Sistem ini mengambil data historis dari Binance API, melatih model machine learning, dan memprediksi pergerakan harga berikutnya.

### Keunggulan

-  **Real-time Data**: Menggunakan Binance API untuk data terkini
-  **Visualisasi Interaktif**: Grafik harga dan volume yang menarik
-  **Model Evaluasi**: Metrik lengkap (MSE, RMSE, MAE, MAPE)
-  **Easy to Use**: Jupyter Notebook yang user-friendly
-  **Akurat**: Training dengan 1000 data historis

---

<h1 align="center">Fitur</h1>

### Data Analysis
- Fetch data historis hingga 1000 hari dari Binance
- Validasi dan cleaning data otomatis
- Exploratory Data Analysis (EDA) lengkap
- Analisis korelasi antar fitur

### Machine Learning
- Linear Regression model
- Train-test split (80:20)
- Multiple metrics evaluation
- Feature engineering

### Visualisasi
- Grafik harga historis dengan rentang high-low
- Volume trading bar chart
- Perbandingan prediksi vs aktual
- Scatter plot untuk evaluasi model
- Proyeksi harga 30 hari terakhir

### Prediksi
- Prediksi harga hari berikutnya
- Persentase perubahan
- Indikator naik/turun
- Real-time price fetching

---
<h1 align="center">Instalasi</h1>

### Prerequisites

Pastikan Anda sudah menginstal:
- Python 3.12.1
- pip (Python package manager)
- Jupyter Notebook atau JupyterLab

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/bnb-price-predictor.git
cd BNBUSDT-Predict
```

### Step 2: Install Dependencies

```bash
pip install pandas==2.3.3
pip install numpy==1.26.4
pip install requests==2.32.3
pip install scikit-learn==1.6.1
pip install matplotlib==3.8.2
```

**Atau install semua sekaligus:**

```bash
pip install pandas==2.3.3 numpy==1.26.4 requests==2.32.3 scikit-learn==1.6.1 matplotlib==3.8.2
```

### Step 3: Jalankan Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

---
<h1 align="center">Penggunaan</h1>

<h3 align="center">Quick Start</h3>

1. **Buka Jupyter Notebook**: Jalankan `main.ipynb`
2. **Run All Cells**: Klik `Cell` → `Run All` atau tekan `Shift + Enter` pada setiap cell
3. **Lihat Hasil**: Prediksi akan muncul di output terakhir

<h3 align="center">Konfigurasi</h3>

Anda dapat mengubah parameter di cell "Konfig":

```python
SYMBOL = 'BNBUSDT'    # Pasangan trading (default: BNB/USDT)
INTERVAL = '1d'       # Interval data (1m, 5m, 1h, 1d, dll)
LIMIT = 1000          # Jumlah data historis (max: 1000)
```

<h3 align="center">Workflow</h3>

<div align="center">

Fetch Data Historis  
↓  
Data Validation & Cleaning  
↓  
Exploratory Data Analysis  
↓  
Feature Engineering  
↓  
Model Training (Linear Regression)  
↓  
Model Evaluation  
↓  
Price Prediction  
↓  
Visualization  

</div>

---

## Output Contoh

### Hasil Prediksi

```
============================================================
HASIL PREDIKSI
============================================================
Harga saat ini: 948.90 USDT
Prediksi harga berikutnya: 945.32 USDT
Turun 0.38%
============================================================
```

### Metrik Evaluasi

```
Evaluasi Model:
Mean Squared Error (MSE): 849.41
Root Mean Squared Error (RMSE): 29.14
Mean Absolute Error (MAE): 17.44
Mean Absolute Percentage Error (MAPE): 1.93%
```

---

## Catatan Penting

### Untuk Pengguna di Indonesia

> **PENTING**: Binance diblokir sebagian di Indonesia. Sangat disarankan menggunakan **VPN** untuk mengakses Binance API.

**Rekomendasi VPN:**
- ProtonVPN (Free tier tersedia)
- TunnelBear (yang saya gunakan)
- Windscribe

**Cara Menggunakan:**
1. Install dan aktifkan VPN
2. Pilih server di negara yang tidak memblokir Binance (Singapore, Thailand, Malaysia, dll)
3. Jalankan notebook setelah VPN terhubung

### Keamanan

-  Proyek ini **TIDAK memerlukan API Key**
-  Hanya menggunakan public endpoint Binance
-  Tidak ada transaksi trading
-  Read-only data fetching

### Limitasi

- Rate limit Binance API: 1200 requests/minute
- Data historis maksimal: 1000 candles per request
- Model ini untuk **edukasi**, bukan financial advice
- Past performance ≠ Future results

---

## Penjelasan Teknis

### Model: Linear Regression

Linear Regression digunakan karena:
- Sederhana dan cepat untuk training
- Bagus untuk trend analysis
- Mudah diinterpretasi
- Cocok untuk data time series yang relatif stabil

### Features yang Digunakan

| Feature | Deskripsi |
|---------|-----------|
| `close` | Harga penutupan |
| `open` | Harga pembukaan |
| `high` | Harga tertinggi |
| `low` | Harga terendah |
| `volume` | Volume trading |

### Target Variable

- `next_close`: Harga penutupan hari berikutnya

---


## Disclaimer

> **PERINGATAN**: Proyek ini dibuat untuk tujuan **EDUKASI** dan **PEMBELAJARAN** machine learning.
> 
> -  BUKAN financial advice
> -  BUKAN rekomendasi trading
> -  Tidak bertanggung jawab atas kerugian trading
> -  Gunakan dengan risiko Anda sendiri
> -  Selalu lakukan riset sendiri (DYOR)

**Cryptocurrency trading mengandung risiko tinggi. Jangan investasi lebih dari yang Anda mampu kehilangan.**

---

## Author

Created with by **wahyuNurahmadinuh**

- GitHub: [@wahyuNurahmadinuh](https://github.com/wahyuNurahmadinuh)
- Email: wahyunurahmagaming@gmail.com

---


<div align="center">

**Jika proyek ini membantu Anda, berikan star ya!**

Made with Python and Machine Learning

</div>