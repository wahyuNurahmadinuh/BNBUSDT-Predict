# BNB Price Predictor

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12.1-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**BNB/USDT Price Prediction Using Machine Learning with Linear Regression**

[Fitur](#-fitur) • [Instalasi](#-instalasi) • [Penggunaan](#-penggunaan) • [Catatan Penting](#-catatan-penting)

</div>

---
<h1 align="center">About Project</h1>


This project is a cryptocurrency price prediction system for BNB (Binance Coin) using the **Linear Regression** algorithm. The system retrieves historical data from the Binance API, trains a machine learning model, and predicts the next price movement.

### Advantages

-  **Real-time Data**: Uses the Binance API for up-to-date data  
-  **Interactive Visualization**: Attractive price and volume charts  
-  **Model Evaluation**: Complete metrics (MSE, RMSE, MAE, MAPE)  
-  **Easy to Use**: User-friendly Jupyter Notebook  
-  **Accurate**: Trained with 1000 historical data points  

---

<h1 align="center">Features</h1>

### Data Analysis
- Fetch up to 1000 days of historical data from Binance  
- Automatic data validation and cleaning  
- Comprehensive Exploratory Data Analysis (EDA)  
- Correlation analysis between features  

### Machine Learning
- Linear Regression model  
- Train-test split (80:20)  
- Multiple evaluation metrics  
- Feature engineering  

### Visualization
- Historical price chart with high-low range  
- Trading volume bar chart  
- Prediction vs actual comparison  
- Scatter plot for model evaluation  
- 30-day price projection  

### Prediction
- Next-day price prediction  
- Percentage change  
- Up/down indicator  
- Real-time price fetching  

---
<h1 align="center">Installation</h1>

### Prerequisites

Make sure you have installed:
- Python 3.12.1  
- pip (Python package manager)  
- Jupyter Notebook or JupyterLab  

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

**Or install everything at once:**

```bash
pip install pandas==2.3.3 numpy==1.26.4 requests==2.32.3 scikit-learn==1.6.1 matplotlib==3.8.2
```

### Step 3: Run Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

---
<h1 align="center">Usage</h1>

<h3 align="center">Quick Start</h3>

1. **Open Jupyter Notebook**: Run `main.ipynb`
2. **Run All Cells**: Click `Cell` → `Run All` or press `Shift + Enter` on each cell
3. **View Results**: The prediction will appear in the last output

You can also directly run the file "Predict With GUI.py" to try the finished application

<h3 align="center">Configuration</h3>

You can modify the parameters in the "Config" cell:

```python
SYMBOL = 'BNBUSDT'    # Trading pair (default: BNB/USDT)
INTERVAL = '1d'       # Data interval (1m, 5m, 1h, 1d, etc.)
LIMIT = 1000          # Number of historical data points (max: 1000)
```

<h3 align="center">Workflow</h3>

<div align="center">

Fetch Historical Data
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

## Sample Output

### Prediction Result

```
============================================================
Prediction Result
============================================================
Current price: 948.90 USDT
Next predicted price: 945.32 USDT
Down 0.38%
============================================================
```

### Evaluation Metrics

```
Model Evaluation:
Mean Squared Error (MSE): 849.41
Root Mean Squared Error (RMSE): 29.14
Mean Absolute Error (MAE): 17.44
Mean Absolute Percentage Error (MAPE): 1.93%
```

---

## Important Notes

### For Users in Indonesia

> **IMPORTANT**: Binance is partially blocked in Indonesia. It is highly recommended to use a VPN to access the Binance API

**Recommended VPNs:**
- ProtonVPN (Free tier available)
- TunnelBear (the one I use)
- Windscribe

**How to Use:**
1. Install and activate a VPN
2. Select a server in a country that does not block Binance (Singapore, Thailand, Malaysia, etc.)
3. Run the notebook after the VPN is connected

### Security

-  This project DOES NOT require an API Key
-  Only uses Binance public endpoints
-  No trading transactions
-  Read-only data fetching

### Limitations

- Binance API rate limit: 1200 requests/minute
- Maximum historical data: 1000 candles per request
- This model is for education, not financial advice
- Past performance ≠ Future results
---

## Technical Explanation

### Model: Linear Regression

Linear Regression is used because:
- Simple and fast to train
- Good for trend analysis
- Easy to interpret
- Suitable for relatively stable time series data

### Features yang Digunakan

| Feature | Description |
|---------|-----------|
| `close` | Closing price |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `volume` | Trading volume |

### Target Variable

- `next_close`: Next day closing price

---


## Disclaimer

> **WARNING**: This project is created for **EDUCATIONAL** and **LEARNING** purposes in machine learning.
> -  NOT financial advice
> -  NOT a trading recommendation
> -  No responsibility for trading losses
> -  Use at your own risk
> -  Always do your own research (DYOR)

**Cryptocurrency trading involves high risk. Do not invest more than you can afford to lose.**

---

## Author

Created with by **wahyuNurahmadinuh**

- GitHub: [@wahyuNurahmadinuh](https://github.com/wahyuNurahmadinuh)
- Email: wahyunurahmagaming@gmail.com

---


<div align="center">

**If this project helps you, please give it a star!**

Made with Python and Machine Learning

</div>
