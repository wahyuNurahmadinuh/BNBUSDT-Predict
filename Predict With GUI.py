import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import warnings
warnings.filterwarnings('ignore')

class BNBPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BNB Price Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e2e')
        
        # Variables
        self.symbol = tk.StringVar(value='BNBUSDT')
        self.interval = tk.StringVar(value='1d')
        self.limit = tk.IntVar(value=1000)
        self.df = None
        self.model = None
        self.is_training = False
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2d2d44', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚀 BNB Price Predictor", 
                               font=('Arial', 24, 'bold'), 
                               bg='#2d2d44', fg='#f9e2af')
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#1e1e2e')
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='#2d2d44', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.setup_controls(left_panel)
        
        # Right panel - Charts and Results
        right_panel = tk.Frame(main_container, bg='#1e1e2e')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_results(right_panel)
        
    def setup_controls(self, parent):
        # Configuration section
        config_label = tk.Label(parent, text="Konfigurasi", 
                               font=('Arial', 14, 'bold'), 
                               bg='#2d2d44', fg='#cdd6f4')
        config_label.pack(pady=(20, 10))
        
        # Symbol
        self.create_input_field(parent, "Symbol:", self.symbol)
        
        # Interval
        interval_frame = tk.Frame(parent, bg='#2d2d44')
        interval_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(interval_frame, text="Interval:", 
                bg='#2d2d44', fg='#cdd6f4', font=('Arial', 10)).pack(anchor='w')
        interval_combo = ttk.Combobox(interval_frame, textvariable=self.interval, 
                                     values=['1m', '5m', '15m', '1h', '4h', '1d', '1w'],
                                     state='readonly', width=25)
        interval_combo.pack(fill='x', pady=2)
        
        # Limit
        self.create_input_field(parent, "Data Limit:", self.limit)
        
        # Buttons
        btn_frame = tk.Frame(parent, bg='#2d2d44')
        btn_frame.pack(fill='x', padx=20, pady=20)
        
        self.fetch_btn = tk.Button(btn_frame, text="📥 Fetch Data", 
                                   command=self.fetch_data_thread,
                                   bg='#89b4fa', fg='#1e1e2e', 
                                   font=('Arial', 11, 'bold'),
                                   relief='flat', cursor='hand2')
        self.fetch_btn.pack(fill='x', pady=5)
        
        self.train_btn = tk.Button(btn_frame, text="🎯 Train Model", 
                                   command=self.train_model_thread,
                                   bg='#a6e3a1', fg='#1e1e2e', 
                                   font=('Arial', 11, 'bold'),
                                   relief='flat', cursor='hand2', state='disabled')
        self.train_btn.pack(fill='x', pady=5)
        
        self.predict_btn = tk.Button(btn_frame, text="🔮 Predict Price", 
                                     command=self.predict_price_thread,
                                     bg='#f9e2af', fg='#1e1e2e', 
                                     font=('Arial', 11, 'bold'),
                                     relief='flat', cursor='hand2', state='disabled')
        self.predict_btn.pack(fill='x', pady=5)
        
        # Status
        status_label = tk.Label(parent, text="Status", 
                               font=('Arial', 12, 'bold'), 
                               bg='#2d2d44', fg='#cdd6f4')
        status_label.pack(pady=(20, 10))
        
        self.status_text = scrolledtext.ScrolledText(parent, height=10, 
                                                     bg='#181825', fg='#cdd6f4',
                                                     font=('Courier', 9),
                                                     relief='flat')
        self.status_text.pack(fill='both', expand=True, padx=20, pady=5)
        
    def create_input_field(self, parent, label, variable):
        frame = tk.Frame(parent, bg='#2d2d44')
        frame.pack(fill='x', padx=20, pady=5)
        tk.Label(frame, text=label, bg='#2d2d44', fg='#cdd6f4', 
                font=('Arial', 10)).pack(anchor='w')
        entry = tk.Entry(frame, textvariable=variable, 
                        bg='#181825', fg='#cdd6f4', 
                        font=('Arial', 10), relief='flat',
                        insertbackground='#cdd6f4')
        entry.pack(fill='x', pady=2, ipady=5)
        
    def setup_results(self, parent):
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)
        
        # Style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d44', 
                       foreground='#cdd6f4', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#89b4fa')],
                 foreground=[('selected', '#1e1e2e')])
        
        # Chart tab
        chart_frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(chart_frame, text='📊 Charts')
        
        self.figure = Figure(figsize=(10, 6), facecolor='#1e1e2e')
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Results tab
        results_frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(results_frame, text='📈 Results')
        
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                      bg='#181825', fg='#cdd6f4',
                                                      font=('Courier', 11),
                                                      relief='flat')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
    def log_status(self, message):
        self.status_text.insert('end', f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see('end')
        self.root.update()
        
    def fetch_data_thread(self):
        threading.Thread(target=self.fetch_data, daemon=True).start()
        
    def fetch_data(self):
        try:
            self.fetch_btn.config(state='disabled')
            self.log_status("Fetching historical data from Binance API...")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': self.symbol.get(),
                'interval': self.interval.get(),
                'limit': self.limit.get()
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self.df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            self.df = self.df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.df[col] = self.df[col].astype(float)
            
            # Validate and clean
            self.df = self.df.dropna()
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
            self.log_status(f"✓ Successfully fetched {len(self.df)} data points")
            self.train_btn.config(state='normal')
            
            # Plot data
            self.plot_historical_data()
            
        except Exception as e:
            self.log_status(f"✗ Error fetching data: {str(e)}")
            messagebox.showerror("Error", f"Failed to fetch data:\n{str(e)}")
        finally:
            self.fetch_btn.config(state='normal')
            
    def plot_historical_data(self):
        self.figure.clear()
        
        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.df['timestamp'], self.df['close'], 
                color='#89b4fa', linewidth=2, label='Close Price')
        ax1.fill_between(self.df['timestamp'], self.df['low'], self.df['high'], 
                        alpha=0.3, color='#89b4fa')
        ax1.set_xlabel('Date', color='#cdd6f4')
        ax1.set_ylabel('Price (USDT)', color='#cdd6f4')
        ax1.set_title('Historical Price', color='#cdd6f4', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.2, color='#cdd6f4')
        ax1.set_facecolor('#181825')
        ax1.tick_params(colors='#cdd6f4')
        
        ax2 = self.figure.add_subplot(212)
        ax2.bar(self.df['timestamp'], self.df['volume'], 
               alpha=0.6, color='#a6e3a1')
        ax2.set_xlabel('Date', color='#cdd6f4')
        ax2.set_ylabel('Volume', color='#cdd6f4')
        ax2.set_title('Trading Volume', color='#cdd6f4', fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#cdd6f4')
        ax2.set_facecolor('#181825')
        ax2.tick_params(colors='#cdd6f4')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def train_model_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()
        
    def train_model(self):
        try:
            self.train_btn.config(state='disabled')
            self.log_status("Training model...")
            
            # Prepare data
            self.df['next_close'] = self.df['close'].shift(-1)
            df_train = self.df[:-1].copy()
            
            X = df_train[['close', 'open', 'high', 'low', 'volume']].values
            y = df_train['next_close'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            self.log_status(f"✓ Model trained successfully")
            self.log_status(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            
            # Display results
            results = f"""
{'='*60}
MODEL TRAINING RESULTS
{'='*60}

Dataset Information:
  • Total data points: {len(self.df)} 
  • Training samples: {len(X_train)}
  • Testing samples: {len(X_test)}

Model Performance:
  • Mean Squared Error (MSE): {mse:.2f}
  • Root Mean Squared Error (RMSE): {rmse:.2f}
  • Mean Absolute Error (MAE): {mae:.2f}
  • Mean Absolute Percentage Error (MAPE): {mape:.2f}%

Model Coefficients:
  • Close: {self.model.coef_[0]:.6f}
  • Open: {self.model.coef_[1]:.6f}
  • High: {self.model.coef_[2]:.6f}
  • Low: {self.model.coef_[3]:.6f}
  • Volume: {self.model.coef_[4]:.10f}
  • Intercept: {self.model.intercept_:.2f}

{'='*60}
"""
            self.results_text.delete(1.0, 'end')
            self.results_text.insert(1.0, results)
            
            # Plot predictions
            self.plot_predictions(y_test, y_pred)
            
            self.predict_btn.config(state='normal')
            
        except Exception as e:
            self.log_status(f"✗ Error training model: {str(e)}")
            messagebox.showerror("Error", f"Failed to train model:\n{str(e)}")
        finally:
            self.train_btn.config(state='normal')
            
    def plot_predictions(self, y_test, y_pred):
        self.figure.clear()
        
        ax1 = self.figure.add_subplot(121)
        ax1.plot(y_test, label='Actual', marker='o', alpha=0.7, 
                linewidth=2, color='#89b4fa')
        ax1.plot(y_pred, label='Predicted', marker='x', alpha=0.7, 
                linewidth=2, color='#f9e2af')
        ax1.set_xlabel('Sample', color='#cdd6f4')
        ax1.set_ylabel('Price (USDT)', color='#cdd6f4')
        ax1.set_title('Actual vs Predicted', color='#cdd6f4', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.2, color='#cdd6f4')
        ax1.set_facecolor('#181825')
        ax1.tick_params(colors='#cdd6f4')
        
        ax2 = self.figure.add_subplot(122)
        ax2.scatter(y_test, y_pred, alpha=0.5, color='#a6e3a1')
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Price (USDT)', color='#cdd6f4')
        ax2.set_ylabel('Predicted Price (USDT)', color='#cdd6f4')
        ax2.set_title('Scatter Plot', color='#cdd6f4', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.2, color='#cdd6f4')
        ax2.set_facecolor('#181825')
        ax2.tick_params(colors='#cdd6f4')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def predict_price_thread(self):
        threading.Thread(target=self.predict_price, daemon=True).start()
        
    def predict_price(self):
        try:
            self.predict_btn.config(state='disabled')
            self.log_status("Fetching real-time price...")
            
            # Get real-time price
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol.get()}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            current_price = float(response.json()['price'])
            
            self.log_status(f"Current price: {current_price:.2f} USDT")
            
            # Prepare features
            last_row = self.df.iloc[-1]
            features = np.array([[
                current_price,
                last_row['open'],
                last_row['high'],
                last_row['low'],
                last_row['volume']
            ]])
            
            # Predict
            predicted_price = self.model.predict(features)[0]
            change_pct = ((predicted_price - current_price) / current_price) * 100
            direction = "📈 UP" if change_pct > 0 else "📉 DOWN"
            
            self.log_status(f"Predicted price: {predicted_price:.2f} USDT ({direction})")
            
            # Display prediction
            prediction_result = f"""
{'='*60}
PRICE PREDICTION RESULT
{'='*60}

Current Status:
  • Symbol: {self.symbol.get()}
  • Current Price: ${current_price:.2f} USDT
  • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Prediction:
  • Next Price: ${predicted_price:.2f} USDT
  • Expected Change: {direction} {abs(change_pct):.2f}%
  • Price Difference: ${abs(predicted_price - current_price):.2f}

Input Features Used:
  • Close: {current_price:.2f}
  • Open: {last_row['open']:.2f}
  • High: {last_row['high']:.2f}
  • Low: {last_row['low']:.2f}
  • Volume: {last_row['volume']:.2f}

{'='*60}

⚠️  DISCLAIMER:
This prediction is based on historical data and linear regression.
Cryptocurrency markets are highly volatile and unpredictable.
Always do your own research before making investment decisions.

{'='*60}
"""
            self.results_text.delete(1.0, 'end')
            self.results_text.insert(1.0, prediction_result)
            
            # Plot prediction
            self.plot_prediction_chart(current_price, predicted_price)
            
        except Exception as e:
            self.log_status(f"✗ Error predicting price: {str(e)}")
            messagebox.showerror("Error", f"Failed to predict price:\n{str(e)}")
        finally:
            self.predict_btn.config(state='normal')
            
    def plot_prediction_chart(self, current_price, predicted_price):
        self.figure.clear()
        
        recent_df = self.df.tail(30).copy()
        
        ax = self.figure.add_subplot(111)
        ax.plot(recent_df['timestamp'], recent_df['close'], 
               marker='o', label='Historical Price', 
               linewidth=2, color='#89b4fa')
        
        next_date = recent_df['timestamp'].iloc[-1] + timedelta(days=1)
        ax.plot([recent_df['timestamp'].iloc[-1], next_date], 
               [current_price, predicted_price],
               marker='*', markersize=15, linestyle='--', 
               linewidth=2, color='#f38ba8', label='Prediction')
        
        ax.axhline(y=current_price, color='#a6e3a1', 
                  linestyle=':', alpha=0.5, label='Current Price')
        ax.axhline(y=predicted_price, color='#f9e2af', 
                  linestyle=':', alpha=0.5, label='Predicted Price')
        
        ax.set_xlabel('Date', color='#cdd6f4')
        ax.set_ylabel('Price (USDT)', color='#cdd6f4')
        ax.set_title('Price Prediction - Last 30 Days', 
                    color='#cdd6f4', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2, color='#cdd6f4')
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = BNBPredictorGUI(root)
    root.mainloop()