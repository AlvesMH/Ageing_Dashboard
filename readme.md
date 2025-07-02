# 🧓 OECD Ageing‑Risk Workbench

A dynamic and interactive Streamlit dashboard for visualizing and forecasting demographic and fiscal vulnerabilities among OECD countries — powered by World Bank data and ARIMA forecasting.

![Dashboard Overview](https://user-images.githubusercontent.com/your-placeholder/preview.png)

---

## ✨ Features

- 📊 **Composite Vulnerability Score** based on:
  - Population 65+ (%)
  - Fertility Rate
  - Health Expenditure (% of GDP)
  - GDP per Capita Growth
  - Old‑Age Dependency Ratio
  - Public Debt (% of GDP) — from **World Bank**

- 🔮 **ARIMA Forecasting** to project trends up to 2050

- 🎛️ Customizable weight presets:
  - Balanced, Fiscal, Health, or Custom Weights

- 📈 Country-specific time series & radar plots

- 🌍 Interactive world map of OECD countries

- 📥 Download buttons for snapshot & full panel CSVs

---

## 🛠️ Tech Stack

- **Language**: Python 3
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Sources**: [World Bank API](https://data.worldbank.org/)
- **Forecasting**: ARIMA models (`statsmodels`)
- **Visualization**: Plotly

---

## 🚀 Live Demo

👉 [View Deployed App on Render](https://your-app-url.onrender.com)

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/alvesmh/oecd-aging-workbench.git
cd oecd-aging-workbench
```

(Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run dash4.py
```

🧮 Methodology
Each country receives a composite vulnerability score based on weighted indicators. You can customize these weights or use predefined themes. The dashboard uses ARIMA modeling for time series forecasts, allowing future projections of key indicators up to year 2050.

📊 Example Use Cases
* Comparative policy analysis of aging risks

* Long-term fiscal vulnerability forecasting

* Demographic sustainability visualization

📁 Project Structure
```bash
📦 oecd-ageing-workbench
├── oecd_aging_dashboard.py # Main Streamlit app
├── requirements.txt        # Dependencies
├── setup.sh                # Render deployment config
└── README.md               # Project documentation

🙌 Acknowledgements
World Bank Open Data

Streamlit for rapid app development

Plotly for powerful visualizations

OECD country codes and demographic concepts

📄 License
This project is licensed under the MIT License.