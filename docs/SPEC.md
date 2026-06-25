The following is a task for me: 

**Deephaven**, a highly efficient real-time data engine that’s fantastic for time-series analysis and live dataframe operations. Working locally with Python, containerization, and declarative UI frameworks will fit perfectly into your standard wheelhouse of orchestrating robust data pipelines.
Here is a straightforward path to get your stock market "what-if" engine running locally.
### 1. Where to Get Historical Stock Data (For Free)
* **yfinance (Yahoo Finance):** This is the absolute best starting point. It’s a Python library that pulls historical market data from Yahoo Finance directly into Pandas DataFrames, which Deephaven can natively ingest. There are no API keys, no rate limits for simple historical pulls, and it is 100% free.
* **Alpaca or Alpha Vantage:** Keep these in your back pocket. They offer generous free tiers for historical data and are great options if you eventually decide to stream live market data into your platform.
### 2. Setting Up Deephaven on a Local Machine
To keep everything isolated and running cleanly on your machine, Docker Compose is the standard and most frictionless method. It provisions the Deephaven engine and its web-based IDE simultaneously.
1. Create a directory for your project and add a docker-compose.yml file. You can pull the standard Python base image provided by Deephaven:
```yaml
version: '3'
services:
deephaven:
image: ghcr.io/deephaven/server:latest
ports:
- "10000:10000"
volumes:
- ./data:/data

```
2. Run docker compose up -d in your terminal.
3. Open your browser and navigate to http://localhost:10000/ide. You now have a full IDE ready to execute Python scripts against the query engine.
### 3. Creating "What-If" Dashboards
Deephaven recently introduced deephaven.ui, a reactive, component-based UI framework driven entirely by Python. You won't need to write any front-end code (like JavaScript or CSS) to build your dashboard.
You can write a script directly in the Deephaven IDE to fetch your Intel data, run your volatility scenario, and render the controls. Here is a conceptual example to get you started:
```python
import yfinance as yf
import deephaven.pandas as dhpd
from deephaven import ui
# 1. Fetch 10 years of historical Intel data
intel_df = yf.download('INTC', start='2014-01-01', end='2024-01-01')
intel_df.reset_index(inplace=True)
# 2. Convert Pandas to a live Deephaven table
dh_table = dhpd.to_table(intel_df)
# 3. Define the What-If logic
def calculate_profit(buy_price, sell_price):
# Filter the table and calculate simulated profit based on volatility
return dh_table.where(f"Close <= {buy_price}").update(f"Simulated_Profit = {sell_price} - Close")
# 4. Construct the UI Dashboard
@ui.component
def stock_dashboard():
# Set up interactive state for buy and sell prices
buy_price, set_buy_price = ui.use_state(25.0)
sell_price, set_sell_price = ui.use_state(45.0)
# Reactively update the table when sliders change
result_table = ui.use_memo(lambda: calculate_profit(buy_price, sell_price), [buy_price, sell_price])
# Layout the dashboard elements
return ui.flex(
ui.slider(value=buy_price, on_change=set_buy_price, min=10.0, max=50.0),
ui.slider(value=sell_price, on_change=set_sell_price, min=20.0, max=100.0),
ui.table(result_table)
)
# Render the dashboard in the IDE
my_dashboard = stock_dashboard()
```

This setup gets your historical data flowing, gives you a playground for your volatility logic, and spins up a localized UI.
To help tailor this simulation as you enhance it, are you looking to run these scenarios purely as static backtests, or do you plan to eventually feed live, ticking market data into the engine?
## Implementation Steps for Coding Agent
### 1. Initialization & Version Control
* Initialize a Git repository.
* Create a .gitignore file to explicitly exclude the data/ directory, __pycache__/, and OS files like .DS_Store.
### 2. Dependency Management & Docker Setup
The standard Deephaven image must be extended to include yfinance.
* Create requirements.txt containing yfinance and pandas.
* Create a Dockerfile that uses ghcr.io/deephaven/server:latest as the base image. Add instructions to copy the requirements.txt and install the dependencies via pip.
* Create docker-compose.yml to:
* Build the service using the custom Dockerfile.
* Expose port 10000:10000.
* Map volumes accurately: ./data:/data and ./scripts:/data/storage/notebooks.
### 3. Core Logic & Dashboard Generation
Generate the primary script in scripts/what_if_dashboard.py. The script must execute the following flow:
1. **Data Acquisition:** Use yfinance to download 10 years of historical data for Intel (INTC). Reset the pandas DataFrame index.
2. **Data Integration:** Convert the Pandas DataFrame into a Deephaven live table using deephaven.pandas.to_table.
3. **Scenario Engine:** Write a function that takes buy_price and sell_price as parameters, filters the Deephaven table for days where the closing price was less than or equal to the buy price, and calculates a new column Simulated_Profit = sell_price - Close.
4. **UI Construction:** Utilize @ui.component from deephaven.ui.
* Set up state variables for buy_price and sell_price using ui.use_state.
* Bind these states to interactive sliders (ui.slider).
* Wrap the scenario calculation in ui.use_memo so the resulting table updates reactively when the sliders change.
* Render the layout returning a ui.flex container holding the sliders and the results table.
### 4. Verification
Ensure the Docker configuration is ready to run via docker compose up -d without requiring external API keys or cloud service dependencies.
"""
with open('deephaven_project_spec.md', 'w') as f:
f.write(markdown_content)
print("File generated successfully.")
```
Your markdown file is ready
[file-tag: code-generated-file-0-1782245574632345543]

I have structured the file as a comprehensive prompt and specification document. It outlines the exact tech stack, directory structure, Docker requirements, and Python logic necessary for your coding agent to build the local containerized environment and dashboard completely from scratch.