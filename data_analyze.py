import pandas as pd

# === Загрузка данных ===
transactions = pd.read_parquet('transaction_fraud_data.parquet', engine='fastparquet')
exchange = pd.read_parquet('historical_currency_exchange.parquet', engine='fastparquet').set_index('date')

# === Преобразуем timestamp ===
if not pd.api.types.is_datetime64_any_dtype(transactions['timestamp']):
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])

# === 1. Доля всех мошеннических транзакций ===
fraud_share = transactions['is_fraud'].mean()
print(f"Доля мошеннических транзакций: {fraud_share:.4f}")

# === 2. Топ-5 стран по количеству мошеннических транзакций ===
top_countries = (
    transactions[transactions['is_fraud'] == True]['country']
    .value_counts()
    .head(5)
    .index.tolist()
)
print(",".join(top_countries))

# === 3. Доля мошенничества у high risk vendor ===
high_risk = transactions[transactions['is_high_risk_vendor'] == True]
high_risk_fraud_share = high_risk['is_fraud'].mean()
print(f"Доля мошенничества у high risk vendor: {high_risk_fraud_share:.4f}")

# === 4. Среднее число транзакций на клиента в час ===
transactions['hour'] = transactions['timestamp'].dt.floor('H')
client_hour_counts = transactions.groupby(['customer_id', 'hour']).size()
avg_tx_per_client_per_hour = client_hour_counts.mean()
print(f"Среднее число транзакций на клиента в час: {avg_tx_per_client_per_hour:.4f}")

# === 5. Город с наибольшей средней суммой транзакций (исключая Unknown City) ===
city_avg = transactions.groupby('city')['amount'].mean().sort_values(ascending=False)
# Исключаем Unknown City
valid_cities = city_avg[~city_avg.index.str.lower().str.contains("unknown")]
city_with_max_avg = valid_cities.idxmax()
print(city_with_max_avg)

# === 6. Город с наибольшим средним чеком среди fast_food (исключая Unknown City) ===
transactions['vendor_type_clean'] = transactions['vendor_type'].str.lower().str.replace(" ", "_")
fast_food_df = transactions[transactions['vendor_type_clean'] == 'fast_food']

if not fast_food_df.empty:
    fast_food_city_avg = fast_food_df.groupby('city')['amount'].mean().sort_values(ascending=False)
    valid_cities = fast_food_city_avg[~fast_food_city_avg.index.str.lower().str.contains("unknown")]
    result_city = valid_cities.idxmax()
    print(result_city)
else:
    print("Нет данных по fast_food в vendor_type")

# === 7. Средний размер немошеннической транзакции в USD ===
non_fraud = transactions[transactions['is_fraud'] == False].copy()
non_fraud['date'] = non_fraud['timestamp'].dt.date.astype(str)

def convert_to_usd(row):
    date = row['date']
    curr = row['currency']
    amount = row['amount']
    if curr == 'USD':
        return amount
    try:
        rate = exchange.loc[date, curr]
        return amount / rate if rate and rate != 0 else None
    except Exception:
        return None

non_fraud['amount_usd'] = non_fraud.apply(convert_to_usd, axis=1)
mean_usd = non_fraud['amount_usd'].dropna().mean()
print(f"Средний размер немошеннической транзакции в USD: {mean_usd:.2f}")

# === 8. Стандартное отклонение немошеннических транзакций в USD ===
std_usd = non_fraud['amount_usd'].dropna().std()
print(f"Стандартное отклонение немошеннических транзакций в USD: {std_usd:.2f}")

# === 9–10. Аналогично для мошеннических транзакций ===
fraud = transactions[transactions['is_fraud'] == True].copy()
fraud['date'] = fraud['timestamp'].dt.date.astype(str)
fraud['amount_usd'] = fraud.apply(convert_to_usd, axis=1)

mean_usd_fraud = fraud['amount_usd'].dropna().mean()
print(f"Средний размер мошеннической транзакции в USD: {mean_usd_fraud:.2f}")

std_usd_fraud = fraud['amount_usd'].dropna().std()
print(f"Стандартное отклонение мошеннических транзакций в USD: {std_usd_fraud:.2f}")

# === 11. Количество клиентов с потенциально опасным поведением ===
if 'last_hour_activity.unique_merchants' in transactions.columns:
    transactions['unique_merchants'] = transactions['last_hour_activity.unique_merchants']
    medians = transactions.groupby('customer_id')['unique_merchants'].median()
    q95 = medians.quantile(0.95)
    risky_count = (medians > q95).sum()
    print(risky_count)
else:
    raise ValueError("Столбец 'last_hour_activity.unique_merchants' не найден!")
