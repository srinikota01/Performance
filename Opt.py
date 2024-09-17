import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Load JMeter CSV Results
# Assuming the CSV contains fields like timestamp, label, response_time, latency, error, throughput, etc.
# You can generate the CSV from JMeter using the Simple Data Writer Listener.
df = pd.read_csv('jmeter_results.csv')

# Step 2: Basic Analysis of Key Metrics

# Calculate summary statistics for Response Time, Throughput, Errors
response_time_mean = df['response_time'].mean()
response_time_90th = np.percentile(df['response_time'], 90)
error_rate = (df['error'].sum() / len(df)) * 100  # Error rate in percentage
throughput_mean = df['throughput'].mean()

print(f"Mean Response Time: {response_time_mean} ms")
print(f"90th Percentile Response Time: {response_time_90th} ms")
print(f"Error Rate: {error_rate:.2f}%")
print(f"Mean Throughput: {throughput_mean} requests/sec")

# Step 3: Detect Anomalies using Isolation Forest (Anomaly Detection)

# Isolation Forest for detecting outliers in response time
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% anomaly rate
df['anomaly'] = iso_forest.fit_predict(df[['response_time', 'throughput', 'latency']])

# Mark the anomalies
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Step 4: Root Cause Analysis on Anomalies

# Analyze response time and errors for anomalies
anomalies = df[df['anomaly'] == 'Anomaly']
if not anomalies.empty:
    print(f"\nNumber of anomalies detected: {len(anomalies)}")
    # Analyze the common causes for anomalies
    high_response_anomalies = anomalies[anomalies['response_time'] > response_time_90th]
    error_anomalies = anomalies[anomalies['error'] == 1]

    print(f"Anomalies with High Response Time (>90th percentile): {len(high_response_anomalies)}")
    print(f"Anomalies with Errors: {len(error_anomalies)}")

    # Analyze by labels (API endpoints)
    api_anomaly_counts = anomalies['label'].value_counts()
    print("\nTop API Endpoints with Anomalies:")
    print(api_anomaly_counts.head())

    # Plot anomaly detection
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df['response_time'], c=(df['anomaly'] == 'Anomaly'), cmap='coolwarm', label='Response Time')
    plt.axhline(y=response_time_90th, color='r', linestyle='--', label='90th Percentile Response Time')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (ms)')
    plt.title('Response Time with Anomalies')
    plt.legend()
    plt.show()

else:
    print("\nNo anomalies detected.")


# Step 5: Insights and Recommendations

def provide_recommendations(df):
    recommendations = []

    # Case 1: High Response Time
    if response_time_mean > response_time_90th:
        recommendations.append(
            "Optimize backend services or scale your API server resources. High response time detected.")

    # Case 2: High Error Rate
    if error_rate > 5:  # Example threshold for errors
        recommendations.append(
            f"High error rate of {error_rate:.2f}% detected. Investigate server logs for database or timeout issues.")

    # Case 3: Low Throughput
    if throughput_mean < 10:  # Example threshold for throughput
        recommendations.append(
            f"Throughput is below acceptable levels ({throughput_mean} req/sec). Consider horizontal scaling or optimizing code.")

    if recommendations:
        print("\nRecommendations for Optimizing Performance:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("\nPerformance looks good. No critical issues found.")


provide_recommendations(df)
