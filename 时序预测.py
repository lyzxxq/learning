from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import random


class AutoProphet():
    def __init__(self, data, service_cfg):
        self.cfg = service_cfg
        self.data = self._data_process(data)
        self.model = None

    def _data_process(self, data):
        # 数据预处理
        processed_data = data.rename(columns={self.cfg['date']: 'ds', self.cfg['value']: 'y'})
        processed_data['ds'] = pd.to_datetime(processed_data['ds'], format='%Y%m%d')
        processed_data = processed_data[['ds', 'y']].dropna().copy()
        # 异常值检测（Isolation Forest）
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(processed_data[['y']])
        processed_data['outlier'] = model.predict(processed_data[['y']]) == -1
        # 删除异常值
        processed_data = processed_data[~processed_data['outlier']].copy()
        processed_data = processed_data.drop(columns=['outlier'])
        return processed_data

    def _output_process(self, prediction):
        # 格式化预测结果
        prediction = prediction.copy()  
        prediction['predict_date'] = prediction['ds'].dt.strftime('%Y%m%d')
        prediction['predict_daily_change_amount'] = prediction['yhat'].diff().fillna(0).astype(int)
        return prediction[['predict_date', 'yhat']].to_dict('records')

    def predict(self, prediction_weeks):
        recent_data = self.data[self.data['ds'] >= (self.data['ds'].max() - pd.DateOffset(years=1))]
        if len(recent_data) < 52:  # 至少需要一年的数据
            recent_data = self.data
        self.model = Prophet(
            growth=self.cfg.get('growth', 'linear'),
            seasonality_mode=self.cfg.get('seasonality_mode', 'additive'),
            weekly_seasonality=self.cfg.get('weekly_seasonality', True),
            yearly_seasonality=self.cfg.get('yearly_seasonality', True),
            holidays=self.cfg.get('holidays', None),
            changepoint_prior_scale=0.01,  # 调整敏感度，降低过拟合
            seasonality_prior_scale=10.0
        )

        # 使用插值处理缺失值
        self.data['y'] = self.data['y'].interpolate()

        self.model.fit(recent_data)

        # 按天预测
        prediction_days = prediction_weeks * 7
        future_daily = self.model.make_future_dataframe(periods=prediction_days, include_history=True)
        daily_prediction = self.model.predict(future_daily)

        weekly_result = []
        if prediction_weeks:
            future_weekly = self.model.make_future_dataframe(periods=prediction_weeks, freq='W-SUN')
            weekly_prediction = self.model.predict(future_weekly).tail(prediction_weeks)
            weekly_result = weekly_prediction[['ds', 'yhat']].to_dict('records')
            
        daily_result = {
            'all_predictions': daily_prediction,  # 包含训练集和测试集
            'daily_predictions_test': self._output_process(daily_prediction.tail(prediction_days)),
            'daily_predictions': self._output_process(daily_prediction),
            'weekly_predictions': weekly_result
        }
        return daily_result


def calculate_smape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    denominator = np.where(denominator == 0, 1, denominator)  # 将分母为零的情况替换为1
    smape = np.mean(numerator / denominator) * 100  # 转化为百分比

    return smape

# 读取数据
df = pd.read_csv("D:\\Files\\时序预测\\obs_usage.csv", encoding="gbk")
df['collect_date'] = pd.to_datetime(df['collect_date'], format='%Y%m%d', errors='coerce')
df = df[df['collect_date'] >= '2024-02-20'].copy()
df.sort_values(by=['tenant_user_name', 'cluster', 'collect_date'], inplace=True)
# 获取唯一组合
unique_combinations = df.groupby(['tenant_user_name', 'cluster']).size().reset_index()[['tenant_user_name', 'cluster']]
unique_combinations = list(unique_combinations.itertuples(index=False))
total_combinations = len(unique_combinations)
print(f"Total combinations: {total_combinations}")
# 配置项
service_cfg = {
    'date': 'collect_date',
    'value': 'hdd',
    'growth': 'linear',
    'seasonality_mode': 'additive',
    'weekly_seasonality': True,
    'yearly_seasonality': True,
    'holidays': None,
    'out_data': {'tenant_user_name': '', 'cluster': ''},
}
all_predictions = []
all_smape_scores = []
for combination in unique_combinations:
    tenant_user_name, cluster = combination
    print(f"Processing combination: {tenant_user_name}, {cluster}")

    df_filtered = df[(df['tenant_user_name'] == tenant_user_name) & (df['cluster'] == cluster)].copy()
    if len(df_filtered) < 168:
        print("Insufficient data, skipping this combination.")
        continue

    test_weeks = 12
    test_days = test_weeks * 7
    split_date = df_filtered['collect_date'].max() - pd.DateOffset(weeks=test_weeks)
    train_df = df_filtered[df_filtered['collect_date'] <= split_date].copy()
    test_df = df_filtered[df_filtered['collect_date'] > split_date].copy()

    if len(train_df) < 12:
        print("Insufficient training data, skipping this combination.")
        continue

    # 初始化并进行预测
    auto_prophet = AutoProphet(train_df, service_cfg)
    predictions = auto_prophet.predict(prediction_weeks=test_weeks)

    # 保存预测结果
    all_predictions.append({
        'tenant_user_name': tenant_user_name,
        'cluster': cluster,
        'daily_predictions': predictions['daily_predictions'],
        'weekly_predictions': predictions.get('weekly_predictions', [])
    })

    predicted_daily_values = [pred['yhat'] for pred in predictions['daily_predictions_test']]
    actual_daily_values = test_df['hdd'].fillna(0).astype(float).tolist()
    if len(actual_daily_values) < len(predicted_daily_values):
        print(f"Insufficient actual data for {tenant_user_name} - {cluster}. Skipping SMAPE calculation.")
        continue

    actual_daily_values = actual_daily_values[-len(predicted_daily_values):]  # 截取最近数据以匹配预测值
    if len(actual_daily_values) == len(predicted_daily_values):
        smape = calculate_smape(actual_daily_values, predicted_daily_values)
        all_smape_scores.append({
            'tenant_user_name': tenant_user_name,
            'cluster': cluster,
            'SMAPE': smape
        })
        print(f"SMAPE for {tenant_user_name} - {cluster}: {smape:.2f}%")
    else:
        print(f"Length mismatch for {tenant_user_name} - {cluster}. Skipping SMAPE calculation.")

    # 转化为 DataFrame
    predicted_daily = pd.DataFrame(predictions['daily_predictions'])
    predicted_daily['predict_date'] = pd.to_datetime(predicted_daily['predict_date'], format='%Y%m%d')
    predicted_daily.rename(columns={'predict_daily_change_amount': 'predicted'}, inplace=True)
    full_predictions = predictions['all_predictions']  # 包括历史数据的完整预测结果
    full_predictions['ds'] = pd.to_datetime(full_predictions['ds'])  # 确保日期格式一致

    # 提取训练集预测值
    train_predictions = full_predictions[full_predictions['ds'].isin(train_df['collect_date'])]

    # 提取测试集预测值
    test_predictions = full_predictions[full_predictions['ds'].isin(test_df['collect_date'])]

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['collect_date'], train_df['hdd'], label='Training Data', color='black')
    plt.plot(test_df['collect_date'], test_df['hdd'], label='Testing Data', color='brown')
    plt.plot(train_predictions['ds'], train_predictions['yhat'], label='Training Predictions', color='steelblue', linestyle='--')
    plt.plot(test_predictions['ds'], test_predictions['yhat'], label='Testing Predictions', color='orange', linestyle='--')
    plt.fill_between(test_predictions['ds'], test_predictions['yhat_lower'], test_predictions['yhat_upper'],
                     color='orange', alpha=0.2, label='Confidence Interval (Test)')
    plt.title(f"Time Series Forecast for {tenant_user_name} - {cluster}")
    plt.text(
        x=0.98,
        y=0.05,
        s=f"SMAPE: {smape:.2f}%",
        fontsize=12,
        # color='blue',
        ha='right',
        va='top',
        transform=plt.gca().transAxes
    )
    plt.legend()
    plt.grid()
    output_dir = "D:\\Files\\时序预测\\Predict_SMAPE"
    os.makedirs(output_dir, exist_ok=True)
    image_name = f"Prophet_{tenant_user_name}_{cluster}.png"
    image_path = os.path.join(output_dir, image_name)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close() 

smape_df = pd.DataFrame(all_smape_scores)
output_file = "D:\\Files\\时序预测\\Predict_SMAPE\\smape_scores.csv"
smape_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"SMAPE scores saved to {output_file}")

overall_smape = smape_df['SMAPE'].mean()
print(f"Overall SMAPE: {overall_smape:.2f}%")
