import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出文件夹
output_folder = "极端天气分析结果"
os.makedirs(output_folder, exist_ok=True)

# 创建一个文本文件来保存分析结果
result_file = os.path.join(output_folder, "分析结果.txt")

# 修改 print_and_write 函数
def print_and_write(text, file=None):
    print(text)
    if file:
        file.write(text + "\n")

# 读取数据
file_path = r"D:\AAAAAAAA\SHU-FIle\华为杯\2024年中国研究生数学建模竞赛赛题\D题数据\数据output\降雨量\降雨量csv\clustered_rainfall_data_with_temperature.csv"
df = pd.read_csv(file_path)

# 检查数据
print_and_write("数据前几行：")
print_and_write(str(df.head()))
print_and_write("\n列名：")
print_and_write(str(df.columns))

# 定义自变量和因变量
X = df[['RASTERVALU', 'Avg_Precip', 'Refined_Temperature']]
y_vars = ['Frequency1', 'Frequency2', 'Frequenc3']  # 更新为实际的列名

# 中文列名映射
column_names = {
    'RASTERVALU': '海拔高度',
    'Avg_Precip': '平均降雨量',
    'Refined_Temperature': '平均气温',
    'Frequency1': '暴雨频率',
    'Frequency2': '大暴雨频率',
    'Frequenc3': '特大暴雨频率'
}

# 标准化自变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义分析函数
def analyze_extreme_weather(X, y, model_type='linear'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# 定义偏相关函数
def partial_correlation(x, y, z):
    xy_corr, _ = pearsonr(x, y)
    xz_corr, _ = pearsonr(x, z.iloc[:, 0])  # 只使用第一列
    yz_corr, _ = pearsonr(y, z.iloc[:, 0])  # 只使用第一列
    return (xy_corr - xz_corr * yz_corr) / (np.sqrt(1 - xz_corr**2) * np.sqrt(1 - yz_corr**2))

# 分析每种暴雨类型
with open(result_file, "w", encoding="utf-8") as f:
    # 将数据检查结果写入文件
    f.write("数据前几行：\n")
    f.write(str(df.head()) + "\n\n")
    f.write("列名：\n")
    f.write(str(df.columns) + "\n\n")

    for y_var in y_vars:
        print_and_write(f"\n分析 {column_names[y_var]} 的影响因素", f)
        print_and_write("="*50, f)
        
        # 多元线性回归模型
        linear_model, linear_mse, linear_r2 = analyze_extreme_weather(X_scaled, df[y_var])
        print_and_write("多元线性回归模型结果：", f)
        print_and_write(f"均方误差: {linear_mse:.4f}", f)
        print_and_write(f"R方: {linear_r2:.4f}", f)
        print_and_write("系数:", f)
        for feature, coef in zip(X.columns, linear_model.coef_):
            print_and_write(f"{column_names[feature]}: {coef:.4f}", f)
        
        print_and_write("\n注释：", f)
        print_and_write("1. 均方误差（MSE）越小，模型预测越准确。", f)
        print_and_write("2. R方值介于0和1之间，越接近1表示模型解释力越强。", f)
        print_and_write("3. 系数表示每个特征对因变量的影响程度，正值表示正相关，负值表示负相关。", f)
        
        # 随机森林模型
        rf_model, rf_mse, rf_r2 = analyze_extreme_weather(X_scaled, df[y_var], model_type='random_forest')
        print_and_write("\n随机森林模型结果：", f)
        print_and_write(f"均方误差: {rf_mse:.4f}", f)
        print_and_write(f"R方: {rf_r2:.4f}", f)
        
        print_and_write("\n注释：", f)
        print_and_write("1. 随机森林模型通常能够捕捉到更复杂的非线性关系。", f)
        print_and_write("2. 比较随机森林和线性回归的结果，可以判断数据关系的复杂程度。", f)
        
        # 特征重要性
        feature_importance = pd.DataFrame({'feature': [column_names[f] for f in X.columns], 'importance': rf_model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print_and_write("\n特征重要性：", f)
        print_and_write(str(feature_importance), f)
        
        print_and_write("\n注释：", f)
        print_and_write("1. 特征重要性反映了每个特征对预测结果的贡献程度。", f)
        print_and_write("2. 重要性越高的特征，对预测结果的影响越大。", f)
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'{column_names[y_var]}的特征重要性')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.savefig(os.path.join(output_folder, f"{y_var}_特征重要性.png"))
        plt.close()

        # 三维散点图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df['RASTERVALU'], df['Avg_Precip'], df['Refined_Temperature'], 
                             c=df[y_var], cmap='viridis', s=20)
        ax.set_xlabel('海拔高度')
        ax.set_ylabel('平均降雨量')
        ax.set_zlabel('平均气温')
        plt.colorbar(scatter, label=column_names[y_var])
        plt.title(f'海拔高度、平均降雨量和平均气温对{column_names[y_var]}的影响')
        plt.savefig(os.path.join(output_folder, f"{y_var}_3D散点图.png"))
        plt.close()

    # 相关性热图
    correlation_matrix = df[['RASTERVALU', 'Avg_Precip', 'Refined_Temperature'] + y_vars].corr()
    correlation_matrix.columns = [column_names.get(col, col) for col in correlation_matrix.columns]
    correlation_matrix.index = [column_names.get(idx, idx) for idx in correlation_matrix.index]

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('变量间相关性热图')
    plt.savefig(os.path.join(output_folder, "相关性热图.png"))
    plt.close()

    print_and_write("\n相关性热图注释：", f)
    print_and_write("1. 热图显示了各变量之间的相关系数。", f)
    print_and_write("2. 颜色越深表示相关性越强，红色表示正相关，蓝色表示负相关。", f)
    print_and_write("3. 对角线上的值总是1，表示变量与自身的完全相关。", f)

    # 偏相关分析
    print_and_write("\n偏相关分析：", f)
    for y_var in y_vars:
        print_and_write(f"\n{column_names[y_var]}:", f)
        pc_elevation = partial_correlation(df['RASTERVALU'], df[y_var], df[['Avg_Precip', 'Refined_Temperature']])
        pc_precip = partial_correlation(df['Avg_Precip'], df[y_var], df[['RASTERVALU', 'Refined_Temperature']])
        pc_temp = partial_correlation(df['Refined_Temperature'], df[y_var], df[['RASTERVALU', 'Avg_Precip']])
        
        print_and_write(f"海拔高度的偏相关系数: {pc_elevation:.4f}", f)
        print_and_write(f"平均降雨量的偏相关系数: {pc_precip:.4f}", f)
        print_and_write(f"平均气温的偏相关系数: {pc_temp:.4f}", f)
    
    print_and_write("\n偏相关分析注释：", f)
    print_and_write("1. 偏相关系数衡量了在控制其他变量的情况下，两个变量之间的线性关系强度。", f)
    print_and_write("2. 偏相关系数的范围是-1到1，绝对值越大表示关系越强。", f)
    print_and_write("3. 正值表示正相关，负值表示负相关。", f)

    print_and_write("\n分析完成。请查看输出文件夹中的统计结果和图表。", f)

print(f"分析结果已保存到 {output_folder} 文件夹中。")