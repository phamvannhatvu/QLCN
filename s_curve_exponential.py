from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu số lượng trang web theo năm
years = np.array([
    1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
    2011, 2012, 2013
]).reshape(-1, 1)

websites = np.array([
    1, 10, 130, 2738, 23500, 257601, 1117255, 2410067, 3177453, 17087182, 
    29254370, 38760373, 40912332, 51611646, 64780617, 85507314, 121892559, 172338726, 238027855, 206956723, 
    346004403, 697089489, 672985183
])

# Hàm mô hình tăng trưởng theo cấp số nhân: f(x) = a * exp(b * x)
def exponential_growth(x, a, b):
    return a * np.exp(b * (x - years.min()))

# Tìm các tham số tối ưu cho mô hình
params, _ = curve_fit(exponential_growth, years.flatten(), websites, p0=[1, 0.1])

# Dự báo số lượng trang web vào năm 2020 và 2025
future_years = np.array([2020, 2025]).reshape(-1, 1)
exp_predictions = exponential_growth(future_years.flatten(), *params)

# Tính năm mà số lượng trang web đạt 10 tỷ (10^10)
target_websites = 10**10
target_year = years.min() + (np.log(target_websites / params[0]) / params[1])

# Tính toán giá trị dự đoán cho các năm trong dữ liệu
predicted_websites = exponential_growth(years.flatten(), *params)

# Tính hệ số tương quan (correlation coefficient)
correlation = np.corrcoef(websites, predicted_websites)[0, 1]

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.scatter(years, websites, color='blue', label='Dữ liệu thực tế')
plt.plot(future_years, exp_predictions, 'ro', label='Dự báo (hàm mũ)')
future_range = np.arange(years.min(), 2050)  # Khoảng thời gian dự báo
plt.plot(future_range, exponential_growth(future_range, *params), 'g--', label='Mô hình cấp số nhân')

# Thêm đường ngang và dọc để hiển thị target_year
plt.axvline(x=target_year, color='orange', linestyle=':', label=f'Năm đạt 10 tỷ: {target_year:.2f}')

plt.xlabel("Năm")
plt.ylabel("Số lượng trang web")
plt.title(f"Dự báo số lượng trang web theo mô hình tăng trưởng mũ\nHệ số tương quan: {correlation:.4f}")
plt.legend()
plt.grid(True)
plt.show()

# Trả về kết quả dự báo, năm ước tính đạt 10 tỷ trang web, và hệ số tương quan
exp_predictions, target_year, correlation