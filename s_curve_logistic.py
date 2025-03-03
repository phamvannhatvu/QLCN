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

# Hàm logistic
def logistic_growth(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Tìm các tham số tối ưu cho mô hình logistic
params, _ = curve_fit(logistic_growth, years.flatten(), websites, p0=[1e9, 0.1, 2000], maxfev=10000)

# Dự báo số lượng trang web vào tháng 1/2020 và tháng 1/2025
future_years = np.array([2020, 2025]).reshape(-1, 1)
logistic_predictions = logistic_growth(future_years.flatten(), *params)

# Tính toán giá trị tối đa (L)
L = params[0]

# Tính năm mà số lượng trang web đạt 99% mức tối đa
target_percentage = 0.99
target_year = params[2] - (np.log((1 / target_percentage) - 1) / params[1])

# Tính toán giá trị dự đoán cho các năm trong dữ liệu
predicted_websites = logistic_growth(years.flatten(), *params)

# Tính hệ số tương quan (correlation coefficient)
correlation = np.corrcoef(websites, predicted_websites)[0, 1]

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.scatter(years, websites, color='blue', label='Dữ liệu thực tế')
plt.plot(future_years, logistic_predictions, 'ro', label='Dự báo (hàm logistic)')
future_range = np.arange(years.min(), 2050)  # Khoảng thời gian dự báo
plt.plot(future_range, logistic_growth(future_range, *params), 'g--', label='Mô hình logistic')
plt.axhline(y=L, color='purple', linestyle=':', label=f'Giá trị tối đa: {L:.2f}')
plt.axvline(x=target_year, color='orange', linestyle=':', label=f'Năm đạt 99% tối đa: {target_year:.2f}')
plt.xlabel("Năm")
plt.ylabel("Số lượng trang web")
plt.title(f"Dự báo số lượng trang web theo mô hình logistic\nHệ số tương quan: {correlation:.4f}")
plt.legend()
plt.grid(True)
plt.show()

# Trả về kết quả dự báo, giá trị tối đa, và năm đạt 99% mức tối đa
logistic_predictions, L, target_year