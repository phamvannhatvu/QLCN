import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(years, websites)

# Dự báo số lượng trang web vào năm 2020 và 2025
future_years = np.array([2020, 2025]).reshape(-1, 1)
predictions = model.predict(future_years)

# Tính toán giá trị dự đoán cho các năm trong dữ liệu
predicted_websites = model.predict(years)

# Tính hệ số tương quan (correlation coefficient)
correlation = np.corrcoef(websites, predicted_websites)[0, 1]

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.scatter(years, websites, color='blue', label='Dữ liệu thực tế')
plt.plot(future_years, predictions, 'ro', label='Dự báo')
plt.plot(np.append(years, future_years), model.predict(np.append(years, future_years).reshape(-1, 1)), 'g--', label='Hồi quy tuyến tính')
plt.xlabel("Năm")
plt.ylabel("Số lượng trang web")
plt.title(f"Dự báo số lượng trang web trong tương lai\nHệ số tương quan: {correlation:.4f}")
plt.legend()
plt.grid(True)
plt.show()

# Trả về kết quả dự báo và hệ số tương quan
predictions, correlation