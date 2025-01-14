import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.special import erf

data = pd.read_csv('/Users/miisa/Desktop/toinen python homma/fh.txt', delim_whitespace=True)
print(data.describe())

def skewed_gaussian(x, amp, mean, stddev, skew):
    norm_gaussian = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    return norm_gaussian * (1 + erf(skew * (x - mean) / (stddev * np.sqrt(2))))

def double_skewed_gaussian(x, amp1, mean1, stddev1, skew1, amp2, mean2, stddev2, skew2):
    return skewed_gaussian(x, amp1, mean1, stddev1, skew1) + skewed_gaussian(x, amp2, mean2, stddev2, skew2)

popt, pcov = curve_fit(double_skewed_gaussian, data['#U(V)'], data['I_A(E-10A)'], p0=[1, 10, 1, 0, 1, 5, 1, 0])
print(f"Optimaaliset parametrit: a1={popt[0]}, b1={popt[1]}, c1={popt[2]}, skew1={popt[3]}, a2={popt[4]}, b2={popt[5]}, c2={popt[6]}, skew2={popt[7]}")

plt.style.use('ggplot')
plt.figure(figsize=(12, 8), facecolor='lightgrey')
plt.errorbar(data['#U(V)'], data['I_A(E-10A)'], yerr=data['ErI_A'], xerr=data['ErU'], fmt='o', ecolor='red', capsize=5, label='Data', markersize=5, markerfacecolor='black')
x_fit = np.linspace(min(data['#U(V)']), max(data['#U(V)']), 1000)
y_fit = double_skewed_gaussian(x_fit, *popt)
plt.plot(x_fit, y_fit, label='Double Skewed Gaussian Fit', color='blue', linewidth=2)
plt.xlabel('Jännite (V)', fontsize=14)
plt.ylabel('Virta (E-10A)', fontsize=14)
plt.title('Virta jännitteen funktiona', fontsize=16)
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True, fancybox=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
