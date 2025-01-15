import pandas as pd  # Tuo pandas tietojen käsittelyyn
import matplotlib.pyplot as plt  # Tuo matplotlib piirtoihin
from scipy.optimize import curve_fit  # Tuo curve_fit funktioiden sovittamiseen dataan
import numpy as np  # Tuo numpy numeerisiin operaatioihin
from scipy.special import erf  # Tuo erf virhefunktioon

# Lue tiedot tiedostosta pandas DataFrameen
data = pd.read_csv('/Users/miisa/Desktop/toinen python homma/fh.txt', delim_whitespace=True)
print(data.describe())  # Tulosta perus tilastotiedot datasta

def skewed_gaussian(x, amp, mean, stddev, skew):
    """
    Määrittele vino Gaussin funktio.
    
    Parametrit:
    x - riippumaton muuttuja
    amp - amplitudi
    mean - keskiarvo
    stddev - keskihajonta
    skew - vinous
    
    Palauttaa:
    Vinon Gaussin funktion arvot.
    """
    norm_gaussian = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    return norm_gaussian * (1 + erf(skew * (x - mean) / (stddev * np.sqrt(2))))

def double_skewed_gaussian(x, amp1, mean1, stddev1, skew1, amp2, mean2, stddev2, skew2):
    """
    Määrittele kaksoisvinogaussin funktio.
    
    Parametrit:
    x - riippumaton muuttuja
    amp1, mean1, stddev1, skew1 - ensimmäisen vinon Gaussin parametrit
    amp2, mean2, stddev2, skew2 - toisen vinon Gaussin parametrit
    
    Palauttaa:
    Kahden vinon Gaussin funktioiden summa.
    """
    return skewed_gaussian(x, amp1, mean1, stddev1, skew1) + skewed_gaussian(x, amp2, mean2, stddev2, skew2)

# Sovita kaksoisvinogaussin funktio dataan
popt, pcov = curve_fit(double_skewed_gaussian, data['#U(V)'], data['I_A(E-10A)'], p0=[1, 10, 1, 0, 1, 5, 1, 0])
# Tulosta sovituksen optimaaliset parametrit
print(f"Optimaaliset parametrit: a1={popt[0]}, b1={popt[1]}, c1={popt[2]}, skew1={popt[3]}, a2={popt[4]}, b2={popt[5]}, c2={popt[6]}, skew2={popt[7]}")

# Aseta piirrostyyli 'ggplot'
plt.style.use('ggplot')
# Luo kuva määritellyllä koolla ja taustavärillä
plt.figure(figsize=(12, 8), facecolor='lightgrey')
# Piirrä data virhepalkkien kanssa
plt.errorbar(data['#U(V)'], data['I_A(E-10A)'], yerr=data['ErI_A'], xerr=data['ErU'], fmt='o', ecolor='red', capsize=5, label='Data', markersize=5, markerfacecolor='black')
# Generoi x-arvot sovitetulle käyrälle
x_fit = np.linspace(min(data['#U(V)']), max(data['#U(V)']), 1000)
# Laske y-arvot sovitetuilla parametreilla
y_fit = double_skewed_gaussian(x_fit, *popt)
# Piirrä sovitettu käyrä
plt.plot(x_fit, y_fit, label='Double Skewed Gaussian Fit', color='blue', linewidth=2)
# Aseta x-akselin nimi
plt.xlabel('Jännite (V)', fontsize=14)
# Aseta y-akselin nimi
plt.ylabel('Virta (E-10A)', fontsize=14)
# Aseta piirroksen otsikko
plt.title('Virta jännitteen funktiona', fontsize=16)
# Lisää selite määritellyillä ominaisuuksilla
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True, fancybox=True)
# Lisää ruudukko määritellyillä ominaisuuksilla
plt.grid(True, linestyle='--', alpha=0.7)
# Näytä piirros
plt.show()
