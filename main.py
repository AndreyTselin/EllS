from ellipsometry.ellipsometry_classes import EllModel 
from ellipsometry.models.layers_classes import IsoLayer, AirLayer, CauchyLayer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


si_pt = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\Platinum_early_research\MBE\Ellipsometry\Pt_Si\Substrate_Si_sample_5_Psi_Delta_190-3500nm_50-70deg_step10.csv",
                     sep=r'\s+', header=0, names=['wl', 'psi50', 'delta50', 'psi60', 'delta60', 'psi70', 'delta70'])

sio2_si_spectra = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\data for ellipsometry soft\ellipsometry_spectra\109nm_SiO2_on_Si_190-3500nm_70deg.txt",
                      sep=r'\s+', index_col=False, header=0, names=['wl', 'psi', 'delta'])
#print(sio2_si_spectra.head())

"""
plt.plot(si_pt['wl'], si_pt['psi50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['psi60'], label='60deg')
plt.plot(si_pt['wl'], si_pt['psi70'], label='70deg')
plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(200, 1000)
plt.show()


plt.plot(si_pt['wl'], si_pt['delta50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['delta60'], label='60deg')          
plt.plot(si_pt['wl'], si_pt['delta70'], label='70deg')
plt.xlabel('Wavelength, nm')
plt.ylabel('Delta, deg')
plt.xlim(200, 1000)
plt.show()
"""

si_asp = pd.read_csv(r'D:\VS code projects\EllS\assets\Si_Aspnes.txt', sep=r'\s+')
si_asp['wl'] = si_asp['wl(mu)'] * 1000

si_ell = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\data for ellipsometry soft\Si DUV NIR.txt",
                     sep=r'\s+', index_col=False, header=0, names=['wl', 'n', 'k'])

#print(si_ell)

sio2_cauchy = CauchyLayer(n_0=1.452, n_1=36, n_2=0, k_0=0, k_1=0, k_2=0, wl = np.arange(190, 4000, 1), thickness=109.4)
sio2_2 = CauchyLayer(n_0=1.452, n_1=36, n_2=0, k_0=0, k_1=0, k_2=0, wl = np.arange(190, 4000, 1), thickness=1)


# create t nm thick Si layer
si_l = IsoLayer(si_ell, 150)


# create Si substrate and air semi-infinite layers
si_sub = IsoLayer(si_ell)
#air = IsoLayer(air_nk)
air = AirLayer()

# create model
model_1 = EllModel([air, sio2_cauchy, si_sub], [50, 60, 75], si_asp['wl'])
model_2 = EllModel([air, si_l, si_sub], [50, 60, 75], si_asp['wl'])
model_3 = EllModel([air, sio2_cauchy, si_sub], [65, 70, 75], si_ell['wl'])

model_4 = EllModel([air, sio2_2,si_sub], [50, 60, 70], np.arange(200, 3500, 2))

#print(model_1.transfer_matrix(sio2, air, pol='s'))
#print(model_1.transfer_matrix_2(sio2, air, pol='s'))

#for i, angle in enumerate([50]):
#   print(model_1.transfer_matrix_2(sio2, air, pol='s')[i, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :, :, :] @ 
# model_1.transfer_matrix_2(sio2, air, pol='s')[1, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :])


##print(model_3.Psi_calc_plot())
#print(model_3.Delta_calc_plot())

model_3.Psi_calc()
model_3.Delta_calc()
#print(model_4.Delta.head())

model_4.Psi_calc()
model_4.Delta_calc()


plt.subplot(1, 2, 1)
plt.plot(si_pt['wl'], si_pt['psi50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['psi60'], label='60deg')
plt.plot(si_pt['wl'], si_pt['psi70'], label='70deg')

plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['50'], label='Calc 50deg', linestyle='--')
plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['60'], label='Calc 60deg', linestyle='--')
plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['70'], label='Calc 70deg', linestyle='--')

plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(200, 3500)

plt.legend()


plt.subplot(1, 2, 2)
plt.plot(si_pt['wl'], si_pt['delta50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['delta60'], label='60deg')          
plt.plot(si_pt['wl'], si_pt['delta70'], label='70deg')

plt.plot(model_4.Delta['wl'], model_4.Delta['50'], label='Calc 50deg', linestyle='--')
plt.plot(model_4.Delta['wl'], model_4.Delta['60'], label='Calc 60deg', linestyle='--')
plt.plot(model_4.Delta['wl'], model_4.Delta['70'], label='Calc 70deg', linestyle='--')

plt.xlabel('Wavelength, nm')
plt.ylabel('Delta, deg')
plt.xlim(200, 3500)
plt.legend()
plt.show()


plt.subplot(1, 2, 1)
plt.plot(sio2_si_spectra['wl'], sio2_si_spectra['psi'], label='70deg')

plt.plot(model_3.Psi['wl'], 90 - model_3.Psi['70'], label='Calc 50deg', linestyle='--')

plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(190, 3500)

plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sio2_si_spectra['wl'], sio2_si_spectra['delta'], label='70deg')

plt.plot(model_3.Delta['wl'], (model_3.Delta['70'] + 180) % 360, label='Calc 70deg', linestyle='--')


plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(190, 3500)

plt.legend()
plt.show()