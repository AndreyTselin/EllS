from Ellipsometry_Classes import EllModel 
from Layers_Classes import IsoLayer, AirLayer, CauchyLayer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


si_pt = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\Platinum_early_research\MBE\Ellipsometry\Pt_Si\Substrate_Si_sample_5_Psi_Delta_190-3500nm_50-70deg_step10.csv",
                     sep=r'\s+', header=0, names=['wl', 'psi50', 'delta50', 'psi60', 'delta60', 'psi70', 'delta70'])

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

air_nk = pd.DataFrame({'wl' : [200, 400, 600],
                       'n': [1, 1, 1],
                       'k': [0, 0, 0]})

sio2_nk = pd.DataFrame({'wl' : [200, 400, 600, 800],
                       'n': [1.53, 1.47, 1.45, 1.45],
                       'k': [0, 0, 0, 0]})

sio2_cauchy = CauchyLayer(n_0=1.452, n_1=36, n_2=0, k_0=0, k_1=0, k_2=0, wl = np.arange(200, 2000, 1), thickness=150)

# create t nm thick SiO2 layer
sio2_l = IsoLayer(sio2_nk, 100)

# create t nm thick Si layer
si_l = IsoLayer(si_asp, 150)

# create Si substrate and air semi-infinite layers
si_sub = IsoLayer(si_asp)
#air = IsoLayer(air_nk)
air = AirLayer()

# create model
model_1 = EllModel([air, sio2_l, si_sub], [50, 60, 75], si_asp['wl'])
model_2 = EllModel([air, si_l, si_sub], [50, 60, 75], si_asp['wl'])
model_3 = EllModel([air, sio2_cauchy, si_sub], [50, 60, 75], np.arange(200, 1000, 1))

model_4 = EllModel([air, si_sub], [50, 60, 70], np.arange(200, 1000, 1))

#print(model_1.transfer_matrix(sio2, air, pol='s'))
#print(model_1.transfer_matrix_2(sio2, air, pol='s'))

#for i, angle in enumerate([50]):
#   print(model_1.transfer_matrix_2(sio2, air, pol='s')[i, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :, :, :] @ 
# model_1.transfer_matrix_2(sio2, air, pol='s')[1, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :])


#print(model_4.Psi_calc_plot())
#print(model_4.Delta_calc_plot())

model_4.Psi_calc()
model_4.Delta_calc()
#print(model_4.Delta.head())


plt.plot(si_pt['wl'], si_pt['psi50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['psi60'], label='60deg')
plt.plot(si_pt['wl'], si_pt['psi70'], label='70deg')

plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['50'], label='Calc 50deg', linestyle='--')
plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['60'], label='Calc 60deg', linestyle='--')
plt.plot(model_4.Psi['wl'], 90 - model_4.Psi['70'], label='Calc 70deg', linestyle='--')

plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(200, 1000)

plt.legend()
plt.show()


plt.plot(si_pt['wl'], si_pt['delta50'], label='50deg')
plt.plot(si_pt['wl'], si_pt['delta60'], label='60deg')          
plt.plot(si_pt['wl'], si_pt['delta70'], label='70deg')

plt.plot(model_4.Delta['wl'], model_4.Delta['50'] + 180, label='Calc 50deg', linestyle='--')
plt.plot(model_4.Delta['wl'], model_4.Delta['60'] + 180, label='Calc 60deg', linestyle='--')
plt.plot(model_4.Delta['wl'], model_4.Delta['70'] + 180, label='Calc 70deg', linestyle='--')

plt.xlabel('Wavelength, nm')
plt.ylabel('Delta, deg')
plt.xlim(200, 1000)
plt.legend()
plt.show()