from Classes import EllModel 
from Layers_Classes import IsoLayer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


si_asp = pd.read_csv(r'D:\VS code projects\EllS\assets\Si_Aspnes.txt', sep=r'\s+')
si_asp['wl'] = si_asp['wl(mu)'] * 1000

air_nk = pd.DataFrame({'wl' : [200, 400, 600],
                       'n': [1, 1, 1],
                       'k': [0, 0, 0]})

sio2_mal = pd.DataFrame()
sio2_mal['wl'] = si_asp['wl']
sio2_mal['n1'] = ((1+0.6961663/(1-(0.0684043/sio2_mal['wl'])**2)
                 +0.4079426/(1-(0.1162414/sio2_mal['wl'])**2)
                 +0.8974794/(1-(9.896161/sio2_mal['wl'])**2))**.5)
sio2_mal['n'] = 1.46 * np.ones(len(si_asp['wl']))
sio2_mal['k'] = np.zeros(len(sio2_mal['wl']))


sio2_nk = pd.DataFrame({'wl' : [200, 400, 600, 800],
                       'n': [1.53, 1.47, 1.45, 1.45],
                       'k': [0, 0, 0, 0]})

# create t nm thick SiO2 layer
sio2_l = IsoLayer(sio2_nk, 10)

# create t nm thick Si layer
si_l = IsoLayer(si_asp, 150)

# create Si substrate and air semi-infinite layers
si_sub = IsoLayer(si_asp)
air = IsoLayer(air_nk)

# create model
model_1 = EllModel([air, sio2_l, si_sub], [50, 60, 75], si_asp['wl'])
model_2 = EllModel([air, si_l, si_sub], [50, 60, 75], si_asp['wl'])


#print(model_1.transfer_matrix(sio2, air, pol='s'))
#print(model_1.transfer_matrix_2(sio2, air, pol='s'))

#for i, angle in enumerate([50]):
#   print(model_1.transfer_matrix_2(sio2, air, pol='s')[i, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :, :, :] @ 
# model_1.transfer_matrix_2(sio2, air, pol='s')[1, :, :, :])

#print(model_1.transfer_matrix_2(sio2, air, pol='s')[0, :])


#print(model_1.Psi_calc_plot())
#print(model_1.Delta_calc_plot())

print(sio2_l.get_nk([200, 300, 900, 1550]))

