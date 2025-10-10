from Classes import EllModel, IsoLayer 
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
sio2_mal['n'] = ((1+0.6961663/(1-(0.0684043/sio2_mal['wl'])**2)
                 +0.4079426/(1-(0.1162414/sio2_mal['wl'])**2)
                 +0.8974794/(1-(9.896161/sio2_mal['wl'])**2))**.5)
sio2_mal['k'] = np.zeros(len(sio2_mal['wl']))

# create 100 nmthick SiO2 layer
sio2 = IsoLayer(sio2_mal, 100)

# create Si substrate and air semi-infinite layers
si_sub = IsoLayer(si_asp)
air = IsoLayer(air_nk)

# create model
model_1 = EllModel([air_nk, sio2, si_sub], [50, 60], si_asp['wl'][10])


print(model_1.transfer_matrix_2(sio2, air, pol='s'))