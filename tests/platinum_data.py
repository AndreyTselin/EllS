from ellipsometry import IsoLayer, AirLayer, CauchyLayer, DrudeLorentzLayer, EllModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cm2eV(wl_cm):
    """Convert wavelength in cm^-1 to energy in eV."""
    return wl_cm / 8065.5


pt_ell = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\data for ellipsometry soft\ellipsometry_spectra\Pt_Si_s1_annealed_190-3500nm_50-70deg.txt",
                     sep=r'\s+', index_col=False, header=0, names=['wl', 'psi50', 'delta50', 'psi60', 'delta60', 'psi70', 'delta70'])

# Load Si nk data
si_ell = pd.read_csv(r"C:\Users\Andrey\Desktop\Plasminics\data for ellipsometry soft\Si DUV NIR.txt",
                     sep=r'\s+', index_col=False, header=0, names=['wl', 'n', 'k'])

# create SiO2 Cauchy layer
sio2_cauchy = CauchyLayer(n_0=1.452, n_1=36, n_2=0, k_0=0, k_1=0, k_2=0, wl = np.arange(190, 4000, 1), thickness=0.5)

# create Si substrate and air semi-infinite layers
si_sub = IsoLayer(si_ell)

#air = IsoLayer(air_nk)
air = AirLayer()

# platinum parameters from "C:\Users\Andrey\Desktop\Plasminics\data for ellipsometry soft\ellipsometry_spectra\Pt_Si_sample_1.JPG"
eps_inf = 1.5

drude_params = {
    'omega_p': cm2eV(55937),  # plasma frequency in eV  
    'gamma': cm2eV(651)  # damping constant in eV
}

lor_params = [
    {'omega_0': cm2eV(11802), 'f': cm2eV(48146), 'gamma': cm2eV(11846)},    
    {'omega_0': cm2eV(6593), 'f': cm2eV(54416), 'gamma': cm2eV(5829)},    
    {'omega_0': cm2eV(25607), 'f': cm2eV(81886), 'gamma': cm2eV(40335)},
    {'omega_0': cm2eV(84205), 'f': cm2eV(129975), 'gamma': cm2eV(44720)},]

wl = np.arange(190, 4000, 1)

# create platinum layer
pt_layer = DrudeLorentzLayer(eps_inf, drude_params, lor_params, wl, thickness=62)

model_pt = EllModel([air, pt_layer, sio2_cauchy, si_sub], [50, 60, 70], wl)

model_pt.Psi_calc()
model_pt.Delta_calc()


plt.subplot(1, 2, 1)

plt.plot(pt_ell['wl'], pt_ell['psi50'], label='Measured 50deg', color='blue')
plt.plot(pt_ell['wl'], pt_ell['psi60'], label='Measured 60deg', color='green')
plt.plot(pt_ell['wl'], pt_ell['psi70'], label='Measured 70deg', color='orange')

plt.plot(model_pt.wl, model_pt.Psi['50'], label='Calculated 50deg', linestyle='--', color='red')
plt.plot(model_pt.wl, model_pt.Psi['60'], label='Calculated 60deg', linestyle='--', color='red')
plt.plot(model_pt.wl, model_pt.Psi['70'], label='Calculated 70deg', linestyle='--', color='red')

plt.xlabel('Wavelength, nm')
plt.ylabel('Psi, deg')
plt.xlim(200, 3500)

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(pt_ell['wl'], pt_ell['delta50'], label='Measured 50deg', color='blue')
plt.plot(pt_ell['wl'], pt_ell['delta60'], label='Measured 60deg', color='green')
plt.plot(pt_ell['wl'], pt_ell['delta70'], label='Measured 70deg', color='orange')

plt.plot(model_pt.wl, 180 - model_pt.Delta['50'], label='Calculated 50deg', linestyle='--', color='red')
plt.plot(model_pt.wl, 180 - model_pt.Delta['60'], label='Calculated 60deg', linestyle='--', color='red')
plt.plot(model_pt.wl, 180 - model_pt.Delta['70'], label='Calculated 70deg', linestyle='--', color='red')

plt.xlabel('Wavelength, nm')
plt.ylabel('Delta, deg')
plt.xlim(200, 3500)
plt.legend()
plt.show()
