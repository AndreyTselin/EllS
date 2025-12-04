"""
Layer Classes
--------------
Defines classes related to optical layers:
    Isolayer : common layer with tabulated wl, n, k (Parent class for other)
    -----------------------------------------------
    GyroLayer : layer with tabulated wl, n, k, g1, g2, based on Yeh formalism,
        see "MAGNETO-OPTICAL POLAR KERR EFFECT IN A FILM-SUBSTRATE SYSTEM  S. Vigfiovsky"
    ----------------------------------------------------------------------------------------------
    Caushy layer :
    --------------
    Drude-Lorentz :
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator



class IsoLayer:
    """
    Describe 1 isotropic layer.
    Optical properties are table wl|n|k or wl|e1|e2
    
    """
    
    def __init__(self, df:pd.DataFrame, thickness: float = None, layer_type: str = 'layer'):
        """
        take nk dataframe df with columns ['wl', 'n', 'k'] or ['wl', 'e1', 'e2']
        column wl - wavelength in nm
        column n - refractive index
        column k - extinction coeficient
        e1 - permittivity real part 
        e2 - permittivity imag part
        thickness in nm, if None -> semiimfinite medium (air or substrate)
        layer_type - 'layer' or 'substrate' or 'ambient'
        """
        self.thickness = thickness
        self.layer_type = layer_type

        if "n" in df.columns and "k" in df.columns:
            self.wl = df["wl"].to_numpy(dtype=float)
            self.n = df["n"].to_numpy(dtype=float)
            self.k = df["k"].to_numpy(dtype=float)

            # calculate ε1, ε2
            self.e1 = self.n**2 - self.k**2
            self.e2 = 2 * self.n * self.k

        elif "e1" in df.columns and "e2" in df.columns:
            self.wl = df["wl"].to_numpy(dtype=float)
            self.e1 = df["e1"].to_numpy(dtype=float)
            self.e2 = df["e2"].to_numpy(dtype=float)

            # calculate n, k
            abs_eps = np.sqrt(self.e1**2 + self.e2**2)
            self.n = np.sqrt((abs_eps + self.e1) / 2)
            self.k = np.sqrt((abs_eps - self.e1) / 2)
        else:
            raise ValueError("DataFrame have to contain either (wl, n, k) or (wl, e1, e2)")
        
        self.complex_n = self.n + 1j * self.k
        self.complex_eps = self.e1 + 1j * self.e2


    @staticmethod
    def complex_interp(x, xp, fp, method: str = 'CubicSpline'):
        """
        apply interpolation for real and imag parts
        x - required wavelengthes, at which you need calculate f
        xp - existing wavelengthes
        fp - existing values for xp, can be complex values
        method - interpolation method (mostly from scipy), cubicSpline - defoult

        return : complex array of f(x)
        """

        if method == 'CubicSpline':
            cs_real = CubicSpline(xp, np.real(fp))
            cs_imag = CubicSpline(xp, np.imag(fp))

            if np.all(cs_imag(x) == 0):
                return cs_real(x)
            else:
                return cs_real(x) + 1j * cs_imag(x)
        
        elif method == 'PchipInterpolator':
            pc_real = PchipInterpolator(xp, np.real(fp))
            pc_imag = PchipInterpolator(xp, np.imag(fp))

            if np.all(pc_imag(x) == 0):
                return pc_real(x)
            else:
                return pc_real(x) + 1j * pc_imag(x)
        
        else:
            raise ValueError('Method name!!!')

    
    def interpolate_nk(self, wavelength, method = 'PchipInterpolator') -> pd.DataFrame:
        """
        Interpolate n, k for specified wavelengthes in nm
        wavelength : float or np.array
        method : interpolation method
        return : n + i*k -> same shape, as wavelength
        """
        
        wl = np.atleast_1d(wavelength)

        if method == 'CubicSpline':
            n_cs = CubicSpline(self.wl, self.n)
            k_cs = CubicSpline(self.wl, self.k)
            return n_cs(wl) + 1j * k_cs(wl)
        
        elif method == 'PchipInterpolator':
            n_pc = PchipInterpolator(self.wl, self.n)
            k_pc = PchipInterpolator(self.wl, self.k)
            return n_pc(wl) + 1j * k_pc(wl)



    def is_infinite(self) -> bool:
        """True, if imfinite medium (air, substrate)."""
        return self.thickness is None
    
    
    def get_nk(self, wavelength, method = 'PchipInterpolator') -> pd.DataFrame:
        """
        Interpolate n, k for specified wavelengthes in nm
        wavelength : float or np.array
        return : (n, k) → массивы той же формы, что и wavelength
        """
        
        df = pd.DataFrame()
        wl = np.atleast_1d(wavelength)
        n = self.complex_interp(wl, self.wl, self.n, method)
        k = self.complex_interp(wl, self.wl, self.k, method)
        df['wl'] = wl
        df['n'] = n
        df['k'] = k
        return df
    
    
    def get_epsilon(self, wavelength, method='PchipInterpolator') -> pd.DataFrame:
        """
        Interpolate e1, e2 for specified wavelengthes in nm
        wavelength : float or np.array
        return : (e1, e2) → массивы той же формы, что и wavelength
        """
        
        df = pd.DataFrame()
        wl = np.atleast_1d(wavelength)
        e1 = self.complex_interp(wl, self.wl, self.e1, method)
        e2 = self.complex_interp(wl, self.wl, self.e2, method)
        df['wl'] = wl
        df['n'] = e1
        df['k'] = e2
        return df
    


class GyroLayer(IsoLayer):
    """
    Describe 1 gyrolayer with dielectric fucntion
            | e_0   -i*e_1  0   |
    Eps  =  | i*e_1  e_0    0   |
            | 0      0      e_0 |
    e_3 = e_0 in that case
    """

    def __init__(self, df: pd.DataFrame, thickness : float):
        """
        take dielectric tensor
        """
        super().__init__(df, thickness)
        
        # ---- add g1, g2 ----
        if 'g1' in df.columns and 'g2' in df.columns:
            
            self.g1 = df['g1'].to_numpy(dtype=float)
            self.g2 = df['g2'].to_numpy(dtype=float)
            self.complex_g = self.g1 + 1j * self.g2
            
        else:
            raise ValueError('DataFrame must contain g1 and g2 columns for GyroLayer. Required columns: "wl", "e1",' \
            '"e2", "g1", "g2"')
        
        self.n_z_plus = None
        self.n_z_minus = None
    

        

    def propag_mx(self, angle: float, wl: float) -> np.array:
        """
        calculate propagation matrix according to "MAGNETO-OPTICAL POLAR KERR EFFECT 
        IN A FILM-SUBSTRATE SYSTEM  S. Vigfiovsky" equation 6 for matrix P.
        Matrix is calculated for single angle and single wavelength

        angle in deg
        wl in nm 
        """

        angle_rad = np.deg2rad(angle)

        e_1 = self.complex_interp(wl, self.wl, self.complex_g)
        n = self.complex_interp(wl, self.wl, self.complex_n)
        e_0 = self.complex_interp(wl, self.wl, self.complex_eps)
        e_3 = e_0

        a_z = np.cos(angle_rad)
        a_y = np.sin(angle_rad)

        n_y = 1 * a_y # n_y is same for each layer due to Snell's law

        n_z_plus = ((e_0 * (e_3 - n_y ** 2) + e_3 * (e_0 - n_y ** 2)) + 
                    np.sqrt(n_y ** 4 * (e_3 - e_0) ** 2 + 4 * e_1 ** 2 * e_3 * (e_3 - n_y ** 2))) / (2 * e_3)
        
        n_z_minus = ((e_0 * (e_3 - n_y ** 2) + e_3 * (e_0 - n_y ** 2)) - 
                    np.sqrt(n_y ** 4 * (e_3 - e_0) ** 2 + 4 * e_1 ** 2 * e_3 * (e_3 - n_y ** 2))) / (2 * e_3)
        
        beta_plus = (2 * np.pi * self.thickness / wl) * n_z_plus
        beta_minus = (2 * np.pi * self.thickness / wl) * n_z_minus

        # Propagation matrix P
        #P = np.array([[np.exp(1j * beta_plus), 0, 0, 0],
        #              [0, np.exp(-1j * beta_plus), 0, 0],
        #              [0, 0, np.exp(1j * beta_minus), 0],
        #              [0, 0, 0, np.exp(-1j * beta_minus)]])
        
        P = np.diag([np.exp(1j * beta_plus), np.exp(-1j * beta_plus),
                     np.exp(1j * beta_minus), np.exp(-1j * beta_minus)])

        return P
    

    def dynamic_mx(self, angle: float, wl: float) -> np.array:
        """
        calculate dynamic matrix according to "MAGNETO-OPTICAL POLAR KERR EFFECT 
        IN A FILM-SUBSTRATE SYSTEM  S. Vigfiovsky" equation 6 for matrix D.
        Matrix is calculated for single angle and single wavelength

        angle in deg
        wl in nm 
        """

        angle_rad = np.deg2rad(angle)

        e_1 = self.complex_interp(wl, self.wl, self.complex_g)
        n = self.complex_interp(wl, self.wl, self.complex_n)
        e_0 = self.complex_interp(wl, self.wl, self.complex_eps)
        e_3 = e_0

        a_z = np.cos(angle_rad)
        a_y = np.sin(angle_rad)

        n_y = 1 * a_y # n_y is same for each layer due to Snell's law

        n_z_plus = ((e_0 * (e_3 - n_y **2) + e_3 * (e_0 - n_y **2)) + 
                    np.sqrt(n_y **4 * (e_3 - e_0) **2 + 4 * e_1 **2 * e_3 * (e_3 - n_y **2))) / (2 * e_3)
        
        n_z_minus = ((e_0 * (e_3 - n_y **2) + e_3 * (e_0 - n_y **2)) - 
                    np.sqrt(n_y **4 * (e_3 - e_0) **2 + 4 * e_1 **2 * e_3 * (e_3 - n_y **2))) / (2 * e_3)
        
        # Dynamic matrix elements 
        d_11 = 1j * e_1 * (e_3 - n_y **2)
        d_21 = 1j * e_1 * n_z_plus *(e_3 - n_y **2)
        d_23 = 1j * e_1 * n_z_minus * (e_3 - n_y **2)
        d_31 = (e_3 - n_y **2) * (e_0 - n_y ** 2 - n_z_plus **2)
        d_33 = (e_3 - n_y **2) * (e_0 - n_y ** 2 - n_z_minus **2)
        d_41 = - n_z_plus * e_3 * (e_0 - n_y **2 - n_z_plus **2)
        d_43 = - n_z_minus * e_3 * (e_0 - n_y **2 - n_z_minus **2)

        # Dynamic matrix D
        D = np.array([[d_11, d_11, d_11, d_11],
                      [d_21, -d_21, d_23, -d_23],
                      [d_31, d_31, d_33, d_33],
                      [d_41, -d_41, d_43, -d_43]])

        
        
        return D
    

    def d_0_matrix(self, angle: float, wl: float) -> np.array:
        """
        Docstring for d_0_matrix
        
        :param angle: angle of incedence in deg
        :type angle: float
        :param wl: wavelength in nm
        :type wl: float
        :return: matrix of isotropic ambient d_0
        :rtype: Any
        """

        
    

    def transfer_mx_m(self, angle: float, wl: float) -> np.array:
        """
        calculate transfer matrix according to "MAGNETO-OPTICAL POLAR KERR EFFECT 
        IN A FILM-SUBSTRATE SYSTEM  S. Vigfiovsky" equation 7 for matrix M.
        Matrix is calculated for single angle and single wavelength

        angle in deg
        wl in nm 
        """

        D = self.dynamic_mx(angle, wl)
        P = self.propag_mx(angle, wl)

        D_inv = np.linalg.inv(D)

        M = D @ P @ D_inv

        return M
        

class AirLayer(IsoLayer):
    """
    Describe air layer with n=1, k=0, thickness = None (semi-infinite)
    """

    def __init__(self):
        """
        Initialize air layer with n=1, k=0, thickness = None
        """
        wl = np.arange(200, 4000, 1)  # Wavelength range in nm
        n = np.ones(len(wl))    # Refractive index
        k = np.zeros(len(wl))    # Extinction coefficient

        df = pd.DataFrame({'wl': wl, 'n': n, 'k': k})

        super().__init__(df, thickness=None, layer_type='ambient')

    
    def d_0_inverse(self, angle: float) -> np.array:
        """
        Calculate d_0 matrix for air layer according to "MAGNETO-OPTICAL POLAR KERR EFFECT 
        IN A FILM-SUBSTRATE SYSTEM  S. Vigfiovsky" equation 6 for matrix D.
        Matrix is calculated for single angle and single wavelength

        angle in deg
        """

        angle_rad = np.deg2rad(angle)

        n = 1  # Refractive index of air

        a_z = np.cos(angle_rad)

        # Dynamic matrix D for isotropic ambient
        D = (1 / (2 * n * a_z)) * np.array([[n * a_z, 1, 0, 0],
                                            [n * a_z, -1, 0, 0],
                                            [0, 0, n, -a_z],
                                            [0, 0, n, a_z]])

        return D

class CauchyLayer(IsoLayer):
    """
    Describe 1 isotropic layer with Cauchy model for n and k
    n(λ) = A + B/λ^2 + C/λ^4
    k(λ) = D*exp(E/λ)
    """

    def __init__(self, n_0: float, n_1: float, n_2: float, k_0: float, k_1: float, k_2: float,  wl: np.array, thickness: float = None):
        """
        n_0, n_1, n_2 - Cauchy parameters for n
        k_0, k_1, k_2 - Cauchy parameters for k
        thickness in nm
        wl - wavelength range in nm for which the model is valid
        """

        self.n_0 = n_0
        self.n_1 = n_1
        self.n_2 = n_2
        self.k_0 = k_0
        self.k_1 = k_1
        self.k_2 = k_2
        self.wl = np.atleast_1d(wl) # wl
        self.thickness = thickness


        # calculate n and k using Cauchy model
        c_0 = 100
        c_1 = 1e7
        self.n =  self.n_0 + c_0 * self.n_1 / self.wl**2 + c_1 * self.n_2 / self.wl**4
        self.k = self.k_0 + c_0 * self.k_1 / self.wl**2 + c_1 * self.k_2 / self.wl**4


        # calculate ε1, ε2
        self.e1 = self.n**2 - self.k**2
        self.e2 = 2 * self.n * self.k

        # calculate complex n and ε
        self.complex_n = self.n + 1j * self.k
        self.complex_eps = self.e1 + 1j * self.e2

class DrudeLorentzLayer(IsoLayer):
    """
    Describe 1 isotropic layer with Drude-Lorentz model for ε
    ε(ω) = ε_inf - (ω_p^2)/(ω^2 + i*γ*ω) + Σ (f_j * ω_pj^2) / (ω_j^2 - ω^2 - i*γ_j*ω)
    """

    def __init__(self, epsilon_inf: float, drude_params: dict, lorentz_params: list, wl: np.array, thickness: float = None):
        """
        epsilon_inf - high-frequency dielectric constant
        drude_params - dictionary with keys 'omega_p' (plasma frequency) and 'gamma' (damping constant) in eV
        lorentz_params - list of dictionaries, each with keys 'f' (oscillator strength), 'omega_0' (resonance frequency), 
        and 'gamma' (damping constant) in eV
        wl - wavelength range in nm for which the model is valid
        thickness in nm
        """
        self.epsilon_inf = epsilon_inf
        self.drude_params = drude_params
        self.lorentz_params = lorentz_params
        self.wl = np.atleast_1d(wl) # wl
        self.thickness = thickness

        # Convert wavelength to eV
        omega = 1240 / self.wl 

        # Calculate ε(ω) using Drude-Lorentz model
        epsilon = np.full_like(omega, self.epsilon_inf, dtype=complex)

        # Drude term
        omega_p = drude_params['omega_p']
        gamma = drude_params['gamma']
        epsilon -= (omega_p**2) / (omega**2 + 1j * gamma * omega)

        # Lorentz terms
        for params in lorentz_params:
            f = params['f']
            omega_0 = params['omega_0']
            gamma_j = params['gamma']
            epsilon += (f * omega_0**2) / (omega_0**2 - omega**2 - 1j * gamma_j * omega)

        # Extract n and k from ε
        self.e1 = np.real(epsilon)
        self.e2 = np.imag(epsilon)

        # calculate n and k
        self.n = np.sqrt((np.sqrt(self.e1**2 + self.e2**2) + self.e1) / 2)
        self.k = np.sqrt((np.sqrt(self.e1**2 + self.e2**2) - self.e1) / 2)

        # calculate complex n and ε
        self.complex_n = self.n + 1j * self.k
        self.complex_eps = self.e1 + 1j * self.e2