from ast import Raise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IsoLayer:
    """
    Describe 1 isotropic layer.
    Optical properties are table wl|n|k or wl|e1|e2
    
    """
    
    def __init__(self, df:pd.DataFrame, thickness: float = None):
        """
        take nk dataframe df with columns ['wl', 'n', 'k'] or ['wl', 'e1', 'e2']
        column wl - wavelength in nm
        column n - refractive index
        column k - extinction coeficient
        e1 - permittivity real part 
        e2 - permittivity imag part
        thickness in nm, if None -> semiimfinite medium (air or substrate)
        """
        self.thickness = thickness

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
            self.n_ = np.sqrt((abs_eps + self.e1) / 2)
            self.k = np.sqrt((abs_eps - self.e1) / 2)
        else:
            raise ValueError("DataFrame have to contain either (wl, n, k) or (wl, e1, e2)")
        
        self.complex_n = self.n + 1j * self.k
        self.complex_eps = self.e1 + 1j * self.e2
    
    def is_infinite(self) -> bool:
        """True, if imfinite medium (air, substrate)."""
        return self.thickness is None
    
    
    def get_nk(self, wavelength) -> pd.DataFrame:
        """
        Interpolate n, k for specified wavelengthes in nm
        wavelength : float or np.array
        return : (n, k) → массивы той же формы, что и wavelength
        """
        
        df = pd.DataFrame()
        wl = np.atleast_1d(wavelength)
        n = np.interp(wl, self.wl, self.n)
        k = np.interp(wl, self.wl, self.k)
        df['wl'] = wl
        df['n'] = n
        df['k'] = k
        return df
    
    
    def get_epsilon(self, wavelength) -> pd.DataFrame:
        """
        Interpolate e1, e2 for specified wavelengthes in nm
        wavelength : float or np.array
        return : (e1, e2) → массивы той же формы, что и wavelength
        """
        
        df = pd.DataFrame()
        wl = np.atleast_1d(wavelength)
        e1 = np.interp(wl, self.wl, self.e1)
        e2 = np.interp(wl, self.wl, self.e2)
        df['wl'] = wl
        df['n'] = e1
        df['k'] = e2
        return df
    


class EllModel:
    """
    take list of Layers [air(layer 0) / layer 1 / layer 2/ ... / layer N / substrate (layer N+1)] from top to bottom
    take angles of incidence and wavelengthes
    calculate reflection coeficients
    """

    def __init__(self, layers: list, angles: list, wavelengthes):
        """
        define layers, angles and wl
        take list of layers with length > 1
        take array of angles in deg
        take wavelengthes in nm
        """

        # list of layers and their counts
        if len(layers) < 2:
            raise ValueError(f"Expected number of layers > 1, got {len(layers)}.")
        else:
            self.layers = layers
            self.n_lrs = len(layers)

        # angles of incedence
        self.angles = np.atleast_1d(angles)
        self.angles_rad = np.atleast_1d(np.deg2rad(angles))
        self.n_angles = len(self.angles)


        # array of wavelengthes
        self.wl = np.atleast_1d(wavelengthes)
        self.n_wl = len(self.wl)

        # pandas df with s-polarisation reflection coefficients
        self.r_s = pd.DataFrame()

        # pandas df with p-polarisation reflection coefficients
        self.r_p = pd.DataFrame()
        
        # lets store each interface in the list which size is (len(layers) - 1)
        self.interfaces = list()

    
    def _r_s(self, layer_1, layer_2, angle):
        """
        The method is used yet only in the Class.
        calculate s-polarisation reflection coefficient.
        N_1 - complex refractive index of 1st medium (incoming).
        N_2 - complex refractive index of 2nd medium (outcoming).
        angle - angle of incidence in 1st medium in deg.
        """
        
        # interpolate complex refractive index to given wavelengthes
        n_1 = np.interp(self.wl, layer_1.wl, layer_1.complex_n)
        n_2 = np.interp(self.wl, layer_2.wl, layer_2.complex_n)

        # transfer angle in deg in rad
        ang = np.deg2rad(angle)

        # calculate angle of refraction using Snell law
        cos_2 = np.sqrt(1 - (n_1 / n_2) ** 2 * np.sin(ang) ** 2)

        # return s- reflection coefficient
        return (n_1 * np.cos(ang) - n_2 * cos_2) / (n_1 * np.cos(angle) + n_2 * cos_2)
    

    def _r_p(self, layer_1, layer_2, angle):
        """
        The method is used yet only in the Class.
        calculate p-polarisation reflection coefficient.
        N_1 - complex refractive index of 1st medium (incoming).
        N_2 - complex refractive index of 2nd medium (outcoming).
        angle - angle of incidence in 1st medium in deg.
        """

        # interpolate complex refractive index to given wavelengthes
        n_1 = np.interp(self.wl, layer_1.wl, layer_1.complex_n)
        n_2 = np.interp(self.wl, layer_2.wl, layer_2.complex_n)

        # transfer angle in deg in rad
        ang = np.deg2rad(angle)

        # calculate angle of refraction using Snell law
        cos_2 = np.sqrt(1 - (n_1 / n_2) ** 2 * np.sin(ang) ** 2)

        # return p- reflection coefficient
        return (n_1 * cos_2 - n_2 * np.cos(ang)) / (n_1 * cos_2 + n_2 * np.cos(ang))
    
    
    def _t_s(self, layer_1, layer_2, angle):
        """
        The method is used yet only in the Class.
        calculate s-polarisation transmission coefficient.
        N_1 - complex refractive index of 1st medium (incoming).
        N_2 - complex refractive index of 2nd medium (outcoming).
        angle - angle of incidence in 1st medium in deg.
        """

        # interpolate complex refractive index to given wavelengthes
        n_1 = np.interp(self.wl, layer_1.wl, layer_1.complex_n)
        n_2 = np.interp(self.wl, layer_2.wl, layer_2.complex_n)

        # transfer angle in deg in rad
        ang = np.deg2rad(angle)

        # calculate angle of refraction using Snell law
        cos_2 = np.sqrt(1 - (n_1 / n_2) ** 2 * np.sin(ang) ** 2)

        # return s- transmission coefficient
        return (2 * n_1 * np.cos(ang)) / (n_1 * np.cos(angle) + n_2 * cos_2)
    

    def _t_p(self, layer_1, layer_2, angle):
        """
        The method is used yet only in the Class.
        calculate p-polarisation transmission coefficient.
        N_1 - complex refractive index of 1st medium (incoming).
        N_2 - complex refractive index of 2nd medium (outcoming).
        angle - angle of incidence in 1st medium in deg.
        """

        # interpolate complex refractive index to given wavelengthes
        n_1 = np.interp(self.wl, layer_1.wl, layer_1.complex_n)
        n_2 = np.interp(self.wl, layer_2.wl, layer_2.complex_n)

        # transfer angle in deg in rad
        ang = np.deg2rad(angle)

        # calculate angle of refraction using Snell law
        cos_2 = np.sqrt(1 - (n_1 / n_2) ** 2 * np.sin(ang) ** 2)

        # return p- transmission coefficient
        return (2 * n_1 * np.cos(ang)) / (n_2 * np.cos(angle) + n_1 * cos_2)
    

    def interface_matrix(self, layer_1, layer_2, angle, pol = 's'):
        """
        Calculate Interface matrix
        """

        t_s = self._t_s(layer_1, layer_2, angle)
        t_p = self._t_p(layer_1, layer_2, angle)

        r_s = self._r_s(layer_1, layer_2, angle)
        r_p = self._r_p(layer_1, layer_2, angle)

        if pol == 's':
            inter_matrix = (1 / t_s) * np.array([[1, r_s], [r_s, 1]])
        
        elif pol == 'p':
            inter_matrix = (1 / t_p) * np.array([[1, r_p], [r_p, 1]])
        
        else:
            raise ValueError(f"Expected value are s or p, got {pol}")

        return inter_matrix
    

    
    def _interp_complex(self, x, xp, fp):
       """
       interpolate comples value
       """
       return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)
    


    def transfer_matrix(self, layer: IsoLayer, ambient_layer: IsoLayer, pol='s'):
        """
        calculate R matrix for layer and 1 interface
        according to Tompkins-Irene-HANDBOOK OF ELLIPSOMETRY, 
        Propagation Matrices, Stratified Structures chapter, p. 84
        equations 1.258.
        """


        if self.n_lrs < 2:
            raise ValueError(f'This method is used for 3 and more layers in structure,use method <INSERT METHOD NAME>')

        matrix_df = pd.DataFrame()

        # ambient complex refractive index
        N_a = np.interp(self.wl, ambient_layer.wl, ambient_layer.complex_n.real) + 1j * np.interp(self.wl, ambient_layer.wl, ambient_layer.complex_n.imag)
        
        # complex refractive index of layer at self.wl which is an array in general case
        N = np.interp(self.wl, layer.wl, layer.complex_n.real) + 1j * np.interp(self.wl, layer.wl, layer.complex_n.imag)



        for angle in self.angles:

            df1 = pd.DataFrame()

            if pol == 's':
                N_eff = N * np.sqrt(1 - (N_a * np.sin(np.deg2rad(angle)) / N) ** 2)

            elif pol == 'p':
                N_eff = N / np.sqrt(1 - (N_a * np.sin(np.deg2rad(angle)) / N) ** 2)
        
            else:
                raise ValueError(f'Expected values: s or p, got {pol}')

            # Propagation constant from equation 1.258 for all wavelengthes self.wl
            theta = 2 * np.pi * (layer.thickness / self.wl) * N * np.sqrt(1 - (N_a * np.sin(np.deg2rad(angle)) / N) ** 2)

        
            # parameters from equation 1.258
            cos_th = (np.exp(1j* theta) + np.exp(-1j* theta)) / 2
            sin_th = (np.exp(1j* theta) - np.exp(-1j* theta)) / 2


            # assemble df1 with elements of R transfer matrix
            df1['angle, deg'] = angle * np.ones(len(self.wl))
            df1['wl, nm'] = self.wl
            df1['r11'] = cos_th
            df1['r12'] = sin_th / N_eff
            df1['r21'] = N_eff * sin_th
            df1['r22'] = cos_th

            matrix_df = pd.concat([matrix_df, df1], ignore_index=True)
        
        return matrix_df
    

    def transfer_matrix_2(self, layer, ambient_layer, pol='s'):
        """
        Calculate 2x2 transfer matrices for given layer and ambient layer.
        Based on Tompkins & Irene, *Handbook of Ellipsometry*, Eq. 1.258.
    
        Returns:
            np.ndarray of shape (n_angles, n_wavelengths, 2, 2)
        """

        if self.n_lrs < 2:
            raise ValueError(
                "This method is used for 3 or more layers in structure; "
                "for 1-2 layers use method <INSERT METHOD NAME>."
            )

        # ---- interpolate complex values ----
        def interp_complex(x, xp, fp):
            return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)

        angles = np.deg2rad(self.angles)

        N_a = interp_complex(self.wl, ambient_layer.wl, ambient_layer.complex_n)
        N = interp_complex(self.wl, layer.wl, layer.complex_n)

        n_angles = len(angles)
        n_wl = len(self.wl)

        # create 4D array 
        M = np.zeros((n_angles, n_wl, 2, 2), dtype=complex)

        for i, angle in enumerate(angles):

            sin_a = np.sin(angle)
            sin_ratio = N_a * sin_a / N

            if pol == 's':
                N_eff = N * np.sqrt(1 - sin_ratio**2)

            elif pol == 'p':
                N_eff = N / np.sqrt(1 - sin_ratio**2)

            else:
                raise ValueError(f"Expected pol='s' or 'p', got {pol!r}")

            #  (theta) from equation 1.258
            theta = 2 * np.pi * (layer.thickness / self.wl) * N * np.sqrt(1 - sin_ratio**2)

            cos_th = (np.exp(1j * theta) + np.exp(-1j * theta)) / 2
            sin_th = (np.exp(1j * theta) - np.exp(-1j * theta)) / 2

            # Matrix elements
            M[i, :, 0, 0] = cos_th
            M[i, :, 0, 1] = sin_th / N_eff
            M[i, :, 1, 0] = N_eff * sin_th
            M[i, :, 1, 1] = cos_th

        return M

    def total_tranfer_matrix(self, pol = 's'):
        """
        Calculate total transfer matrix of whole structure
        according to equation 1.261, p.85, Tompkins-Irene-HANDBOOK OF ELLIPSOMETRY.
        """

        # ---- interpolate complex values ----
        def interp_complex(x, xp, fp):
            return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)
        
        # ---- Interpolate refractive index of air, which must be 1st layer (0 index) ----
        N_a = interp_complex(self.wl, self.layers[0].wl, self.layers[0].complex_n)

        # ---- Interpolate refractive index of substrate, which must be last layer (-1 index) ----
        N_s = interp_complex(self.wl, self.layers[-1].wl, self.layers[-1].complex_n)

        n_angles = len(self.angles)
        n_wl = len(self.wl)

        # ---- 4D array for total matrix ----
        R_total = np.zeros((n_angles, n_wl, 2, 2), dtype=complex)

        # ---- Main loop ----
        for i, angle in enumerate(self.angles_rad):

            # ---- define effective refractive index depends on polarisation
            if pol == 's':
                N_a_eff = N_a * np.cos(angle)

            elif pol == 'p':
                N_a_eff = N_a / np.cos(angle)
            
            else:
                raise ValueError(f"Expected pol='s' or 'p', got {pol!r}")

            
            # ---- calculate R_a matrix from equation 1.261, p.85 ----
            R_a = np.zeros((n_angles, n_wl, 2, 2), dtype=complex)
            R_a[i, :, 0, 0] = 0.5
            R_a[i, :, 0, 1] = 0.5 / N_a_eff
            R_a[i, :, 1, 0] = 0.5
            R_a[i, :, 1, 1] = -0.5 / N_a_eff

            
            # ---- create 4D array for  transfer matrix ----
            R = np.tile(np.eye(2), (n_wl, 1, 1))

            # ---- define substarte vector from equation 1.259, p.84 ----
            #E_s = np.array([[1], [-1 * N_s]])



            # ---- in that loop calculate transfer matrix for each layer from 1 to -1 ----
            for layer in self.layers[1: -1]:
                R_i = self.transfer_matrix_2(layer, self.layers[0], pol)[i, :, :, :]
                R = R @ R_i

            # ---- take into acount air matrix ----
            R = R_a[i, :, :, :] @ R

            # ---- save in total matrix for each angle ----
            R_total[i, :, :, :] = R

            
        return R_total
    


    def reflect_coeff(self, pol='s'):
        """
        calculate r_s and r_p reflection coefficients 
        """

        # ---- Interpolate refractive index of substrate, which must be last layer (-1 index) ----
        N_s = self._interp_complex(self.wl, self.layers[-1].wl, self.layers[-1].complex_n)

        E_s = np.zeros((self.n_wl, 2, 1), dtype=complex)
        E_s[:, 0, 0] = 1
        E_s[:, 1, 0] = -1 / N_s


        # ---- Create null 4D array for vector [E_r, E_i] ----
        E = np.zeros((self.n_angles, self.n_wl, 2, 1), dtype=complex)

        # ---- Create null 3D array for refl coeff r ----
        r = np.zeros((self.n_angles, self.n_wl, 1), dtype=complex)

        for i in range(self.n_angles):
            
            R = self.total_tranfer_matrix(pol)
            E[i, :] = R[i, :] @ E_s
            r[i, :, 0] = E[i, :, 0, 0] / E[i, :, 1, 0]
            

        
        return r


    def ro(self):
        """
        calculate ro $\ro$ = r_p / r_s
        """

        # ---- iniciate 3D array for ro ----
        ro = np.zeros((self.n_angles, self.n_wl, 1), dtype=complex)

        r_p = self.reflect_coeff('p')
        r_s = self.reflect_coeff('s')

        for i in range(self.n_angles):
            ro[i, :] = r_p[i, :] / r_s[i, :]

        return ro
    

    def Psi_calc(self):
        """
        calculate Psi for each wl and angle
        ro = tan(Psi) * exp(i * Delta)
        """

        # ---- calculate ro ----
        ro = self.ro()

        Psi = np.rad2deg(np.atan(np.abs(ro)))


        return Psi
    


    def Delta_calc(self):
        """
        calculate Delta for each wl and angle
        ro = tan(Psi) * exp(i * Delta)
        """

        # ---- calculate ro ----
        ro = self.ro()

        Delta = np.rad2deg(np.angle(ro))


        return Delta
    

    def Psi_calc_plot(self, save_path: str = None):
        """
        draw plot Psi (wl)
        """

        Psi = self.Psi_calc()

        for i, angle in enumerate(self.angles):
            
            plt.plot(self.wl, Psi[i, :, 0], label=f'{angle}')

        plt.legend()

        plt.xlabel('$\lambda$, nm')
        plt.ylabel('$\Psi$, deg')
        plt.savefig(r'D:\VS code projects\EllS\plots\spsi.png')
        plt.show()
        



    def Delta_calc_plot(self):
        """
        draw plot Delta vs wl
        """

        Delta = self.Delta_calc()

        for i, angle in enumerate(self.angles):
            
            fig = plt.plot(self.wl, Delta[i, :, 0], label=f'{angle}')
            
        plt.legend()

        plt.xlabel('$\lambda$, nm')
        plt.ylabel('$\Delta$, deg')

        plt.savefig(r'D:\VS code projects\EllS\plots\Delta.png')

        plt.show()





        

        

        












    

        
