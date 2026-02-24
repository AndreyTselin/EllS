"""
Define classes for ellipsometry modeling:
- EllModel: main class for ellipsometry model
- Excperimental Data class: for handling experimental data Psi and Delta
- Fit class: for fitting ellipsometry data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares
from ellipsometry.models.layers_classes import IsoLayer, GyroLayer




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

        # Psi dict for each angle
        self.Psi = pd.DataFrame()
        self.Psi['wl'] = self.wl # in nm

        # Delta dict for each angle
        self.Delta = pd.DataFrame()
        self.Delta['wl'] = self.wl # in nm


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
        
        # interpolate refractive indices
        N_a = ambient_layer.interpolate_nk(self.wl)
        N = layer.interpolate_nk(self.wl)


        # create 4D array 
        M = np.zeros((self.n_angles, self.n_wl, 2, 2), dtype=complex)

        for i, angle in enumerate(self.angles_rad):

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

    def total_transfer_matrix(self, pol = 's'):
        """
        Calculate total transfer matrix of whole structure
        according to equation 1.261, p.85, Tompkins-Irene-HANDBOOK OF ELLIPSOMETRY.
        """

        
        # ---- Interpolate refractive index of air, which must be 1st layer (0 index) ----
        N_a = self.layers[0].interpolate_nk(self.wl)



        # ---- 4D array for total matrix ----
        R_total = np.zeros((self.n_angles, self.n_wl, 2, 2), dtype=complex)

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
            R_a = np.zeros((self.n_angles, self.n_wl, 2, 2), dtype=complex)
            R_a[i, :, 0, 0] = 0.5
            R_a[i, :, 0, 1] = 0.5 / N_a_eff
            R_a[i, :, 1, 0] = 0.5
            R_a[i, :, 1, 1] = -0.5 / N_a_eff

            
            # ---- create 4D array for  transfer matrix ----
            R = np.tile(np.eye(2, dtype=complex), (self.n_wl, 1, 1))


            # ---- in that loop calculate transfer matrix for each layer from 1 to -1 ----
            for layer in reversed(self.layers[1: -1]):
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
        N_s = self.layers[-1].interpolate_nk(self.wl)

        E_s = np.zeros((self.n_wl, 2, 1), dtype=complex)
        E_s[:, 0, 0] = 1
        E_s[:, 1, 0] = -1 / N_s


        # ---- Create null 4D array for vector [E_r, E_i] ----
        E = np.zeros((self.n_angles, self.n_wl, 2, 1), dtype=complex)

        # ---- Create null 3D array for refl coeff r ----
        r = np.zeros((self.n_angles, self.n_wl, 1), dtype=complex)

        R = self.total_transfer_matrix(pol)

        for i in range(self.n_angles):
            
            E[i, :] = R[i, :] @ E_s
            r[i, :, 0] = E[i, :, 0, 0] / E[i, :, 1, 0]

        if pol == 'p':
            r = -r   # переворачиваем знак только для p-поляризации
            
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

        for angle_idx, angle in enumerate(self.angles):
            self.Psi[str(angle)] = Psi[angle_idx, :, 0]


        return Psi
    


    def Delta_calc(self):
        """
        calculate Delta for each wl and angle
        ro = tan(Psi) * exp(i * Delta)
        """

        # ---- calculate ro ----
        ro = self.ro()

        Delta = (np.rad2deg(np.angle(ro)) + 180) % 360

        for angle_idx, angle in enumerate(self.angles):
            self.Delta[str(angle)] = Delta[angle_idx, :, 0]


        return Delta
    

    def Psi_calc_plot(self, save_path: str = None):
        """
        draw plot Psi (wl)
        """

        Psi = self.Psi_calc()
        plt.figure()

        for i, angle in enumerate(self.angles):
            
            plt.plot(self.wl, Psi[i, :, 0], label=f'{angle}')

        plt.legend()

        plt.xlabel('$\lambda$, nm')
        plt.ylabel('$\Psi$, deg')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        



    def Delta_calc_plot(self, save_path: str = None):
        """
        draw plot Delta vs wl
        """

        Delta = self.Delta_calc()
        plt.figure()

        for i, angle in enumerate(self.angles):
            
            plt.plot(self.wl, Delta[i, :, 0], label=f'{angle}')
            
        plt.legend()

        plt.xlabel('$\lambda$, nm')
        plt.ylabel('$\Delta$, deg')

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


class Fitting:
    """
    fit calculated Psi and Delta to experimental
    The method uses Levenberg-Marquardt algorithm
    """

    def __init__(self, model: EllModel, experimental_data: pd.DataFrame):
        """
        model - EllModel object
        experimental_data - DataFrame with columns 'wl', 'Psi', 'Delta' at different angles
        example of experimental_data columns: 'wl', 'Psi_60', 'Delta_60', 'Psi_70', 'Delta_70'
        """

        self.model = model
        self.experimental_data = experimental_data

