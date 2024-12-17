# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:57:47 2024

@author: Shuangjie Zhao
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, convolve, gaussian
from scipy.ndimage import gaussian_filter1d


class VASP_tool:
    def __init__(self, files_path, spin_polarization=False, function = None, fermi_level=0):
        """
        Initializes the VASP_tool object and reads/processes data.

        Args:
            files_path (str): Path to the directory containing VASP output files.
            spin_polarization (bool, optional): Whether the calculation is spin-polarized. Defaults to False.
            fermi_level (int or str, optional): Fermi level (0, "VBM", or "CBM"). Defaults to 0.
        """
        self.up, self.dw, self.totband = None, None, None  # Initialize band data
        self.up_dos, self.dw_dos, self.energy, self.totdos = None, None, None, None  # Initialize dos data
        self.KLABELS, self.KLINES, self.kpath = None, None, None  # Initialize k-point data
        self.Fermi, self.spin_polarized = None, None  # Initialize Fermi level and spin_polarization
        self.high_symmetry = None  # Initialize high symmetry k-point labels and coordinates
        if function == "regular band":
            self._read_and_process_data_regularband(files_path, spin_polarization, fermi_level)  # Read and process data
        if function == "regular DOS":
            self._read_and_process_data_regulardos(files_path, spin_polarization, fermi_level)  # Read and process data
        if function == "Projected band":
            self._read_and_process_data_projectedband(files_path, spin_polarization, fermi_level)  # Read and process data
        if function == "Projected DOS":
            self._read_and_process_data_projecteddos(files_path, spiecies, spin_polarization, fermi_level)  # Read and process data

    def _read_and_process_data_regularband(self, files_path, spin_polarization, fermi_level):
        """Reads and processes VASP data files."""

        # Determine file keys based on spin polarization
        keys = ["UP", "DW", "KLABELS", "KLINES"] if spin_polarization else ["REFORMATTED", "KLABELS", "KLINES"]
        files = {}
        for entry in os.scandir(files_path):
            if entry.is_file() and not entry.name.startswith("."):  
                for key in keys:
                    if key.upper() in entry.name.upper(): 
                        files[key] = entry.path
                        keys.remove(key)  # Remove the key once found to avoid duplicates
                        break  # Move on to the next file once a match is found

        if keys:  # Raise an error if any keys were not found in any file
            raise FileNotFoundError(f"The following files are missing: {', '.join(keys)}")

        # Read and process high-symmetry k-points
        high_symmetry_data = np.genfromtxt(files["KLABELS"], skip_header=1, dtype=str, comments="#")
        high_symmetry_data[high_symmetry_data[:, 0] == 'GAMMA', 0] = 'G'
        self.high_symmetry = high_symmetry_data
        self.KLABELS = high_symmetry_data[:, 1].astype(float)

        # Read K-lines data
        self.KLINES = np.loadtxt(files["KLINES"])

        # Read and process band data
        if spin_polarization:
            self.up = np.loadtxt(files["UP"], skiprows=1)[:, 1:]
            self.dw = np.loadtxt(files["DW"], skiprows=1)[:, 1:]
            self.totband = [self.up, self.dw]
            self.kpath = np.loadtxt(files["UP"], skiprows=1)[:, 0] 
            self.Fermi = 0
        else:
            self.totband = np.loadtxt(files["REFORMATTED"], skiprows=1)[:, 1:]
            df_energy_levels = pd.DataFrame(self.totband)
            self.Fermi = (
                df_energy_levels[df_energy_levels < 0].dropna(axis=1).iloc[:, -1].max()
                if fermi_level == "VBM"
                else df_energy_levels[df_energy_levels > 0].dropna(axis=1).iloc[:, -1].min()
                if fermi_level == "CBM"
                else 0
            )
            self.kpath = np.loadtxt(files["REFORMATTED"], skiprows=1)[:, 0] 
            self.totband -= self.Fermi  # Shift bands relative to Fermi level
            self.Fermi = 0

        self.spin_polarized = spin_polarization
    
    def plot_regular_band(self, save_path=None, figsize=(8, 6), band_color='k', band_linewidth=2,
                        sym_point_color='k', sym_point_linestyle='--', fermi_color = 'r', kline_linewidth=1,fermi_linewidth=1.5,
                        x_label='k-path', y_label=r'E-E$_{F}$ (eV)', title='Band Structure', frame_linewidth = 1.5,
                        ylim=None, dpi=300, label_fontsize=20, xmargin=0, yticks=1,ytick_fontsize=16,xtick_fontsize=16):
        
        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        spin_polarization = self.spin_polarized
        if spin_polarization:
            up_plot = ax.plot([], [], color=band_color[0], linewidth=band_linewidth, label="Spin-Up")[0]
            dw_plot = ax.plot([], [], color=band_color[1], linewidth=band_linewidth, label="Spin-Down")[0]
            # Plot band structure 
            for i,bcolor in zip(self.totband,band_color):
                for j in range(len(i[0])):
                    ax.plot(self.kpath, i[:, j], color=bcolor, linewidth=band_linewidth)
                    
            legend = ax.legend(handles=[up_plot, dw_plot], loc='upper right', fontsize=label_fontsize)
            # Adjust legend to fit within the figure (optional)
            legend.get_frame().set_linewidth(0.5)  # Set legend frame width for visual appeal
            plt.tight_layout() 

        else:
            for i in range(len(self.totband[0])):
                ax.plot(self.kpath, self.totband[:, i], color=band_color, linewidth=band_linewidth)
        
        # Highlight high symmetry k points
        for point in self.KLABELS:
            ax.axvline(x=point, color=sym_point_color, linestyle=sym_point_linestyle, linewidth=kline_linewidth)

        # Plot Fermi level 
        ax.axhline(y=self.Fermi, color=fermi_color, linestyle='-.', linewidth=fermi_linewidth)  
            

    
        # Automatic x-axis limit adjustment
        margin = xmargin  # Adjust this percentage value as needed
        data_min = self.kpath.min()
        data_max = self.kpath.max()
        x_range = data_max - data_min
        ax.set_xlim(data_min - margin * x_range, data_max + margin * x_range)
    
        # Add labels and yticks
        if yticks:  
            # Calculate yticks based on ylim and desired tick spacing
            if ylim:
                y_min, y_max = ylim
            else:
                y_min = ax.get_ylim()[0]
                y_max = ax.get_ylim()[1]

            yticks = np.arange(np.ceil(y_min / yticks) * yticks, 
                           np.floor(y_max / yticks) * yticks + yticks,
                           yticks)
            ax.set_yticks(yticks)

        ax.tick_params(axis='y', labelsize=ytick_fontsize)     
        ax.set_xticks(self.KLABELS)
        ax.set_xticklabels(self.high_symmetry[:, 0], fontsize=xtick_fontsize)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.set_title(title, fontsize=label_fontsize)
        
    
        # Set ylim and xlim if provided
        if ylim:
            ax.set_ylim(ylim)

        # Save the plot if save_path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi)
            
        for spine in ax.spines.values():
            spine.set_linewidth(frame_linewidth)
    
        plt.show()
        
##process dos data    
    def _read_and_process_data_regulardos(self, files_path, spin_polarization, fermi_level):

        # Determine file keys based on spin polarization
        keys = ["TDOS"]
        files = {}
        for entry in os.scandir(files_path):
            if entry.is_file() and not entry.name.startswith("."):  
                for key in keys:
                    if key.upper() in entry.name.upper(): 
                        files[key] = entry.path
                        keys.remove(key)  # Remove the key once found to avoid duplicates
                        break  # Move on to the next file once a match is found

        if keys:  # Raise an error if any keys were not found in any file
            raise FileNotFoundError(f"The following files are missing: {', '.join(keys)}") 
        
        
        self.energy = np.loadtxt(files["TDOS"], skiprows=1)[:, 0]
        
        if spin_polarization:
            self.up_dos = np.loadtxt(files["TDOS"], skiprows=1)[:, 1]
            self.dw_dos = np.loadtxt(files["TDOS"], skiprows=1)[:, 2]
            
        else:
            self.totdos = np.loadtxt(files["TDOS"], skiprows=1)[:, 1]
        
        self.spin_polarized = spin_polarization
        
    
    def plot_regular_dos(self, save_path=None, figsize=(6, 12), line_color='k', fermi_color='r',
                    line_linewidth=2, fermi_linewidth=1.5, x_label='number of states',
                    y_label=r'E-E$_{F}$ (eV)', title='DOS', xlim=None, ylim=None,
                    dpi=300, label_fontsize=20, legend_size=10, frame_linewidth = 1.5,xtick=20, ytick=2,
                    ytick_fontsize=16, xtick_fontsize=16, smoothing_method='savgol', window_length=11, polyorder=2,
                    broadening_method='gaussian', broadening_sigma=0.1,orientation = "vertical"):
        

        fig, ax = plt.subplots(figsize=figsize)
        spin_polarization = self.spin_polarized

    # Calculate max DOS to determine mirrored x-axis limits
        max_dos = max(np.abs(self.up_dos).max(), np.abs(self.dw_dos).max())


        # Smoothing
        if smoothing_method == 'savgol':
            up_dos_smoothed = savgol_filter(self.up_dos, window_length, polyorder)
            dw_dos_smoothed = savgol_filter(self.dw_dos, window_length, polyorder)
        elif smoothing_method == 'convolution':
            window = np.ones(window_length) / window_length
            up_dos_smoothed = np.convolve(self.up_dos, window, mode='same')
            dw_dos_smoothed = np.convolve(self.dw_dos, window, mode='same')

    # Peak Broadening (applied after smoothing)
        if broadening_method == 'gaussian':
            up_dos_broadened = gaussian_filter1d(up_dos_smoothed, sigma=broadening_sigma)
            dw_dos_broadened = gaussian_filter1d(dw_dos_smoothed, sigma=broadening_sigma)
    # Add more broadening methods if needed (e.g., Lorentzian)
        if orientation == "vertical":
            if spin_polarization:
        # Mirrored plotting for spin-up and spin-down (using broadened data)
                up_plot = ax.plot(np.abs(up_dos_broadened), self.energy, color=line_color[0], 
                        linewidth=line_linewidth, label="Spin-Up")[0]
                dw_plot = ax.plot(-np.abs(dw_dos_broadened), self.energy, color=line_color[1], 
                        linewidth=line_linewidth, label="Spin-Down")[0]
            else:
                totdos_smoothed = savgol_filter(self.totdos, window_length, polyorder)
                totdos_broadened = gaussian_filter1d(totdos_smoothed, sigma=broadening_sigma)
                ax.plot(totdos_broadened, self.energy, color=line_color, linewidth=line_linewidth)

            self.Fermi = 0
            ax.axhline(y=self.Fermi, color=fermi_color, linestyle='-.', linewidth=fermi_linewidth)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=line_linewidth)
            if xlim is None:
                xlim = [-max_dos, max_dos]
            ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlabel(x_label, fontsize=label_fontsize)
            ax.set_ylabel(y_label, fontsize=label_fontsize)
            ax.set_title(title, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
            ax.legend(fontsize=legend_size,loc='upper right')
            if xtick is not None:
                max_dos = max(np.abs(xlim[0]), np.abs(xlim[1]))  # Max absolute value in xlim
                tick_positions = np.arange(0, max_dos + xtick, xtick)
                tick_positions = tick_positions.astype(int)
                tick_labels = [f"{x}" for x in tick_positions]  # Format with one decimal place
                ax.set_xticks(np.concatenate((-tick_positions[::-1], tick_positions[1:])))  # Mirrored ticks
                ax.set_xticklabels(tick_labels[::-1] + tick_labels[1:])

            if ytick is not None:
                ax.set_yticks(np.arange(ylim[0], ylim[1] + ytick, ytick))
            
        elif orientation == "horizontal":
            if spin_polarization:
                up_plot = ax.plot(self.energy, np.abs(up_dos_broadened), color=line_color[0], 
                        linewidth=line_linewidth, label="Spin-Up")[0]
                dw_plot = ax.plot(self.energy, -np.abs(dw_dos_broadened), color=line_color[1], 
                        linewidth=line_linewidth, label="Spin-Down")[0]
            else:
                totdos_smoothed = savgol_filter(self.totdos, window_length, polyorder)
                totdos_broadened = gaussian_filter1d(totdos_smoothed, sigma=broadening_sigma)
                ax.plot(self.energy, totdos_broadened, color=line_color, linewidth=line_linewidth)
            self.Fermi = 0
            ax.axvline(x=self.Fermi, color=fermi_color, linestyle='-.', linewidth=fermi_linewidth)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=line_linewidth)
            if ylim is None:
                ylim = [-max_dos, max_dos]
            ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_xlabel(x_label, fontsize=label_fontsize)
            ax.set_ylabel(y_label, fontsize=label_fontsize)
            ax.set_title(title, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
            ax.legend(fontsize=legend_size,loc='upper right')
            if ytick is not None:
                max_dos = max(np.abs(ylim[0]), np.abs(ylim[1]))  # Max absolute value in xlim
                tick_positions = np.arange(0, max_dos + ytick, ytick)
                tick_positions = tick_positions.astype(int)
                tick_labels = [f"{y}" for y in tick_positions]  # Format with one decimal place
                ax.set_yticks(np.concatenate((-tick_positions[::-1], tick_positions[1:])))  # Mirrored ticks
                ax.set_yticklabels(tick_labels[::-1] + tick_labels[1:])

            if xtick is not None:
                ax.set_xticks(np.arange(xlim[0], xlim[1] + xtick, xtick))
            
        else:
            raise ValueError("Invalid orientation. Choose 'vertical' or 'horizontal'.")

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
        for spine in ax.spines.values():
            spine.set_linewidth(frame_linewidth)
        else:
            plt.show()
    
    def _read_and_process_data_projecteddos(self, files_path, spiecies, spin_polarization, fermi_level):

        # Determine file keys based on spin polarization
        if spin_polarization:
            key_up = []
            key_dw = []
            for i in spiecies:
                key_up.append(f'{i}_up')
                key_dw.append(f'{i}_dw')
                
            ##prepare path of files    
            files_up = {}
            for entry in os.scandir(files_path):
                if entry.is_file() and not entry.name.startswith("."): 
                    for key in key_up:
                        if key.upper() in entry.name.upper():
                            files_up[key] = entry.path
                            key_up.remove(key)  # Remove the key once found to avoid duplicates
                            break  # Move on to the next file once a match is found
            if keys:  # Raise an error if any keys were not found in any file
                raise FileNotFoundError(f"The following files are missing: {', '.join(keys)}") 
            
            
            files_dw = {}
            for entry in os.scandir(files_path):
                if entry.is_file() and not entry.name.startswith("."): 
                    for key in key_dw:
                        if key.upper() in entry.name.upper():
                            files_dw[key] = entry.path
                            key_dw.remove(key)  # Remove the key once found to avoid duplicates
                            break  # Move on to the next file once a match is found
            if keys:  # Raise an error if any keys were not found in any file
                raise FileNotFoundError(f"The following files are missing: {', '.join(keys)}") 
            
            self.up_dos = []  # Initialize as a list
            self.dw_dos = []
            self.totdos = []

            for file in files_up:
                self.energy = np.loadtxt(files_up[file], skiprows=1)[:, 0]
                buffer = np.loadtxt(files_up[file], skiprows=1)[:, 1:]
                buffer = np.hstack((
                          buffer[:, 0:1],   # First column
                          buffer[:, 1:4].sum(axis=1, keepdims=True),  # Sum of 2nd, 3rd, 4th
                          buffer[:, 4:9].sum(axis=1, keepdims=True)   # Sum of 5th through 9th
                          buffer[:, 9::].sum(axis=1, keepdims=True) #last column
                          ))
                self.up_dos.append(buffer) 

            self.up_dos = np.hstack(self.up_dos)  # Stack into a single NumPy array
            
            for file in files_dw:
                buffer = np.loadtxt(files_dw[file], skiprows=1)[:, 1:]
                buffer = np.hstack((
                          buffer[:, 0:1],   # First column
                          buffer[:, 1:4].sum(axis=1, keepdims=True),  # Sum of 2nd, 3rd, 4th
                          buffer[:, 4:9].sum(axis=1, keepdims=True)   # Sum of 5th through 9th
                          buffer[:, 9::].sum(axis=1, keepdims=True) #last column
                          ))
                self.dw_dos.append(buffer) 

            self.dw_dos = np.hstack(self.dw_dos)  # Stack into a single NumPy array            
        else:
            keys = spiecies
            files = {}
            for entry in os.scandir(files_path):
                if entry.is_file() and not entry.name.startswith("."): 
                    for key in keys:
                        if key.upper() in entry.name.upper():
                            files[key] = entry.path
                            key_up.remove(key)  # Remove the key once found to avoid duplicates
                            break  # Move on to the next file once a match is found
            if keys:  # Raise an error if any keys were not found in any file
                raise FileNotFoundError(f"The following files are missing: {', '.join(keys)}")
            
            for file in files:
                self.energy = np.loadtxt(files[file], skiprows=1)[:, 0]
                buffer = np.loadtxt(files[file], skiprows=1)[:, 1:]
                buffer = np.hstack((
                          buffer[:, 0:1],   # First column
                          buffer[:, 1:4].sum(axis=1, keepdims=True),  # Sum of 2nd, 3rd, 4th
                          buffer[:, 4:9].sum(axis=1, keepdims=True)   # Sum of 5th through 9th
                          buffer[:, 9::]    #last column
                          ))
                self.totdos.append(buffer)
            self.totdos = np.hstack(self.totdos)
                
    def plot_projected_dos(self, save_path=None, spieces = None, orbitals=['tot'], figsize=(6, 12), line_color='k', fermi_color='r',
                    line_linewidth=2, fermi_linewidth=1.5, x_label='number of states', up_symbol = "\u2191",dw_symbol ="\u2193",
                    y_label=r'E-E$_{F}$ (eV)', title='PDOS', xlim=None, ylim=None,
                    dpi=300, label_fontsize=20, legend_size=10, frame_linewidth = 1.5,xtick=20, ytick=2,
                    ytick_fontsize=16, xtick_fontsize=16, smoothing_method='savgol', window_length=11, polyorder=2,
                    broadening_method='gaussian', broadening_sigma=0.1,orientation = "vertical", plot_tot = True):
        
        fig, ax = plt.subplots(figsize=figsize)
        spin_polarization = self.spin_polarized

    # Calculate max DOS to determine mirrored x-axis limits
        max_dos = max(np.abs(self.up_dos).max(), np.abs(self.dw_dos).max())
        
        ###orbital contribution indexing
        orbital_name = {"s":0, "p":1, "d":2, "tot":3}
        key_list = list(orbital_name.keys())
        orbital_index = [orbital_name[orbital_key] for orbital_key in orbitals]

        # Smoothing
        if smoothing_method == 'savgol':
            up_dos_smoothed = savgol_filter(self.up_dos, window_length, polyorder)
            dw_dos_smoothed = savgol_filter(self.dw_dos, window_length, polyorder)
        elif smoothing_method == 'convolution':
            window = np.ones(window_length) / window_length
            up_dos_smoothed = np.convolve(self.up_dos, window, mode='same')
            dw_dos_smoothed = np.convolve(self.dw_dos, window, mode='same')

    # Peak Broadening (applied after smoothing)
        if broadening_method == 'gaussian':
            up_dos_broadened = gaussian_filter1d(up_dos_smoothed, sigma=broadening_sigma)
            dw_dos_broadened = gaussian_filter1d(dw_dos_smoothed, sigma=broadening_sigma)
    # Add more broadening methods if needed (e.g., Lorentzian)
        if orientation == "vertical":
            if spin_polarization:
                if orbital_index != [3]:
                    for i,m in zip(orbital_index,spieces):
                        for j,n in zip(range(i,len(spieces)*4,4),line_color):
                            up_plot = ax.plot(np.abs(up_dos_broadened[:,j]), self.energy, color=n, 
                                      linewidth=line_linewidth, label=f'{m}_{key_list[i]}{up_symbol}')[0]
                            dw_plot = ax.plot(-np.abs(dw_dos_broadened[:,j]), self.energy, color=n, 
                                      linewidth=line_linewidth, label=f'{m}_{key_list[i]}{dw_symbol}')[0]
                else:
                    for i,n in zip(range(3,len(orbital_index)*4,4),line_color):
                        up_plot = ax.plot(np.abs(up_dos_broadened[:,i]), self.energy, color=n, 
                                      linewidth=line_linewidth, label=f'{m}{up_symbol}')[0]
                        dw_plot = ax.plot(-np.abs(dw_dos_broadened[:,i]), self.energy, color=n, 
                                      linewidth=line_linewidth, label=f'{m}{dw_symbol}')[0]
            else:
                if orbital_index != [3]:
                    
                        #totdos_smoothed = savgol_filter(self.totdos, window_length, polyorder)
                        #totdos_broadened = gaussian_filter1d(totdos_smoothed, sigma=broadening_sigma)
                        #ax.plot(totdos_broadened[:,i], self.energy, color=n, linewidth=line_linewidth)

            self.Fermi = 0
            ax.axhline(y=self.Fermi, color=fermi_color, linestyle='-.', linewidth=fermi_linewidth)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=line_linewidth)
            if xlim is None:
                xlim = [-max_dos, max_dos]
            ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlabel(x_label, fontsize=label_fontsize)
            ax.set_ylabel(y_label, fontsize=label_fontsize)
            ax.set_title(title, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
            ax.legend(fontsize=legend_size,loc='upper right')
            if xtick is not None:
                max_dos = max(np.abs(xlim[0]), np.abs(xlim[1]))  # Max absolute value in xlim
                tick_positions = np.arange(0, max_dos + xtick, xtick)
                tick_positions = tick_positions.astype(int)
                tick_labels = [f"{x}" for x in tick_positions]  # Format with one decimal place
                ax.set_xticks(np.concatenate((-tick_positions[::-1], tick_positions[1:])))  # Mirrored ticks
                ax.set_xticklabels(tick_labels[::-1] + tick_labels[1:])

            if ytick is not None:
                ax.set_yticks(np.arange(ylim[0], ylim[1] + ytick, ytick))
            
        elif orientation == "horizontal":
            if spin_polarization:
                up_plot = ax.plot(self.energy, np.abs(up_dos_broadened), color=line_color[0], 
                        linewidth=line_linewidth, label="Spin-Up")[0]
                dw_plot = ax.plot(self.energy, -np.abs(dw_dos_broadened), color=line_color[1], 
                        linewidth=line_linewidth, label="Spin-Down")[0]
            else:
                totdos_smoothed = savgol_filter(self.totdos, window_length, polyorder)
                totdos_broadened = gaussian_filter1d(totdos_smoothed, sigma=broadening_sigma)
                ax.plot(self.energy, totdos_broadened, color=line_color, linewidth=line_linewidth)
            self.Fermi = 0
            ax.axvline(x=self.Fermi, color=fermi_color, linestyle='-.', linewidth=fermi_linewidth)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=line_linewidth)
            if ylim is None:
                ylim = [-max_dos, max_dos]
            ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_xlabel(x_label, fontsize=label_fontsize)
            ax.set_ylabel(y_label, fontsize=label_fontsize)
            ax.set_title(title, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
            ax.legend(fontsize=legend_size,loc='upper right')
            if ytick is not None:
                max_dos = max(np.abs(ylim[0]), np.abs(ylim[1]))  # Max absolute value in xlim
                tick_positions = np.arange(0, max_dos + ytick, ytick)
                tick_positions = tick_positions.astype(int)
                tick_labels = [f"{y}" for y in tick_positions]  # Format with one decimal place
                ax.set_yticks(np.concatenate((-tick_positions[::-1], tick_positions[1:])))  # Mirrored ticks
                ax.set_yticklabels(tick_labels[::-1] + tick_labels[1:])

            if xtick is not None:
                ax.set_xticks(np.arange(xlim[0], xlim[1] + xtick, xtick))
            
        else:
            raise ValueError("Invalid orientation. Choose 'vertical' or 'horizontal'.")

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
        for spine in ax.spines.values():
            spine.set_linewidth(frame_linewidth)
        else:
            plt.show()        
  