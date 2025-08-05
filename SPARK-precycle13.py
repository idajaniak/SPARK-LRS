#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import ccdproc
from astropy.nddata import CCDData
from astropy import units as u
from ccdproc import Combiner
from ccdproc import ImageFileCollection
import scipy.signal
from astropy.stats import sigma_clip
from numpy import mean
from sklearn.metrics import r2_score
import pandas as pd 
import matplotlib.patches as patches
from astropy.time import Time


# In[ ]:


class DataLoading:
    def __init__(self, data_path, slit_size):
        self.data_path = data_path
        self.slit_size = slit_size
        #saving data paths so these can be called throughout the code to save outputs
        self.intermediate_path = os.path.join(data_path, 'Intermediate files')
        self.calibrated_path = os.path.join(data_path, 'Calibrated files')
        
    #loading the folder from which the files will be retrieved 
    #makes two folders, where the data will be output
    def load_folder(self): 
        #changes the working folder to the one specified by user
        os.chdir(self.data_path) 
        #specifies a directory name
        directory_intmd = 'Intermediate files'
        #retrieves the parent directory specified by the user
        parent_dir = self.data_path
        #creates a path using the parent directory and the one with intermediate files
        path_intmd = os.path.join(parent_dir, directory_intmd)
        try: 
            #makes the directory in the specified location
            os.mkdir(path_intmd)
        except FileExistsError:
            #does nothing if the folder already exists
            pass
        
        #the same for final files 
        directory_final = 'Calibrated files'
        path_final = os.path.join(parent_dir, directory_final)
        try:
            os.mkdir(path_final)
        except FileExistsError:
            #does nothing if the folder already exists
            pass
        
        #plots folder
        directory_plots = 'Plots'
        self.path_plots = os.path.join(path_final, directory_plots)
        try:
            os.mkdir(self.path_plots)
        except FileExistsError:
            pass
        
        directory_plots_ss = 'Sky subtracted collapsed spectra'
        self.path_plots_ss = os.path.join(self.path_plots, directory_plots_ss)
        try:
            os.mkdir(self.path_plots_ss)
        except FileExistsError:
            pass
        
        directory_plots_ns = 'Not sky subtracted collapsed spectra'
        self.path_plots_ns = os.path.join(self.path_plots, directory_plots_ns)
        try:
            os.mkdir(self.path_plots_ns)
        except FileExistsError:
            pass
        
        directory_extracted_regions = 'Extracted regions'
        self.path_extracted_regions = os.path.join(path_final, directory_extracted_regions)
        try:
            os.mkdir(self.path_extracted_regions)
        except FileExistsError:
            pass
        
        directory_wave_ns = 'Not sky subtracted calibrated spectra'
        self.path_plots_wns = os.path.join(self.path_plots, directory_wave_ns)
        try: 
            os.mkdir(self.path_plots_wns)
        except FileExistsError:
            pass
        
        directory_wave_ss = 'Sky subtracted calibrated spectra'
        self.path_plots_wss = os.path.join(self.path_plots, directory_wave_ss)
        try:
            os.mkdir(self.path_plots_wss)
        except FileExistsError:
            pass
        
        return print(os.getcwd(), 'is the working directory')
    
    
    #loading the files in the folder
    #filtering out files which are not .fit/.fits 
    def load_files(self):
        extensions = ('.fit', '.fits') #specifies which extensions are to be included, allows both
        #filters out the files so that only the specified extensions are taken into consideration
        self.files = [file for file in os.listdir(self.data_path) if os.path.splitext(file)[1] in extensions]
    
    #reading the headers and identifying the files
    def read_headers(self):
        #specifies the directory given in load_folder
        #previously, without this statement the code kept changing directories
        working_directory = os.getcwd()
        #moves working directory to the one given by the user
        os.chdir(self.data_path)

        #creates empty arrays to which the different calibration files get appended to
        identified_arc = []
        identified_bias = []
        identified_dark = []
        identified_flat = []
        identified_science = []
        identified_other = []
        #here i am making it so that the array is accessible outside of the read_headers() function
        self.bias_frames = identified_bias
        self.arc_frames = identified_arc
        self.dark_frames = identified_dark
        self.flat_frames = identified_flat
        self.science_frames = identified_science
        self.other_frames = identified_other
        #opens all fits files and accesses the header
        for file in self.files:
            with fits.open(file) as hdul:
                header = hdul[0].header
                if header.get('IMAGETYP') == 'Light Frame':
                    #if the IMAGETYP in the header is 'Light Frame' look for the keywords in the name
                    #as flats haved 'Light Frame' in headers, keys for master flats will be included here
                    keywords_lamps = ['ThAr', 'NeAr', 'Ne','Ar', 'Th', 'HgAr','Hg']
                    keywords_flats = ['master_flat', 'masterflat','flat', 'Flat-Tungsten', 'W']
                    if any(keyword in file for keyword in keywords_lamps):
                        header['IMAGETYP'] = 'Arc'
                        identified_arc.append(file)
                        #if any keywords are found this is not a science frame, but an arc
                        #change the IMAGETP to arc
                    elif any(keyword in file for keyword in keywords_flats):
                        header['IMAGETYP'] = 'Flat'
                        identified_flat.append(file)
                        #check the OBJECT header specifically for 'Flat-Tungsten'
                        if header.get('OBJECT') == 'Flat-Tungsten':
                            identified_flat.append(file)
                    else:
                        identified_science.append(file)
                        #if no lamp keywords then this is a normal science case
                        #be careful with exposures without the slit- these will not get reduced here and if any image
                        #is found, it will also be classified as a science case
                        #delete/move any before attempting data reduction with this code
                elif header.get('IMAGETYP') == 'Bias Frame':
                    identified_bias.append(file)
                elif header.get('IMAGETYP') == 'Flat Field':
                    identified_flat.append(file)
                elif header.get('IMAGETYP') == 'Dark Frame':
                    identified_dark.append(file)
                else:
                    identified_other.append(file)
                
        #these statements print out the found files - sanity check to see if everything was categorised correctly                     
        print('Bias frames:')
        for identified_biases in identified_bias:
            print(identified_biases)
        print('Dark frames:')
        for identified_darks in identified_dark:
            print(identified_darks)
        print('Flat frames:')
        for identified_flats in identified_flat:
            print(identified_flats)
        print('Arcs:')
        for identified_arcs in identified_arc:
            print(identified_arcs)
        print('Science frames:')
        for identified_sciences in identified_science:
            print(identified_sciences)
        print('Other frames:')
        for identified_others in identified_other:
            if (identified_others not in identified_bias and 
                identified_others not in identified_dark and 
                identified_others not in identified_flat and 
                identified_others not in identified_arc and 
                identified_others not in identified_science):
                print(identified_others)
        #moves the directory back to the one specified by user, with all the files
        #had to add these, otherwise issues with switching directories making latter functions not work
        os.chdir(working_directory) 
          

    #stack bias frames, create a master bias, display it
    def master_bias(self):
        #creates an array where the CCDD read bias frames will be added to
        bias_ccdd = []
        #this part of the code looks for any files that were supplied by the user
        #no processing of the pictures apart from assigning them an internal name to use during reduction
        #keywords work on the same basis as in the reading_headers modules
        keywords_bias = ['master_bias', 'masterbias', 'master_bias.fits', 'master_bias.fit']
        for file in self.other_frames:
            if any(keyword in file for keyword in keywords_bias):
                print('Master bias file found in directory:', os.path.basename(file))
                bias_file = CCDData.read(file, unit="adu")
                self.master_bias = bias_file
                return self.master_bias
        #checks for the existence of any files in the self.bias_frames array
        #if there aren't any then the value is 'False' and this step is skipped
        if self.bias_frames:
            print('No master bias file found. Reducing and combining single bias frames.')
            for bias in self.bias_frames:
                bias_files = CCDData.read(bias, unit="adu")
                trimmed_bias = ccdproc.trim_image(bias_files, fits_section = self.slit_size)
                bias_ccdd.append(trimmed_bias) #(bias_files)
            #combines all the CCDD read bias frames into a master bias
            combiner = Combiner(bias_ccdd)
            self.master_bias = combiner.median_combine() 
            #saves the master bias
            self.master_bias.write(os.path.join(self.intermediate_path, 'master_bias.fits'), overwrite=True)
            return self.master_bias
        #if no files in other & bias frames, display an error message
        raise FileNotFoundError('No bias frames found. Cannot proceed with data reduction.')
        
        
    #this module makes master darks based on the exposure time of each frame
    #it initially finds all darks with separate exposure times, groups them together and then stacks them
    #the same procedure is later utilized for the calibration of arcs, flats and science
    #these three will have different times so thats why we need to be able to distinguish darks with 
    #different exposure times
    def master_dark(self):
        
        #arrays into which the different exposure files will be appended to 
        science_darks = []
        
        #find out exposure times
        dark_exposure_times = [fits.getheader(dark)['EXPTIME'] for dark in self.dark_frames]
        science_exposure_times = [fits.getheader(science)['EXPTIME'] for science in self.science_frames]
        #keywords for darks
        keywords_darks_science = ['master_dark_science', 'masterdarkscience','master_dark', 'masterdark']
        
        for file in self.other_frames:
            if any(keyword in file for keyword in keywords_darks_science):
                print('Master dark file for science found in directory:', os.path.basename(file))
                dark_science_file = CCDData.read(file, unit="adu")
                self.debias_science_master_dark = dark_science_file
                return self.debias_science_master_dark
            
        if self.dark_frames:
            print('No master darks files were found. Reducing uncalibrated dark frames.')
            #if exptimes of single dark frames match those of science frames, they will be appended to science_darks
            for dark, dark_exposure in zip(self.dark_frames, dark_exposure_times):

                #checks if darks and science frames have the same time, same applies for other frames
                if dark_exposure in science_exposure_times:
                    science_darks_files =  CCDData.read(dark, unit='adu')
                    #trimming
                    trimmed_sc_d = ccdproc.trim_image(science_darks_files, fits_section = self.slit_size)
                    print('Darks with t_exp = science:', os.path.basename(dark))
                    #science_darks.append(trimmed_sc_d)
                    #subtracting the bias BEFORE stacking darks
                    debiased_sc_d = ccdproc.subtract_bias(trimmed_sc_d, self.master_bias)
                    science_darks.append(debiased_sc_d)
                 
                    #combines all the arc darks into a master dark 
                    combiner_science = Combiner(science_darks)
                    self.debias_science_master_dark = combiner_science.median_combine()
                    #adding bias subtraction 
                   # self.debias_science_master_dark = ccdproc.subtract_bias(self.science_master_dark, self.master_bias)
                    self.debias_science_master_dark.write(os.path.join(self.intermediate_path, 'master_dark_science.fits'), overwrite=True)
                #if no suitable exposures were found then choose a dark with the closest exposure time
                #the data reduction happens in the else statement
                else: 
                    
                    #first, find an absolute value of a difference between exposure times of science and dark frames
                    #then, find the minimum value with argmin
                    closest_exp_science = np.argmin(np.abs(dark_exposure - science_exposure) for science_exposure in science_exposure_times)
                    #selecting the apparopriate frame
                    matching_science_frame = self.dark_frames[closest_exp_science]
                    match_science = self.science_frames[closest_exp_science]
                    #get headers from these frames
                    matching_science_frame_data = fits.getheader(matching_science_frame)
                    match_science_data = fits.getheader(match_science)
                    #prints out that no matching dark was found
                    print('No darks with exposure time matching science frames were found. Using', matching_science_frame, 'instead and scaling it.')
                    #calculate the scaling factor, assume linear increase in dark current: dark exp time/science exp time
                    science_scaling = matching_science_frame_data['EXPTIME']/match_science_data['EXPTIME']
                    #load ccdd
                    science_dark_ccdd =  CCDData.read(matching_science_frame, unit='adu')
                    #trimming
                    trimmed_sc_d = ccdproc.trim_image(science_dark_ccdd, fits_section = self.slit_size)
                    #subtracting bias from the dark 
                    subtract_bias = ccdproc.subtract_bias(trimmed_sc_d, self.master_bias)
                    #scaling
                    self.debias_science_master_dark = subtract_bias.data*science_scaling
                    fits.writeto(os.path.join(self.intermediate_path, 'master_dark_science.fits'), self.debias_science_master_dark, overwrite=True)


            return self.debias_science_master_dark

        #creates a master flat by subtracting bias and darks from the median-combined flat frame
    def master_flat(self):
        #creates an array to which the flats will be CCDD read into
        self.flat_ccdd = []
        keywords_flat = ['master_flat', 'masterflat', 'master_flat.fits', 'masterflat.fit']
        for file in self.other_frames:
            if any(keyword in file for keyword in keywords_flat):
                print('Master flat file found in directory:', os.path.basename(file))
                flat_file = CCDData.read(file, unit="adu")
                self.master_flat = flat_file
                return self.master_flat
        if self.flat_frames:
            print('No master flat file found. Reducing and combining single flat frames.')
            for flat in self.flat_frames:
                flat_files = CCDData.read(flat, unit='adu')
                trimmed_flats = ccdproc.trim_image(flat_files, fits_section = self.slit_size)
                self.flat_ccdd.append(trimmed_flats)
            #stacking up flats by median combining them
            combiner = Combiner(self.flat_ccdd)
            #stores the median combined flat frames
            combined_master_flat = combiner.median_combine()
            #subtracting the master bias from the combined flat, self.master_bias
            debias_flat = ccdproc.subtract_bias(combined_master_flat, self.master_bias)
            
            self.master_flat = debias_flat

            #saving the master flat
            self.master_flat.write(os.path.join(self.intermediate_path, 'master_flat.fits'), overwrite=True)
            return self.master_flat
        raise FileNotFoundError('No flat frames found. Cannot proceed with data reduction.')
        
    def science_reduction(self):
        self.science_ccdd = []
        self.spectrum_science_names = {}
        self.processed_science = []
        #to store time values for photometric set
        self.time_values = []
        
        for science in self.science_frames:
            #PREPARING THE FILES
            #extracting the filename for saving the frame
            science_filename = os.path.basename(science).replace('.fits','')
            
            science_files = CCDData.read(science, unit='adu')
            #get date/time
            date_obs = science_files.header['DATE-OBS']  # Extract DATE-OBS
            self.time_values.append(date_obs)
            
            trimmed_science = ccdproc.trim_image(science_files, fits_section = self.slit_size)
            self.science_ccdd.append(trimmed_science)
            
            #STANDARD REDUCTION
            #subtract bias from individual exposures, we do not stack the single exposures
            debias_science = ccdproc.subtract_bias(trimmed_science, self.master_bias)
            #subtract science master dark
            dark_subtracted_science = ccdproc.subtract_dark(debias_science, self.debias_science_master_dark, dark_exposure=1*u.second, data_exposure=1*u.second, exposure_unit=u.second, scale=False)
            #flat field the science 
            flatfielded_science = ccdproc.flat_correct(dark_subtracted_science, self.master_flat)
            
            #SAVING COMMANDS
            #saving the flat-fielded science frames
            flatfielded_science.write(os.path.join(self.calibrated_path, f"science_reduced_{science_filename}.fits"), overwrite=True)
            #append the data to the list for later processing
            self.processed_science.append(flatfielded_science)
            
        return self.processed_science, self.spectrum_science_names
    
    #reduces arcs so that they can be used for wavelength calibration
    def arcs_reduction(self):
        self.arcs_ccdd = []
        for arcs in self.arc_frames: 
            arc_files = CCDData.read(arcs, unit='adu')
            #trimming arcs so they fit the slit size
            trimmed_arcs = ccdproc.trim_image(arc_files, fits_section = self.slit_size)
            self.arcs_ccdd.append(trimmed_arcs)
    
            #subtract bias from individual exposures, we do not stack the single exposures yet
            debias_arcs = ccdproc.subtract_bias(trimmed_arcs, self.master_bias)
            #flat field the science 
            flatfielded_arcs = ccdproc.flat_correct(debias_arcs, self.master_flat)
            
            final_arcs = flatfielded_arcs
            
            #appending the reduced frames to the list
            self.arcs_ccdd.append(final_arcs)
        
        #stacking up flats by median combining them
        combiner = Combiner(self.arcs_ccdd)
        #combining the frames
        self.combined_master_arc = combiner.median_combine()
        
        #plotting the master arc
        plt.figure(figsize=(10, 5))
        plt.plot(np.nansum(self.combined_master_arc, axis=0))
        plt.xlabel("Pixel Position")
        plt.ylabel("Intensity")
        plt.title("Combined Spectrum")
        plt.show()
        
        #saving the master arc file
        self.combined_master_arc.write(os.path.join(self.intermediate_path, "Combined master arc.fits"), overwrite=True)
        
        return self.combined_master_arc
    
    
    def hgar_calibration(self, order):
        #defining the lamp wavelengths from the TNT AvaLight-Mini lamp
        #16 peaks
        hgar_lamp = [404.66,407.78,435.83,546.07,578.02,696.54,706.72,714.70,727.29,738.40,750.39,763.51,772.38,794.82,811.53,826.45]
        
        #consults the user for the initial
        threshold = float(input('What is your preferred peak threshold? Remember, you need to set it so that ONLY 16 peaks are found.'))
        
        #collapse the master arc
        self.spectrum_arcs = np.nansum(self.combined_master_arc, axis=0)
        
        while True:
            #find peaks in the spectrum
            peaks, _ = scipy.signal.find_peaks(self.spectrum_arcs, height=threshold)
            if len(peaks) == 16:
                #yay, 16 peaks have been found
                break
            else:
                #ask for a new threshold
                print(f"Found {len(peaks)} peaks. Please adjust the threshold to find exactly 16 peaks.")
                threshold = float(input('Enter a new threshold value: '))
        
        #use the 16 peaks (in the order they were detected)
        peaks = peaks[:16]
        
        #match the detected peaks with the known wavelengths in order
        wavelength_peak_values = dict(zip(peaks, hgar_lamp))
                
        #for a sanity check print out the matching values
        print('These pixel-wavelength pairs were identified:')
        for position, wave in wavelength_peak_values.items():
            print(f'{position} - {wave}')
        
        #saving the pixel-wavelength pairs to a CSV file
        df = pd.DataFrame(list(wavelength_peak_values.items()), columns=['Pixel Position', 'Wavelength (nm)'])
        df.to_csv(os.path.join(self.data_path, 'hgar_wavelength_calibration.csv'), index=False)
        print("Pixel-wavelength pairs saved to 'hgar_wavelength_calibration.csv'.")
        
        #4. apply fitting
        #separates the pairs and creates lists for the values separately
        pixels = list(wavelength_peak_values.keys())
        wavelengths = list(wavelength_peak_values.values())
        
        #fit the function using polynomial fitting
        #deg=1, indicating linear fit, 2 - squared and 3 - cubic
        if order =='linear':
            #if not enough parameters, then the program won't attempt the wavelength calibration
            if len(pixels) < 2:
                print("Warning: Not enough data points for a linear fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=1)
        elif order =='quadratic':
            if len(pixels) < 3:
                print("Warning: Not enough data points for a quadratic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=2)
        elif order =='cubic':
            if len(pixels) <4:
                print("Warning: Not enough data points for a cubic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=3)
        else: raise ValueError('Invalid polynomial order.')
        
        #creates a model using the coefficients from polyfit
        self.wavelength_model = np.poly1d(coefficients)
        #display the goodness of fit to the user
        print('R-squared of the fit:', r2_score(wavelengths ,self.wavelength_model(pixels)))
        #applies the model to the real spectrum 
        calibrated_peaks = self.wavelength_model(np.arange(len(self.spectrum_arcs)))
    
        #5. plotting the lamp spectra to check if it is correct
        #plot the original data and the linear fit
        plt.figure(figsize=(10, 5))
        plt.plot(calibrated_peaks, self.spectrum_arcs)
        plt.xlabel("Wavelength (Nanometres)")
        plt.ylabel("Intensity")
        plt.title('Calibrated HgAr lamp spectrum')
        plt.show()
        
        #returns the polynomial model
        return self.wavelength_model
        
        
    def neon_calibration(self, order): 
        #defining the known wavelengths for AvaLight Mini Neon lamp
        #21 peaks (excluding the two small ones after 743.89 nm)
        neon_lamp = [585.25,588.19,594.48,597.55,603.00,609.62,614.31,621.73,626.65,630.48,633.44,640.22,650.65,659.90,667.83,671.70,692.95,703.24,717.39,724.52,743.89]
        #consults the user for the initial
        threshold = float(input('What is your preferred peak threshold? Remember, you need to set it so that ONLY 16 peaks are found.'))
        
        #collapse the master arc
        self.spectrum_arcs = np.nansum(self.combined_master_arc, axis=0)
        
        while True:
            #find peaks in the spectrum
            peaks, _ = scipy.signal.find_peaks(self.spectrum_arcs, height=threshold)
            if len(peaks) == 21:
                #yay, 21 peaks have been found
                break
            else:
                #ask for a new threshold
                print(f"Found {len(peaks)} peaks. Please adjust the threshold to find exactly 21 peaks.")
                threshold = float(input('Enter a new threshold value: '))
        
        #use the 21 peaks (in the order they were detected)
        peaks = peaks[:21]
        
        #match the detected peaks with the known wavelengths in order
        wavelength_peak_values = dict(zip(peaks, neon_lamp))
                
        #for a sanity check print out the matching values
        print('These pixel-wavelength pairs were identified:')
        for position, wave in wavelength_peak_values.items():
            print(f'{position} - {wave}')
        
        #saving the pixel-wavelength pairs to a CSV file
        df = pd.DataFrame(list(wavelength_peak_values.items()), columns=['Pixel Position', 'Wavelength (nm)'])
        df.to_csv(os.path.join(self.data_path, 'neon_wavelength_calibration.csv'), index=False)
        print("Pixel-wavelength pairs saved to 'neon_wavelength_calibration.csv'.")
        
        #4. apply fitting
        #separates the pairs and creates lists for the values separately
        pixels = list(wavelength_peak_values.keys())
        wavelengths = list(wavelength_peak_values.values())
        
        #fit the function using polynomial fitting
        #deg=1, indicating linear fit, 2 - squared and 3 - cubic
        if order =='linear':
            #if not enough parameters, then the program won't attempt the wavelength calibration
            if len(pixels) < 2:
                print("Warning: Not enough data points for a linear fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=1)
        elif order =='quadratic':
            if len(pixels) < 3:
                print("Warning: Not enough data points for a quadratic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=2)
        elif order =='cubic':
            if len(pixels) <4:
                print("Warning: Not enough data points for a cubic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=3)
        else: raise ValueError('Invalid polynomial order.')
        
        #creates a model using the coefficients from polyfit
        self.wavelength_model = np.poly1d(coefficients)
        #display the goodness of fit to the user
        print('R-squared of the fit:', r2_score(wavelengths ,self.wavelength_model(pixels)))
        #applies the model to the real spectrum 
        calibrated_peaks = self.wavelength_model(np.arange(len(self.spectrum_arcs)))
    
        #5. plotting the lamp spectra to check if it is correct
        # Plot the original data and the linear fit
        plt.figure(figsize=(10, 5))
        plt.plot(calibrated_peaks, self.spectrum_arcs)
        plt.xlabel("Wavelength (Nanometres)")
        plt.ylabel("Intensity")
        plt.title('Calibrated Neon lamp spectrum')
        plt.show()
        
        #returns the polynomial model
        return self.wavelength_model
    
    
    
    def wavelength_calibration(self, order):
        #look for the possibly existing wavelength calibration file in the parent directory
        csv_path = os.path.join(self.data_path, 'wavelength_calibration.csv')
        
        #check if the wavelength calibration file exists
        if os.path.isfile(csv_path):
            #inform the user
            print(f"Found existing wavelength calibration file: {csv_path}")
            df = pd.read_csv(csv_path)
            #convert to dictionary
            wavelength_peak_values = dict(zip(df['Pixel Position'], df['Wavelength (nm)']))
            print('Loaded pixel-wavelength pairs from the file:')
            for position, wave in wavelength_peak_values.items():
                print(f'{position} - {wave}')
        else:
            #to this dictionary key-value pairs of peak-wavelength will be assigned
            wavelength_peak_values = {}

            #1. deciding the level of peaks, user input needed!
            #consults the user for peak threshold which can be specified by the user, transforms the user response into an integer
            threshold = float(input('What is your preferred peak threshold?'))

            #2. finding peaks in the stacked spectrum  
            #only peaks relevant, so '_' refers to other values also returned by the function but not used here
            #_ - properties, e.g., the width are not of use here
            self.spectrum_arcs = np.nansum(self.combined_master_arc, axis=0)
            peaks, _ = scipy.signal.find_peaks(self.spectrum_arcs, height= threshold)
            #plots the spectrum with the detected peaks indicated by dots
            plt.figure(figsize=(10, 5))
            plt.plot(self.spectrum_arcs)
            plt.plot(peaks, self.spectrum_arcs[peaks], "ro", markersize=3)
            for i, peak in enumerate(peaks):
                plt.text(peak, self.spectrum_arcs[peak] , str(i+1), fontsize=12, color='blue', ha='center', va='bottom')
                #consults the user for input and asks what wavelength is associated with each peak
                #converts the output from a list to a float
            plt.xlabel("Pixel position")
            plt.ylabel("Intensity")
            plt.title("Collapsed spectrum with detected peaks")
            plt.show()
            #assigns a number to each peak so that it is displayed next to the peak 
            for i, peak in enumerate(peaks):
                wavelength = float(input(f"What is the wavelength associated with peak {i+1} (pixel position {peak})? "))
                #saves the peak position in x to a matching wavelength given by the user
                wavelength_peak_values[peak] = wavelength

            #3. for sanity checks print out the matching values
            print('These pixel-wavelength pairs were identified:')
            for position, wave in wavelength_peak_values.items():
                print(f'{position} - {wave}')

            #saving the pixel-wavelength pairs to a CSV file
            df = pd.DataFrame(list(wavelength_peak_values.items()), columns=['Pixel Position', 'Wavelength (nm)'])
            df.to_csv(os.path.join(self.data_path, 'wavelength_calibration.csv'), index=False)
            print("Pixel-wavelength pairs saved to 'wavelength_calibration.csv'.")

        #4. apply fitting
        #separates the pairs and creates lists for the values separately
        pixels = list(wavelength_peak_values.keys())
        wavelengths = list(wavelength_peak_values.values())
        
        #fit the function using polynomial fitting
        #deg=1, indicating linear fit, 2 - squared and 3 - cubic
        if order =='linear':
            #if not enough parameters, then the program won't attempt the wavelength calibration
            if len(pixels) < 2:
                print("Warning: Not enough data points for a linear fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=1)
        elif order =='quadratic':
            if len(pixels) < 3:
                print("Warning: Not enough data points for a quadratic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=2)
        elif order =='cubic':
            if len(pixels) <4:
                print("Warning: Not enough data points for a cubic fit.")
                return None
            coefficients = np.polyfit(pixels, wavelengths, deg=3)
        else: raise ValueError('Invalid polynomial order.')
        
        #creates a model using the coefficients from polyfit
        self.wavelength_model = np.poly1d(coefficients)
        #display the goodness of fit to the user
        print('R-squared of the fit:', r2_score(wavelengths ,self.wavelength_model(pixels)))
        #applies the model to the real spectrum 
        calibrated_peaks = self.wavelength_model(np.arange(len(self.spectrum_arcs)))
    
        #5. plotting the lamp spectra to check if it is correct
        plt.figure(figsize=(10, 5))
        plt.plot(calibrated_peaks, self.spectrum_arcs)
        plt.xlabel("Wavelength (Nanometres)")
        plt.ylabel("Intensity")
        plt.title('Calibrated collapsed lamp spectrum')
        plt.show()
        
        #returns the polynomial model
        return self.wavelength_model
    
    #to detect and extract only one spectrum
    def detect_and_extract_1region(self, image, centre, spectrum_width, spectrum_length):
        #image will be defined in the single spec extract function 
        #this is an individual function unrelated to the bulk and will not be used on its own but will be incorporated 
        y_pos, x_pos = image.shape

        #1. values for establishing the slit
        #CENTRE
        #automatic detection of the peak/the spectrum if not defined by the user
        if centre is None:
            row_sums = np.sum(image, axis=1)
            peaks, properties = scipy.signal.find_peaks(row_sums, height=np.nanmedian(row_sums)*2, width=1)

            if len(peaks) == 0:
                print("No peaks found. Defaulting to the middle of the image.")
                spectrum_row = centre0
            else:
                peak_id = np.argmax(properties["peak_heights"])
                spectrum_row = peaks[peak_id]
                print(f"Automatically determined spectrum centre at row {spectrum_row}.")
        else:
            spectrum_row = centre
            print(f"Using provided centre at row {spectrum_row}.")

        #HOW WIDE IS THE SLIT, WHAT REGION AROUND THE CENTRE WILL BE EXTRACTED
        #check if spectrum_width is provided, if not default to 30 (revisit once bright data was obtained!!!!)
        if spectrum_width is None:
            spectrum_width = 30
            print(f"No spectrum width provided. Using default width of {spectrum_width} pixels.")
        else:
            print(f"Using provided spectrum width of {spectrum_width} pixels.")

        
        #centres at the middle of the image, keep that in mind if you don't want to consider ends of the image
        #check if spectrum_length is provided, else default to the full image length
        if isinstance(spectrum_length, slice):
            start_col = max(0, spectrum_length.start)  
            end_col = min(x_pos, spectrum_length.stop) 
            print(f"Using provided spectrum length range: {spectrum_length.start} to {spectrum_length.stop} pixels.")
        elif spectrum_length is None:
            start_col = 0
            end_col = x_pos
            print(f"No spectrum length provided. Using default length of {spectrum_length} pixels (full image length).")
        else:
            start_col = 0
            end_col = spectrum_length
            print(f"Using provided spectrum length from 0 to {spectrum_length} pixels.")

        #2. set the boundaries of the region using the specific parameters
        #calculate the start and end rows for the extraction
        #divides the width in half and adds/subtracts each from the central strip 
        #if outside of the spetrum then the lower boundary defaults to 0 (starting row of pixels)
        start_row = max(0, spectrum_row - spectrum_width // 2)
        #if outside of the image boundaries, defaults to the highest (largest) row possible
        end_row = min(image.shape[0], spectrum_row + spectrum_width // 2)

        #extracting the region of interest
        extracted_region = image[start_row:end_row, start_col:end_col]

        print(f"Extracted region from row {start_row} to {end_row}, and from column {start_col} to {end_col}.")
        return extracted_region, spectrum_row

    
    
    #detect and extract 2 SPECTRA
    def detect_and_extract_2regions(self, image, centre=None, spectrum_width=None, spectrum_length=None):
        #stores image dimensions
        y_pos, x_pos = image.shape

        #detecting spectra (target and comparison)
        if centre is None:
            #collapse image along columns (sum or average intensity across rows)
            row_sums = np.sum(image, axis=1)

            #detect peaks in the collapsed 1D array
            peaks, properties = scipy.signal.find_peaks(row_sums, height=np.nanmedian(row_sums)*1, width=1) #from *2
            print(peaks)

            if len(peaks) == 0:
                #no peaks found, default to the middle of the image
                spectrum_row_1 = y_pos // 2  #middle of the image
                spectrum_row_2 = min(y_pos - 1, spectrum_row_1 + 20)  #default second spectrum offset
            elif len(peaks) == 1:
                #only one peak found, assign second spectrum as offset
                spectrum_row_1 = peaks[0]
                spectrum_row_2 = min(y_pos - 1, spectrum_row_1 + 20)  #default second spectrum offset
                print(f"Only one peak found. Setting target spectrum at row {spectrum_row_1} and comparison spectrum to row {spectrum_row_2}.")
            else:
                #sort peaks by intensity and assign first and second spectra
                intensities = properties['peak_heights']
                sorted_indices = np.argsort(intensities)[::-1]  #sort in descending order
                spectrum_row_1 = peaks[sorted_indices[0]]  #highest intensity - target
                spectrum_row_2 = peaks[sorted_indices[1]]  #second highest intensity - comparison
                print(f"Automatically determined target spectrum at row {spectrum_row_1} and comparison spectrum at row {spectrum_row_2}.")
        else:
            #use provided center
            spectrum_row_1 = centre
            spectrum_row_2 = min(y_pos - 1, centre + 20)  #default second spectrum offset
            print(f"Using provided centre for target spectrum at row {spectrum_row_1}. Setting comparison spectrum to row {spectrum_row_2}.")

        #store rows for further processing
        self.spectrum_row_1 = spectrum_row_1
        self.spectrum_row_2 = spectrum_row_2
        
        #3. finding spectrum width
        if spectrum_width is None:
            spectrum_width = 30
            print(f"No spectrum width provided. Using default width of {spectrum_width} pixels.")
        else:
            print(f"Using provided spectrum width of {spectrum_width} pixels.")

        #4. determining and establishing spectrum length
        if isinstance(spectrum_length, slice):
            start_col = spectrum_length.start or 0
            end_col = spectrum_length.stop or image.shape[1]
            print(f"Spectrum length provided as a slice: {spectrum_length.start}:{spectrum_length.stop}.")
        elif spectrum_length is None:
            spectrum_length = x_pos
            start_col = 0
            end_col = x_pos
            print(f"No spectrum length provided. Using default length of {spectrum_length} pixels (full image length).")
        else:
            #in case that a user wants to provide a spectrum length and make sure that the length is ALWAYS centered
            start_col = max(0, (x_pos-spectrum_length)//2)
            end_col = start_col + spectrum_length
            print(f"Using provided spectrum length of {spectrum_length} pixels.")

        #EXTRACTING THE SLIT
        #5. define boundaries and extract regions

        #target spectrum or spectrum no. 1
        start_row_1 = max(0, spectrum_row_1 - spectrum_width // 2)
        end_row_1 = min(y_pos, spectrum_row_1 + spectrum_width // 2)
        
        extracted_region_1 = image[start_row_1:end_row_1, start_col:end_col]
        print(f"Extracted target region from row {start_row_1} to {end_row_1}.")

        #comparison spectrum or spectrum no. 2
        start_row_2 = max(0, spectrum_row_2 - spectrum_width // 2)
        end_row_2 = min(y_pos, spectrum_row_2 + spectrum_width // 2)

        extracted_region_2 = image[start_row_2:end_row_2, start_col:end_col]
        print(f"Extracted comparison region from row {start_row_2} to {end_row_2}.")

        #returning extracted regions and their centers
        return (extracted_region_1, spectrum_row_1), (extracted_region_2, spectrum_row_2)



    

    #to sky subtract an image of just one spectrum
    def sky_subtraction_1spectrum(self, extracted_region, spectrum_row, image, gap, sky_width, spectrum_length):
        #1. establishing sky dimensions
        #check if gap parameter provided, if not default gap between spectrum and sky regions (revisit when data there)
        #here gap is the separation from the spectral region established in the previous function
        if gap is None:
            gap = 30
            print(f"No gap provided. Using default gap of {gap} pixels.")
        else:
            print(f"Using provided gap of {gap} pixels.")

        #check if sky width is established, if not default width for the sky regions (revisit when data is out)
        if sky_width is None:
            sky_width = 10
            print(f"No sky width provided. Using default sky width of {sky_width} pixels.")
        else:
            print(f"Using provided sky width of {sky_width} pixels.")


        #2. take slices on either side of the spectrum 
        #sky below the spectrum
        #int to assure the pixel number/boundary is an integer
        imin_sky1 = int(spectrum_row - (extracted_region.shape[0] // 2 + gap + sky_width))
        imax_sky1 = int(spectrum_row - (extracted_region.shape[0] // 2 + gap))
        #sky above the spectrum
        imin_sky2 = int(spectrum_row + (extracted_region.shape[0] // 2 + gap))
        imax_sky2 = int(spectrum_row + (extracted_region.shape[0] // 2 + gap + sky_width))
        #to make sure that the indices are within the bounds of the image
        #if min value below 0, default to 0 as the lowest
        imin_sky1 = max(0, imin_sky1)
        #if max value exceeds the max dimension of the image, default to that max dimension
        imax_sky1 = min(image.shape[0], imax_sky1)
        #repeat for the other slice
        imin_sky2 = max(0, imin_sky2)
        imax_sky2 = min(image.shape[0], imax_sky2)

        #3. approximate the sky
        #take a median of the sky background for both regions
        sky1 = np.nanmedian(image[imin_sky1:imax_sky1, :], axis=0)
        sky2 = np.nanmedian(image[imin_sky2:imax_sky2, :], axis=0)
        #interpolate sky by taking the mean of the two sky regions
        sky_background = (sky1 + sky2) / 2
        print(f"Sky background calculated using regions from row {imin_sky1} to {imax_sky1} and {imin_sky2} to {imax_sky2}.")

        if isinstance(spectrum_length, slice):
            start = spectrum_length.start
            stop = spectrum_length.stop
        else:
            raise ValueError("spectrum_length must be a slice object.")

        #match the dimensions
        #slice the sky background to match the spectrum length
        sky_background_adjusted = sky_background[start:stop]
        
        #subtract the sky background from the extracted region
        sky_subtracted_spectrum = np.nansum(extracted_region - sky_background_adjusted, axis=0)

        return sky_subtracted_spectrum

    
    #different to the 1 spectrum sky subtraction
    def sky_subtraction_2spectra(self, extracted_region1, extracted_region2, peak1, peak2, image, spectrum_length, manual_sky_region=None):
        #to make sure the detection is good and for debugging
        print(f"Peak1: {self.spectrum_row_1}")
        print(f"Peak2: {self.spectrum_row_2}")
        
        if manual_sky_region is not None:
            #use manualdef sky region
            sky_start, sky_end = manual_sky_region
            print(f"Using manual sky region from row {sky_start} to {sky_end}.")
        
        else:
            #automatic sky selection :)
            #calculate the middle/centre between the two peaks
            mid_row = (peak1 + peak2) // 2

            #define the sky region: 10 pixels above and below the midpoint by default
            sky_start = max(0, mid_row - 10)
            sky_end = min(image.shape[0], mid_row + 10)
            print(f"Sky background calculated using rows from {sky_start} to {sky_end}.")

        #calculate the sky background by taking the median along the sky region
        sky_background = np.nanmedian(image[sky_start:sky_end, :], axis=0)
        print(f"Sky background calculated using rows from {sky_start} to {sky_end}.")

        #adjust `sky_background` to match the dimensions of `extracted_region1` and `extracted_region2`
        #slicing `sky_background` to match the length of the extracted spectral region
        sky_background1 = sky_background[spectrum_length]
        sky_background2 = sky_background[spectrum_length]

        # subtract the sky background from both spectral regions
        self.sky_subtracted_spectrum1 = np.nansum(extracted_region1 - sky_background1, axis=0)
        self.sky_subtracted_spectrum2 = np.nansum(extracted_region2 - sky_background2, axis=0)

        return self.sky_subtracted_spectrum1, self.sky_subtracted_spectrum2



        
    #to extract a singular line of spectra, from just one object
    def spec_extraction_1spectrum(self, centre, spectrum_width=None, spectrum_length=None, gap=None, sky_width=None):
        #adding these two functions together to extract and clean up a spectrum
        self.extracted_regions = []
        self.collapsed_spectra = []
        self.sky_subtracted_spectra = [] 

        for i, frame in enumerate(self.processed_science):
            image = frame.data
            #detect and extract region with the independent function
            extracted_region, spectrum_row = self.detect_and_extract_1region(image, centre, spectrum_width, spectrum_length)
            #subtract sky background, saves as sky
            sky = self.sky_subtraction_1spectrum(extracted_region, spectrum_row, image, gap, sky_width, spectrum_length) 

            #sigma clipping and collapsing the spectrum
            sigma_clipped_region = sigma_clip(extracted_region, sigma=3.0, maxiters=10)
            collapsed_spectrum = np.sum(sigma_clipped_region, axis=0)

            #store extracted regions and spectra
            self.extracted_regions.append(extracted_region)
            self.collapsed_spectra.append(collapsed_spectrum)
            self.sky_subtracted_spectra.append(sky)

            #get the original filename
            original_filename = os.path.basename(self.science_frames[i]).replace('.fits', '')

            #save extracted region as FITS
            output_region_path = os.path.join(self.path_extracted_regions, f"extracted_region_{original_filename}.fits")
            fits.PrimaryHDU(extracted_region).writeto(output_region_path, overwrite=True)
            print(f"Extracted region saved as FITS: {output_region_path}")

            #save collapsed spectrum as a graph (not sky subtracted)
            output_collapsed_path = os.path.join(self.path_plots_ns, f"collapsed_spectrum_{original_filename}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(collapsed_spectrum, label="Collapsed Spectrum", color="pink", lw=1.5)
            plt.xlabel("Pixel Column")
            plt.ylabel("Summed Intensity")
            plt.title(f"Collapsed Spectrum for {original_filename}")
            plt.legend()
            plt.savefig(output_collapsed_path)
            plt.close()
            print(f"Collapsed spectrum graph saved as: {output_collapsed_path}")

            #save sky-subtracted spectrum as a graph
            output_sky_graph_path = os.path.join(self.path_plots_ss, f"sky_subtracted_spectrum_{original_filename}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(sky, label="Sky-subtracted Spectrum", color="blue", lw=1.5)
            plt.xlabel("Pixel Column")
            plt.ylabel("Intensity")
            plt.title(f"Sky-subtracted Spectrum for {original_filename}")
            plt.legend()
            plt.savefig(output_sky_graph_path)
            plt.close()
            print(f"Sky-subtracted spectrum graph saved as: {output_sky_graph_path}")

        return self.extracted_regions, self.collapsed_spectra, self.sky_subtracted_spectra

    
    def spec_extraction_2spectra(self, centre=None, spectrum_width=20, spectrum_length=None,manual_sky_region=None):
        #stoooring
        self.extracted_regions = []
        self.collapsed_spectra = []
        self.sky_subtracted_spectra = []

        for i, frame in enumerate(self.processed_science):
            image = frame.data
            #using detect_and_extract_2regions to extract spectral regions
            (extracted_region1, spectrum_row1), (extracted_region2, spectrum_row2) = self.detect_and_extract_2regions(image, centre, spectrum_width, spectrum_length)
            #performing sky subtraction for the extracted spectral regions
            sky_subtracted_spectrum1, sky_subtracted_spectrum2 = self.sky_subtraction_2spectra(extracted_region1, extracted_region2, spectrum_row1, spectrum_row2, image, spectrum_length, manual_sky_region)
            #add the extracted regions to the list
            self.extracted_regions.append((extracted_region1, extracted_region2))
            
            for extracted_region in [extracted_region1, extracted_region2]:
                sigma_clipped_region = sigma_clip(extracted_region, sigma=3.0, maxiters=10)
                collapsed_spectrum = np.sum(sigma_clipped_region, axis=0)
                self.collapsed_spectra.append(collapsed_spectrum)

            #storing sky-subtracted spectra
            self.sky_subtracted_spectra.append((sky_subtracted_spectrum1, sky_subtracted_spectrum2))

            #get the original filename
            original_filename = os.path.basename(self.science_frames[i]).replace('.fits', '')

            #save results for both spectra
            for idx, (extracted_regions, collapsed_spectrum, sky_subtracted_spectrum) in enumerate(zip(self.extracted_regions[-1], self.collapsed_spectra[-2:], self.sky_subtracted_spectra[-1])):
                suffix = f"spectrum_{idx + 1}"

                #save extracted region as FITS
                output_region_path = os.path.join(self.path_extracted_regions, f"extracted_region_{suffix}_{original_filename}.fits")
                fits.PrimaryHDU(extracted_regions).writeto(output_region_path, overwrite=True)
                print(f"Extracted region saved as FITS: {output_region_path}")

                #save collapsed spectrum as a graph (not sky subtracted)
                #output_collapsed_path = f"{self.data_path}/collapsed_spectrum_{suffix}_{original_filename}.png"
                output_collapsed_path = os.path.join(self.path_plots_ns, f"collapsed_spectrum_{suffix}_{original_filename}.png")
                plt.figure(figsize=(12, 6))
                plt.plot(collapsed_spectrum, label="Collapsed Spectrum", color="pink", lw=1.5)
                plt.xlabel("Pixel Column")
                plt.ylabel("Summed Intensity")
                plt.title(f"Collapsed Spectrum {suffix} for {original_filename}")
                plt.legend()
                plt.savefig(output_collapsed_path)
                plt.close()
                print(f"Collapsed spectrum graph saved as: {output_collapsed_path}")

                #save sky-subtracted spectrum as FITS
                #output_sky_path = f"{self.data_path}/sky_subtracted_spectrum_{suffix}_{original_filename}.fits"
                output_sky_graph_path = os.path.join(self.path_plots_ss, f"sky_subtracted_spectrum_{suffix}_{original_filename}.png")
                plt.figure(figsize=(12, 6))
                plt.plot(sky_subtracted_spectrum, label="Sky-subtracted Spectrum", color="blue", lw=1.5)
                plt.xlabel("Pixel Column")
                plt.ylabel("Intensity")
                plt.title(f"Sky-subtracted Spectrum {suffix} for {original_filename}")
                plt.legend()
                plt.savefig(output_sky_graph_path)
                plt.close()
                print(f"Sky-subtracted spectrum graph saved as: {output_sky_graph_path}")
                
        return self.extracted_regions, self.collapsed_spectra, self.sky_subtracted_spectra

    
    def apply_wavelength_calibration(self, spectrum_length):
        #check if the wavelength model is available
        if self.wavelength_model is None:
            raise ValueError("Wavelength model is not available. Run wavelength calibration first.")

        #prepare lists to store the wavelength-calibrated spectra
        calibrated_spectra = []
        calibrated_sky_subtracted_spectra = []

        for i, collapsed_spectrum in enumerate(self.collapsed_spectra):
            #apply the wavelength model to pixel positions on the x-axis
            pixel_positions = np.arange(len(collapsed_spectrum))
            calibrated_wavelengths = self.wavelength_model(pixel_positions)

            #retrieve the sky-subtracted spectrum for the current frame
            sky_subtracted_spectrum = np.array(self.sky_subtracted_spectra[i])  #ensure numpy array
            
            #adjusting the length according to provided spectrum length
            calibrated_wavelengths = calibrated_wavelengths[spectrum_length]
            sky_subtracted_spectrum = sky_subtracted_spectrum[spectrum_length]
            collapsed_spectrum = collapsed_spectrum[spectrum_length]
            

            #getting the file name
            original_filename = os.path.basename(self.science_frames[i]).replace('.fits', '')

            #append the calibrated spectra
            calibrated_spectra.append(calibrated_wavelengths)
            calibrated_sky_subtracted_spectra.append(sky_subtracted_spectrum)

            #plot and save the sky-subtracted spectrum
            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths, sky_subtracted_spectrum, label='Sky-Subtracted Spectrum', color='hotpink', lw=1.5)
            plt.xlabel('Wavelength (Nanometres)')
            plt.ylabel('Intensity')
            plt.title(f'Calibrated Sky-Subtracted Spectrum for {original_filename}')
            plt.legend()

            #save in the appropriate folder
            sky_subtracted_plot_path = os.path.join(self.path_plots_wss, f"calibrated_sky_subtracted_spectrum_{original_filename}.png")
            plt.savefig(sky_subtracted_plot_path)
            plt.close()
            print(f"Sky-subtracted spectrum graph saved as: {sky_subtracted_plot_path}")

            #plot and save the non-sky-subtracted spectrum
            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths, collapsed_spectrum, label='Non Sky-Subtracted Spectrum', color='pink', lw=1.5)
            plt.xlabel('Wavelength (Nanometres)')
            plt.ylabel('Intensity')
            plt.title(f'Calibrated Spectrum for {original_filename}')
            plt.legend()

            #save in the appropriate folder
            non_sky_subtracted_plot_path = os.path.join(self.path_plots_wns, f"calibrated_non_sky_subtracted_spectrum_{original_filename}.png")
            plt.savefig(non_sky_subtracted_plot_path)
            plt.close()
            print(f"Non-sky-subtracted spectrum graph saved as: {non_sky_subtracted_plot_path}")

        return calibrated_spectra, calibrated_sky_subtracted_spectra

    
    def apply_wavelength_calibration_2spectra(self, spectrum_length):
        #check if the wavelength model is available
        if self.wavelength_model is None:
            raise ValueError("Wavelength model is not available. Run wavelength calibration first.")

        #prepare lists to store the wavelength-calibrated spectra
        calibrated_spectra_1 = []
        calibrated_spectra_2 = []
        calibrated_sky_subtracted_spectra_1 = []
        calibrated_sky_subtracted_spectra_2 = []

        #loop over each pair of spectra (collapsed and sky-subtracted)
        for i in range(len(self.collapsed_spectra) // 2): 
            #get collapsed spectra for both regions
            collapsed_spectrum_1 = self.collapsed_spectra[2 * i]
            collapsed_spectrum_2 = self.collapsed_spectra[2 * i + 1]

            #unpack and do so that sky-subtracted spectra are numpy arrays
            sky_subtracted_spectrum_1, sky_subtracted_spectrum_2 = self.sky_subtracted_spectra[i]
            sky_subtracted_spectrum_1 = np.array(sky_subtracted_spectrum_1)
            sky_subtracted_spectrum_2 = np.array(sky_subtracted_spectrum_2)

            #apply the wavelength model to pixel positions 
            #this code was changed to allow to be adjusted to the new range after being sliced by spectrum_length
            pixel_positions_1 = np.arange(len(collapsed_spectrum_1))[spectrum_length]
            pixel_positions_2 = np.arange(len(collapsed_spectrum_2))[spectrum_length]
            calibrated_wavelengths_1 = self.wavelength_model(pixel_positions_1)
            calibrated_wavelengths_2 = self.wavelength_model(pixel_positions_2)
            
            sky_subtracted_spectrum_1 = sky_subtracted_spectrum_1[spectrum_length]
            sky_subtracted_spectrum_2 = sky_subtracted_spectrum_2[spectrum_length]
            collapsed_spectrum_1 = collapsed_spectrum_1[spectrum_length]  #match length for non sky-subtracted spectrum
            collapsed_spectrum_2 = collapsed_spectrum_2[spectrum_length]

            #append the calibrated spectra
            calibrated_spectra_1.append(calibrated_wavelengths_1)
            calibrated_spectra_2.append(calibrated_wavelengths_2)
            calibrated_sky_subtracted_spectra_1.append(sky_subtracted_spectrum_1)
            calibrated_sky_subtracted_spectra_2.append(sky_subtracted_spectrum_2)

            #get the original filename
            original_filename = os.path.basename(self.science_frames[i]).replace('.fits', '')

            #plot and save for spectrum 1
            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths_1, sky_subtracted_spectrum_1, label="Sky-Subtracted Spectrum", color="hotpink", lw=1.5)
            plt.xlabel("Wavelength (Nanometres)")
            plt.ylabel("Intensity")
            plt.title(f"Calibrated Sky-Subtracted Spectrum 1 for {original_filename}")
            plt.legend()
            sky_subtracted_plot_path_1 = os.path.join(self.path_plots_wss, f"calibrated_sky_subtracted_spectrum_1_{original_filename}.png")
            plt.savefig(sky_subtracted_plot_path_1)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths_1, collapsed_spectrum_1, label="Non Sky-Subtracted Spectrum", color="pink", lw=1.5)
            plt.xlabel("Wavelength (Nanometres)")
            plt.ylabel("Intensity")
            plt.title(f"Calibrated Spectrum 1 for {original_filename}")
            plt.legend()
            non_sky_subtracted_plot_path_1 = os.path.join(self.path_plots_wns, f"calibrated_non_sky_subtracted_spectrum_1_{original_filename}.png")
            plt.savefig(non_sky_subtracted_plot_path_1)
            plt.close()

            #plot and save for Spectrum 2
            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths_2, sky_subtracted_spectrum_2, label="Sky-Subtracted Spectrum", color="purple", lw=1.5)
            plt.xlabel("Wavelength (Nm)")
            plt.ylabel("Intensity")
            plt.title(f"Calibrated Sky-Subtracted Spectrum 2 for {original_filename}")
            plt.legend()
            sky_subtracted_plot_path_2 = os.path.join(self.path_plots_wss, f"calibrated_sky_subtracted_spectrum_2_{original_filename}.png")
            plt.savefig(sky_subtracted_plot_path_2)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(calibrated_wavelengths_2, collapsed_spectrum_2, label="Non Sky-Subtracted Spectrum", color="purple", lw=1.5)
            plt.xlabel("Wavelength (Nm)")
            plt.ylabel("Intensity")
            plt.title(f"Calibrated Spectrum 2 for {original_filename}")
            plt.legend()
            non_sky_subtracted_plot_path_2 = os.path.join(self.path_plots_wns, f"calibrated_non_sky_subtracted_spectrum_2_{original_filename}.png")
            plt.savefig(non_sky_subtracted_plot_path_2)
            plt.close()
            
        return (calibrated_spectra_1, calibrated_spectra_2, calibrated_sky_subtracted_spectra_1, calibrated_sky_subtracted_spectra_2,)

    
    #wrapper functions for the spectroscopy part of the code
    def setup(self):
        self.load_folder()
        self.load_files()
        self.read_headers()
        print("Setup completed successfully!")
        
    def master_calibrations(self):
        self.master_bias()
        self.master_dark()
        self.master_flat()
        print("Master bias, dark and flat frames created succesfully!")
        
    def reductions(self):
        self.science_reduction()
        self.arcs_reduction()
        print('Science frames and arcs processed succesfully!')
        
    #perform the wavelength calibration manually
    
    def master_reduction(self):
        self.load_folder()
        self.load_files()
        self.read_headers()
        print("Setup completed successfully!")
        self.master_bias()
        self.master_dark()
        self.master_flat()
        print("Master bias, dark and flat frames created succesfully!")
        self.science_reduction()
        self.arcs_reduction()
        print('Science frames and arcs processed succesfully! Now proceed to wavelength calibration using your chosen method.')
    
    def extraction_calibration(self):
        self.spec_extraction_2spectra()
        self.apply_wavelength_calibration_2spectra()
        print('You have fully reduced the data! Good job.')
        
    def extraction_calibration_1spectrum(self):
        self.spec_extraction_1spectrum()
        self.apply_wavelength_calibration()
        print('You have fully reduced the data! Good job.')
    
    def lightcurve(self):
        #store summed flavours
        summed_signal_star1 = []
        summed_signal_star2 = []
        #retrieve time values from science_reduction
        time_values = []
        #convert from lrs format to mjd
        for date_obs in self.time_values:
            time_mjd = Time(date_obs, format='isot', scale='utc').mjd
            time_values.append(time_mjd)
        
        for i, (sky_subtracted_spectrum_1, sky_subtracted_spectrum_2) in enumerate(self.sky_subtracted_spectra):
            #convert spectra to arrays
            sky_subtracted_spectrum_1 = np.array(sky_subtracted_spectrum_1)
            sky_subtracted_spectrum_2 = np.array(sky_subtracted_spectrum_2)

            #sum the signals from stars to a single point
            sum_signal_1 = np.sum(sky_subtracted_spectrum_1)
            summed_signal_star1.append(sum_signal_1)

            sum_signal_2 = np.sum(sky_subtracted_spectrum_2)
            summed_signal_star2.append(sum_signal_2)
        
        #normalizing!
        t_avg_s1 = np.mean(summed_signal_star1)
        t_avg_s2 = np.mean(summed_signal_star2)
        normalized_signal_star1 = [flux / t_avg_s1 for flux in summed_signal_star1]
        normalized_signal_star2 = [flux / t_avg_s2 for flux in summed_signal_star2]
        
        #on the road to response correction
        normalized_flux_star1_arr = np.array(normalized_signal_star1)
        normalized_flux_star2_arr = np.array(normalized_signal_star2)
        
        #response correction for star 1
        response_corrected_light_curve_star1 = summed_signal_star1 / summed_signal_star2
        #response correction for star 2
        response_corrected_light_curve_star2 = summed_signal_star2 / summed_signal_star1
        
        #normalizing rc for star 1
        t_avg_rc1 = np.mean(response_corrected_light_curve_star1)
        t_avg_rc2 = np.mean(response_corrected_light_curve_star2)
        norm_rc1 = [flux / t_avg_rc1 for flux in response_corrected_light_curve_star1]
        norm_rc2 = [flux / t_avg_rc2 for flux in response_corrected_light_curve_star2] 
        
        
        #plotting the light curve for star 1
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, summed_signal_star1, color="hotpink", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Summed Signal (Intensity)")
        plt.title("Light Curve for Star 1")
        plt.xticks(rotation=45) 
        plt.legend()
        lightcurve_star1_path = os.path.join(self.path_plots, f"lightcurve_star1.png")
        plt.savefig(lightcurve_star1_path)
        plt.close()

        #plotting the light curve for star 2
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, summed_signal_star2, color="purple", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Summed Signal (Intensity)")
        plt.title("Light Curve for Star 2")
        plt.xticks(rotation=45) 
        plt.legend()
        lightcurve_star2_path = os.path.join(self.path_plots, f"lightcurve_star2.png")
        plt.savefig(lightcurve_star2_path)
        plt.close()
        
        #plotting the normalized light curve for star 1
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, normalized_signal_star1, color="hotpink", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Normalized Signal")
        plt.title("Normalized Light Curve for Star 1")
        plt.xticks(rotation=45)
        lightcurve_star1_path = os.path.join(self.path_plots, "normalized_lightcurve_star1.png")
        plt.savefig(lightcurve_star1_path)
        plt.close()

        #plotting the normalized light curve for star 2
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, normalized_signal_star2, color="purple", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Normalized Signal")
        plt.title("Normalized Light Curve for Star 2")
        plt.xticks(rotation=45)
        lightcurve_star2_path = os.path.join(self.path_plots, "normalized_lightcurve_star2.png")
        plt.savefig(lightcurve_star2_path)
        plt.close()
        
        #plot the response-corrected light curve 1
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, response_corrected_light_curve_star1, label="Response-Corrected Light Curve", color="blue", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Response-Corrected Flux")
        plt.title("Response-Corrected Light Curve for Star 1")
        plt.xticks(rotation=45)
        plt.legend()
        response_corrected_path = os.path.join(self.path_plots, "response_corrected_lightcurve_star1.png")
        plt.savefig(response_corrected_path)
        plt.close()
        
        #plot the response-corrected light curve 2
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, response_corrected_light_curve_star2, label="Response-Corrected Light Curve", color="blue", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Response-Corrected Flux")
        plt.title("Response-Corrected Light Curve for Star 2")
        plt.xticks(rotation=45)
        plt.legend()
        response_corrected_path = os.path.join(self.path_plots, "response_corrected_lightcurve_star2.png")
        plt.savefig(response_corrected_path)
        plt.close()
        
        #plotting the normalized light curve for star 1
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, norm_rc1, color="purple", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Normalized Signal")
        plt.title("Normalized Response-corrected Light Curve for Star 1")
        plt.xticks(rotation=45)
        lightcurve_star2_path = os.path.join(self.path_plots, "norm_rc_star1.png")
        plt.savefig(lightcurve_star2_path)
        plt.close()
        
        #plotting the normalized light curve for star 2
        plt.figure(figsize=(12, 6))
        plt.scatter(time_values, norm_rc2, color="purple", s=50)
        plt.xlabel("MJD")  
        plt.ylabel("Normalized Signal")
        plt.title("Normalized Response-corrected Light Curve for Star 2")
        plt.xticks(rotation=45)
        lightcurve_star2_path = os.path.join(self.path_plots, "norm_rc_star2.png")
        plt.savefig(lightcurve_star2_path)
        plt.close()
        
        

