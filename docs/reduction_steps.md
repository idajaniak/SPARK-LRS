SPARK-LRS was created to reduce raw science files from the Low Resolution Spectrograph on the 2.4 m Thai National Telescope. It was designed to meet SPEARNET’s need for a dual-slit data reduction pipeline and was also adapted for single-target long-slit observations.

This documentation describes the functionality of the pipeline and explains each user-accessible function in detail. If you are looking for a quick tutorial, please see section 4.

Scroll to the bottom of this file for a quick data reduction tutorial.

## Introduction

The pipeline is structured into distinct stages, each with a specific goal. First, it sets up the working environment and classifies files based on their headers. The second stage performs the core data reduction, where standard processes are applied. Finally, the data are calibrated and exported. At this stage, it is also possible to create and normalize a light curve of the target.

## 1. Getting started

SPARK-LRS is designed to run in an automated way. The user provides the path to the folder containing all raw science and calibration frames, and the code organizes and categorizes the files automatically. This setup allows the following reduction steps to be almost entirely hands-off.

To begin, initialize the pipeline by calling the class with two arguments:

```python
session_1 = SPARK_LRS("your_path_to_folder", slit_size)
```
"your_path_to_folder" should contain the path to your folder containing ALL raw data. This is where the function will set its working environment and create folders where it will output files. 

slit_size defines the region of the detector that will be used in the reduction.
If you want to keep the full image (no trimming), set slit_size to: [:,:]. 
If you want to crop the image to avoid large borders or reduce processing time, provide pixel ranges explicitly: [x_min:x_max, y_min:y_max], e.g., [:,70:250] (if you only want to apply a cut in the y direction).

Once you have initialized your session, the next step is to prepare the working environment. This is done using the load_folder() function.
```python
session_1.load_folder()
```
This does several things. 
1) The code changes the working directory to the folder you specified when creating the session. From this point on, all file creation and saving will happen inside this directory.
2) Creates two main folders (if they do not already exist):
Intermediate files - used for storing temporary products created during the reduction (e.g., partially processed frames).
Calibrated files - used for storing final reduced spectra and related products.

Inside Calibrated files, it creates a structured set of subfolders for plots and extracted data:
Plots
Sky subtracted collapsed spectra - plots of collapsed spectra after sky subtraction.
Not sky subtracted collapsed spectra - plots of collapsed spectra without sky subtraction.
Not sky subtracted calibrated spectra - wavelength-calibrated spectra without sky subtraction.
Sky subtracted calibrated spectra - wavelength-calibrated spectra with sky subtraction.
Extracted regions - images of the spectral extraction regions (used for checking where the trace was taken).


After preparing the folder structure, you need to load the actual science and calibration frames into the session using:
```python
session_1.load_files()
```
This function looks inside your data directory and loads in only files with .fit or .fits.

The most important function of the whole pipeline and its heart is read_headers(), which sorts files into distinct categories based on their header information and file names. If this step is not executed correctly, i.e., the flats are misrecognised as science frames, the later reduction will be a mess. Please verify with the output statement  printed upon the completion of the function to see if the files were recognised correctly.

```python
session_1.read_headers()
```
After the raw data have been loaded, the files are identified as bias, dark, flat, arc (lamp), science, or other (user provided master frames). The key words for headers and file names recognition are very much unique to TNT/LRS so if you are using this pipeline for another telescope, or the key words for TNT/LRS have been updated, please be careful and alter the keywords in the main code.

## 2. Data reduction

Once the environment is set up and files are grouped into calibration frames and science frames, we begin the process of data reduction. Before we approach the science frames, there are several calibration frames that need to be created first.

Starting off, to mitigate the inherent random noise of the detector we first need to measure it using bias frames - images taken with 0 s exposure time. 
To create a master bias, or retrieve one if provided by the user, type:
```python
session_1.master_bias()
```
into your notebook. This will median combine all the biases into a singular master frame and save it into the 'Intermediate files' folder in your working directory. 

It is important to note that all processes here are interconnected, so if you do not run the master_bias() command (or any other commands), you won't be able to run the next steps!

The next step is to prepare a master dark frame. Dark frames are exposures taken with the shutter closed, usually with exposure times matching those of science frames, and are used to measure thermal noise from the detector. A master dark is created by median combining multiple individual dark frames so that random noise averages out. In SPARK-LRS the master dark can be retrieved or created using:

```python
session_1.master_dark()
```
Shortly, this function checks the exposure times of your darks with those of the science frames, combining matching ones into a master dark (best solution) or scaling the closest available dark to match the science exposure time. It then outputs a master dark into the 'Intermediate files' directory.

Important: Try to always match your dark frames with exposure times of science frames. The code can scale exposure times of the darks to match those of science frames, but this can lead to overestimation of the dark current and in return will negatively affect the dark subtraction, weakening the extracted target signal.

The next master frame that we will create it a master flat, which is used to correct for pixel-to-pixel sensitivity variations in the detector. Similarly to the previous master files, the code will median combine all available flat frames (make sure they have the same exposure times) or retireve a user-provided master flat. The combined flat is then bias-subtracted and output to 'Intermediate files'.
To generate a master flat, run in your notebook:
```python
session_1.master_flat()
```

With all the master calibration frames generated, we can now apply them all to reduce our science frames. To begin the process type:
```python
session_1.science_reduction()
```
Using this function, each frame is trimmed to the slit, bias-subtracted, dark-subtracted, and corrected with the master flat. The reduced frames are saved individually in the 'Calibrated files' folder with filenames beginning with 'science_reduced_[your initial file name]'. Observation times from the headers are also stored for later use in photometric or temporal analysis (if the user chooses to do so). 

### 2.1 Wavelength calibration
Since wavelength calibration can be performed in three different ways I decided to make this step into a separate subsection to avoid bulky documentation. This step is also the only point,apart from initializing SPARK-LRS, where user input is required.
#### Collapsing arcs
Our science frames are now reduced, but they are not yet wavelength calibrated. This means that spectral features are still expressed in pixel space rather than in physical units of wavelength. To begin calibration, we first process the arc frames with
```python
session_1.arcs_reduction()
```
Each arc frame is trimmed to the slit region, bias-subtracted, and flat-fielded to remove instrumental signatures. The processed arcs are then median-combined into a single master arc frame, improving the signal-to-noise ratio and suppressing cosmic rays or any other temporal signatures within each exposure. The function also provides a quick-look plot of the summed spectrum and saves the master arc in the 'Intermediate files' folder.

#### Option 1: HgAr or Ne calibration lamps
At the time of writing this tutorial, two calibration lamps are available for the TNT/LRS: mercury-argon (HgAr) and neon (Ne). The wavelength calibration is set up specifically for the peaks produced by each of these lamps. In practice, the functions are configured to recognize 16 peaks for HgAr and 21 peaks for Ne. We’ll start by walking through these two modes, and you can simply choose the calibration function that matches the data you collected.

If you are not using TNT/LRS specific HgAr or Ne lamps, skip this section and go to Option 2 for a custom way of performing wavelength calibration.

The hgar_calibration() and neon_calibration() functions assign wavelengths to your reduced spectra using known emission lines from the TNT/LRS calibration lamps. Upon execution, a prompt will appear asking you to enter a peak detection threshold. To choose a good value, look at the plot of your collapsed master arcs and eyeball the intensity of the shortest peak: if too many peaks appear (noise), increase the threshold slightly, if too few appear (too high threshold or your exposure is simply super short), decrease it. 

Don’t worry about restarting - the function will automatically ask you to adjust the threshold until the correct number of peaks is found (16 for HgAr, 21 for Ne). 

Once the peaks are detected, they are paired with the known wavelengths, a polynomial fit is applied, and a quick plot of the calibrated lamp spectrum is shown. The resulting polynomial model is returned for use in calibrating your science frames later on. 
To execute these two functions type:
```python
session_1.hgar_calibration(order='linear', 'quadratic' or 'cubic')
```
or
```python
session_1.neon_calibration(order='linear', 'quadratic' or 'cubic')
```
including your preferred polynomial fit that you must input.

#### Option 2: Custom wavelength calibration
For users working with different lamps, or in the case that the lamps at the TNO were changed, we developed a very flexible, but also user intense mode where full user control over wavelength calibration is offered, but significantly more input is required. In fact, this was the first wavelength calibration mode developed, before we applied the same logic to a more automated approach for the HgAr and Ne lamps.

In your notebook type:
```python
session_1.wavelength_calibration(order='linear', 'quadratic' or 'cubic')
```
(Users can provide their own wavelength_calibration.csv file, with pixel position-wavelength pairs, and skip the following process.)

Upon execution, you will first be asked to set a peak detection threshold, which will determine which peaks in the collapsed master arc will be considered. Please look at the spectrum displayed in arcs_reduction() to make your final decision. 
Once the threshold is input, you will then be presented with a plot of the arc spectrum with the detected peaks numbered. Following this, and depending on the number of peaks detected, you will be given the same prompt asking you to associate a wavelength (in nano-metres) with each peak, giving full ocntgrol over your calibration. Please be mindful that if you make a mistake, you will have to restart and rerun the function again.

When all peaks have wavelength associated with them, a list of these pairs is shown to the user and the function proceeds to fit a polynomial fit to these values, the calibrated spectrum is plotted for verification, and the resulting wavelength model is saved as a .csv file in your working directory.

## 3. Data calibration and spectral retrieval
Now that we have reduced our science frames and derived a wavelength solution, the next step is to extract the spectra, subtract the sky background, and apply the wavelength calibration. Which functions you use depends on how many targets are in your slit.

- If you’re working with a single target, proceed to Section 3.1.
- If your observations include two targets, go to Section 3.2.

### 3.1 Single target observations
#### 3.1.1 Extracting a single spectrum
Into your notebook type:
```python
session_1.spec_extraction_1spectrum(centre=None, spectrum_width=None, spectrum_length=None, gap=None, sky_width=None)
```
This function handles everything for you internally, using two helper routines (that you won't see but I thought it would be helpful to include them):
1. detect_and_extract_1region - this internal function finds the spectrum within the 2D image. If you provide a centre (row position), it uses that; otherwise, it automatically detects the brightest row. You can also set the spectrum_width (how many pixels around the centre to include) and spectrum_length (which columns of the image to include). If no values are provided, defaults are used: full image length and a width of 30 pixels.
2. sky_subtraction_1spectrum - once the spectrum region is extracted, this internal function calculates the background sky signal from regions above and below the spectrum. It subtracts this sky to produce a clean, 1D sky-subtracted spectrum. You can adjust the gap (distance between the spectrum and sky regions) and sky_width (how many pixels to use for sky sampling), with defaults of 30 and 10 pixels, respectively.

The main function (spec_extraction_1spectrum) applies these internal routines in sequence for each frame, then additionally performs sigma clipping to remove outliers (like bad pixels or cosmic rays) and collapses the 2D spectrum into a 1D array. For your convenience, it also:
a) Saves the extracted 2D region as a FITS file.
b) Saves plots of the collapsed spectrum (before sky subtraction) and the sky-subtracted spectrum.

Inputs you need to consider (or don't, the default values will be then applied):
- centre - optional; row of the spectrum. If None, the code automatically finds it.
- spectrum_width - optional; number of rows to include around the centre. Defaults to 30 pixels.
- spectrum_length - optional; slice of columns to extract. Defaults to the full image width.
- gap - optional; separation between spectrum and sky regions. Default is 30 pixels.
- sky_width - optional; number of pixels used to calculate the sky background. Default is 10 pixels.

By calling this function, all the internal steps are executed automatically, so you get fully reduced, sky-subtracted 1D spectra ready for plotting, analysis, or further calibration.

#### 3.1.2 Applying wavelength calibration
We've extracted the spectrum, sky-subtracted it and cleaned it up- now it is time to apply wavelength calibration.

```python
session_1.apply_wavelength_calibration(spectrum_length)
```
The function first checks that a wavelength calibration model exists- this is the model derived from your HgAr, Ne, or custom calibration.
It applies this model to each extracted spectrum, converting pixel positions along the x-axis into actual wavelengths. Both the collapsed spectrum (non-sky-subtracted) and the sky-subtracted spectrum are processed.

The spectrum_length parameter defines which columns of the spectrum to include. This is the same slice you used during extraction.

Then for each frame the function saves two plots: one of the sky-subtracted spectrum and one of the non-sky-subtracted spectrum, both in wavelength space. These plots are saved automatically in your designated folders for review. Finally, it returns two lists: one with the wavelength-calibrated spectra, and one with the wavelength-calibrated sky-subtracted spectra, ready for analysis.
This function is fully automated once you provide the spectrum_length slice, so all wavelength conversion and plotting happen without additional user input.

### 3.2 Dual target observations
This mode was developed for the goals of the SPEARNET collaboration to monitor exoplanetary transits with the host and reference stars in the same slit.

#### 3.2.1 Extracting two spectra
For observations where your slit contains two targets (e.g., a science target and a comparison star), the pipeline provides a dedicated set of functions to extract and sky-subtract both spectra simultaneously. You only need to call the high-level function:
```python
session_1.spec_extraction_2spectra(centre=None, spectrum_width=None, spectrum_length=None, manual_sky_region=None)
```
There are two internal functions which are sourced here- you will not have to deal with them, but it is important to know about them so the workings of this function are clear.
1. detect_and_extract_2regions - it automatically identifies the two brightest peaks along the slit, assuming these correspond to your two targets (host and comparison stars). This function sources three parameters which you can specify when you call the main function. The centre is used as an initial guess to where one of the peaks might be- you do not need to provide it all, the pipeline will find it itself. The width of the extracted regions is controlled by spectrum_width (default 30 pixels), and the length along the dispersion direction is controlled by spectrum_length (default: full image length). It returns two extracted image regions along with their central row positions.
2. sky_subtraction_2spectra - it determines a sky background region between the two spectra by default, finding the middle ground between your two spectra. In case of a crowded field, pollution from a third star, or the two target spectra being too close, you can define your own sky region using manual_sky_region (by default it's set to None, if altering use this format: manual_sky_region=(start, end)). A sky median is then computed along this region and is later subtracted from both spectra. The length along the dispersion direction is controlled by spectrum_length (default: full image length).

These extracted and sky subtracted spectra are then sigma-clipped to remove any outliers (bad pixels, cosmic rays, hot pixels etc.) and the spectrum is collapsed along the slit direction to produce a 1D spectrum. The results are saved from both sky subtracted and not sky subtracted FITS files and 1D PNG plots. 

#### 3.2.2 Applying wavelength calibration
Once the two spectra have been extracted and sky-subtracted, you can apply the wavelength solution to convert pixel positions into physical units (nanometres). For dual-slit observations, the pipeline provides:
```python
 session_1.apply_wavelength_calibration_2spectra(spectrum_length)
```
The function first checks that a wavelength calibration model exists- this is the model derived from your HgAr, Ne, or custom calibration.
It applies this model to each extracted spectrum, converting pixel positions along the x-axis into actual wavelengths. Both the collapsed spectrum (non-sky-subtracted) and the sky-subtracted spectrum are processed.

The spectrum_length parameter defines which columns of the spectrum to include. This is the same slice you used during extraction.

The pipeline automatically applies the wavelength calibration to both spectra, subtracts the sky, and saves plots for inspection. You do not need to manually manipulate pixel positions- the function handles all mapping and slicing internally.

Fully calibrated and reduced data is available in your 'Calibrated files' folder created within your working directory.

#### 3.2.3 Light curve creation
After extracting, sky-subtracting, and calibrating the two spectra from a dual-slit observation, the user can optionally generate light curves. This allows you to track the flux of each star over time and perform relative and/or response-corrected analysis.
You can call the function simply with:
```python
session_1.lightcurve()
```
The function automatically sums each spectrum, performs normalization and response correction, and saves plots of the resulting light curves for both stars. This gives you a direct view of relative flux variations, which is particularly useful for transit or variability studies

---
Congratulations! You reduced your dataset.




## 4. Quick reduction cheat sheet
Here are all the steps combined into an easily copy-pastable format:

For a single-slit observation:
1. 
```python
session_1.load_folder()
session_1.load_files()
session_1.read_headers()
print("Setup completed successfully!")
session_1.master_bias()
session_1.master_dark()
session_1.master_flat()
print("Master bias, dark and flat frames created succesfully!")
session_1.science_reduction()
session_1.arcs_reduction()
```

2. call any of the three wavelength calibration functions
```python
session_1.hgar_calibration()
session_1.neon_calibration()
session_1.wavelength_calibration()
```
3. to extract spectra and apply the wavelength solution
```python
session_1.spec_extraction_1spectrum()
session_1.apply_wavelength_calibration()
```
All done!

For a dual-slit observation:
1. 
```python
session_1.load_folder()
session_1.load_files()
session_1.read_headers()
print("Setup completed successfully!")
session_1.master_bias()
session_1.master_dark()
session_1.master_flat()
print("Master bias, dark and flat frames created succesfully!")
session_1.science_reduction()
session_1.arcs_reduction()
```

2. call any of the three wavelength calibration functions
```python
session_1.hgar_calibration()
session_1.neon_calibration()
session_1.wavelength_calibration()
```
3. to extract spectra and apply the wavelength solution
```python
session_1.spec_extraction_2spectra()
session_1.apply_wavelength_calibration_2spectra()
```
Awesome!


---
Now, if you want to do everything mentioned above but do not want to call every single function, I've wrapped up steps 1 and 2 under the same name so that it's quicker!

This function sets up the environment, creates master files and reduces arcs and science frames:
```python
session_1.master_reduction()
```
After this you can simply execute any of the three wavelength calibration functions, followed by spectrum extraction and calibration, or you can create a light curve.


