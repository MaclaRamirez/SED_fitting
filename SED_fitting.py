#traslation of Lucas' routine to plot the SED using X-Shooter data
#usage: python -W ignore plot_SED.py -o 'objname' -r  $Rv
 
from astropy.io import fits
from pylab import *
import numpy as np
import itertools
import argparse
from astropy.convolution import convolve, Box1DKernel
from scipy import interpolate
import matplotlib as mpl
import extinction
mpl.rc('font',family='Serif')

def smoothSpec(bin_size,wave,flux):
    new_w = wave
    new_flux = convolve(flux, Box1DKernel(bin_size))
    return new_w, new_flux


def get_dered(Rv, npix):
 
  wave_start=300. #start wavelength value in nm
  wave_end=4000. #end wavelength value in nm
  wave=np.linspace(wave_start, wave_end, npix)
 
  #Values from Cardelli et al. 1989 for different Rv (Table 3):
  x_cat_C = np.array([2.78,2.27,1.82,1.43,1.11,.8,.63,.46,.29])
  a_cat_C = np.array([.9530,.9982,1.,.8686,.6800,.4008,.2693,.1615,.0800])
  b_cat_C = np.array([1.9090,1.0495,0.,-.3660,-.6239,-.3679,-.2473,-.1483,-.0734])
  AlAv_cat_C = a_cat_C+b_cat_C/Rv
  wave_cat_C = 1000/x_cat_C
 
  #fit a polynomial of degree 5 through the catalog values
  coeffs_C = np.polyfit(wave_cat_C,AlAv_cat_C,5)
 
  a=coeffs_C #a are the coeficients of the fit a1+a2*x+a3*x^2+...+a6*x^5
  #evaluate the polinomial for every wavelength in the array 300--4000 nm
  AlAv=np.poly1d(coeffs_C)(wave) #Al/Av for the whole wavelength array from 300 to 4000 nm
  AlAvarray=AlAv_cat_C #AlAvarray is the a(x)+b(x)/Rv for the filters in table 3 of Cardelli89
   
  # print 'wave', wave, len(wave)
  # print 'a', a, len(a)
  # print 'Al/Av', AlAv, len(AlAv)
  # print 'AlAvarray', AlAvarray, len(AlAvarray)
  return wave, a, AlAv, AlAvarray

def get_param(objname):
    # tempK, loggK are the ones to be displayed in the figures
    if objname =='M17-B111':
        temp=45000
        logg=4.5
        tempK=44852
        loggK=3.92
        SpT = ' O4.5 V'
        BC = -4.4 #from Allen's astrophysical quantities
    elif objname == 'M17-B163':
        temp=8250
        logg=3.5
        tempK=8200
        loggK=3.5
        SpT = 'kA5 III'
        BC = -1.1 #from Allen's astrophysical quantities
    elif objname =='M17-B164':
        temp=37000
        logg=4.0
        tempK=36872
        loggK=3.92
        SpT = ' O6 Vz'
        BC = -3.9 #from Allen's astrophysical quantities
    elif objname == 'M17-B215': #B215
        temp=25000
        logg=4.0
        tempK=25400
        loggK=3.90
        SpT = ' B1 V'
        BC = -3.2 #from Allen's astrophysical quantities
    elif objname == 'M17-B243' or objname == 'M17-B243_noexcess':
        temp=13000
        logg=4.0
        tempK=13000
        loggK=4.04
        SpT = ' B8 V'
        BC = -0.8 #from Allen's astrophysical quantities
    elif objname =='M17-B253':
        temp=15000
        logg=3.5
        tempK=15000
        loggK=3.49
        SpT = ' B5 III'
        BC = -1.9 #from Allen's astrophysical quantities
    elif objname == 'M17-B268'  or objname == 'M17-B268_noexcess':
        temp=12000
        logg=4.0
        tempK=11900
        loggK=4.04
        SpT = 'A0 V'
        BC = -0.38 #from Allen's astrophysical quantities
    elif objname == 'M17-B275' or objname == 'M17-B275_noexcess':
        temp=13000
        logg=3.5
        tempK=13000
        loggK=3.5
        SpT = ' B7 III'
        BC = -1.0 #from Allen's astrophysical quantities
    elif objname == 'M17-B289':
         temp=34000
         logg=4.0
         tempK=33879
         loggK=3.92
         SpT = ' O9.7 V'
         BC = -3.25 #from Allen's astrophysical quantities
    elif objname == 'M17-B311':
        temp=36000
        logg=4.0
        tempK=35874
        loggK=3.92
        SpT = ' O8.5 Vz'
        BC = -3.35 #from Allen's astrophysical quantities
    elif objname == 'M17-B331':
        temp=13000
        logg=4.0
        tempK=13000
        loggK=4.04
        SpT = ' B8 V'
        BC = -0.8 #from Allen's astrophysical quantities
        # temp=22000 # From Hoffmeister+ 2008, ApJ, 686, 310
        # logg=3.5
        # tempK=22000
        # loggK=3.50
        # SpT = ' B2 III'
        # BC = -2.4 #from Allen's astrophysical quantities
    elif objname == 'M17-B337':
        temp=13000
        logg=4.0
        tempK=13000
        loggK=4.04
        SpT = ' B8 V'
        BC = -0.8 #from Allen's astrophysical quantities
    elif objname == 'spec462':
        temp=34000
        logg=4.0
        tempK=34200
        loggK=4.22
        SpT='O9 V'
        BC=-3.33
    elif objname == 'spec413':
        temp=35000
        logg=4.0
        tempK=34900
        loggK=4.09
        SpT='O9.5 V'
        BC=-3.2
    elif objname == 'spec179':
        temp=20000
        logg=3.5
        tempK=19800
        loggK=3.5
        SpT='B3 IV'
        BC=-1.3
    elif objname == 'spec408':
        temp=8750
        logg=3.5
        tempK=8720
        loggK=3.5
        SpT='A3 IV'
        BC=-0.17
    else:
        temp=15000
        logg=4.0
        tempK=00000
        loggK=0000
        SpT = '??'
        BC = -1.1 #from Allen's astrophysical quantities
    return temp,logg,tempK,loggK,SpT,BC 

def FluxZeroPoints_waves(nbands): 
    f0=[]
    std_wave=[]
    std_flux = []
    
    # Write the appropriate (wave,flux) values in (nm, erg cm^-2 A^-1)
    for i in range(nbands):
      if std_band[i] == 'U': 
        f0.append(4.266e-9)
        std_wave.append(365.)
      elif std_band[i] == 'B':
        f0.append(6.822e-9)
        std_wave.append(433.)
      elif std_band[i] == 'V':
        f0.append(3.802e-9)
        std_wave.append(550.)
      elif std_band[i] == 'R': 
        f0.append(1.738e-9)
        std_wave.append(700.)
      elif std_band[i] == 'I':
        f0.append(8.318e-10)
        std_wave.append(900.)
      elif std_band[i] == 'J':
        f0.append(3.129e-10)
        std_wave.append(1235.)
      elif std_band[i] == 'H':
        f0.append(1.133e-10)
        std_wave.append(1662.)
      elif std_band[i] == 'K':
        f0.append(4.283e-11)
        std_wave.append(2159.)
      elif std_band[i] == '36um':
        f0.append(6.601e-12)
        std_wave.append(3550.)
      elif std_band[i] == '45um':
        f0.append(2.545968e-12)
        std_wave.append(4500.) 
      elif std_band[i] == '58um':
        f0.append(1.064e-12)
        std_wave.append(5731.)
      elif std_band[i] == '80um':
        f0.append(3.004e-13)
        std_wave.append(8000.)
      elif std_band[i] == 'W1':
          f0.append(8.1787e-12)
          std_wave.append(34000)
      elif std_band[i] == 'W2':
          f0.append(2.4150e-12)
          std_wave.append(46000)
      elif std_band[i] == 'W3': 
          f0.append(6.5151e-14)
          std_wave.append(120000)      
      elif std_band[i] == 'W4': 
          f0.append(5.0901e-15)   
          std_wave.append(220000)
    for i in range(len(std_mag)):
        std_flux.append(f0[i]*(10**(-0.4*float(std_mag[i]))))
   
    #Append N and Q bands Nielbock et al. 2001 and SPITZER
    # conversion to erg s-1 cm-2 A-1 in http://localhost:8888/notebooks/mJy_to_erg_s-1_cm-2_Hz-1.ipynb
 
    if objname == 'M17-B275':
      std_wave.extend([10600,20000])
      std_flux.extend([5.06946920612e-15,5.9208999e-15])
    elif objname == 'M17-B289':
      std_flux[8] = 1.900E-14 #3.6 micron IRAC (Band 1) magnitude from visier
      std_flux[9] = 6.858E-15 #5.8 micron  IRAC (Band 1) magnitude from visier  
      std_wave.extend([10600,20000])
      std_flux.extend([6.67035421858e-14,6.04081686e-14])
    elif objname == 'M17-B331':
      std_wave.extend([10600,20000])
      std_flux.extend([2.29460185119e-14,2.274674835e-14])
    elif objname == 'M17-B337':
      std_flux[5] = 3.317E-15 #3.6 micron IRAC (Band 1) magnitude from visier
      std_flux[6] = 5.181E-15 #5.8 micron IRAC (Band 1) magnitude from visier 
      std_wave.extend([10600,20000])
      std_flux.extend([1.73429209683e-15,2.43581325e-15])
    elif objname == 'M17-B215':
      std_flux[4] = 8.293E-15 #3.6 micron IRAC (Band 1) magnitude from visier
      std_flux[5] = 2.960E-15 #5.8 micron IRAC (Band 1) magnitude from visier 
      std_wave.extend([10600,20000])
      std_flux.extend([8.72482331791e-14,6.88023558e-14])
    elif objname == 'M17-B311':
      std_wave.extend([10600,20000])
      std_flux.extend([1.73429209683e-14,1.61138415e-14])
   
 
    std_flux = np.array(std_flux)
    std_wave = np.array(std_wave) 

    return std_wave,std_flux

def get_Av(AlAv,Avgrid):
    ####################################################################
    # obtain best-fit Av by finding the minimum difference between the 
    # slope of the Kurucz model and the dereddened data
    ####################################################################
    
    # find best-fit Av
    slopegrid=[]
    scalegrid=[]

    for i in range(ngrid):
      flux_der = (f*10**(.4*AlAv*Avgrid[i]))
      idx = np.isfinite(np.log10(w[ph])) & np.isfinite(np.log10(w[ph]*10*flux_der[ph]))
      coeff=np.polyfit(np.log10(w[ph])[idx],np.log10(w[ph]*10*flux_der[ph])[idx],1)

      slopegrid.append(coeff[0])
      scalegrid.append(coeff[1])

    diff = abs(slopegrid-slope_k).tolist()
    ibest = diff.index(min(diff))
    bestslopecoeff=slopegrid[ibest]
    bestscalecoeff=scalegrid[ibest]
    Avbest=Avgrid[ibest]
    
    return Avbest,bestslopecoeff,bestscalecoeff
    
def get_radius(bestslopecoeff,bestscalecoeff):
    ####################################################################
    #
    # find corresponding radius by measuring the distance between tne 
    # kurucz model and the data in a region around 500 nm
    #
    ####################################################################
    v_domain =  [500.,510.,520.,530.,540.,550.,560.]

    vflux=np.median(np.poly1d([bestslopecoeff,bestscalecoeff])(np.log10(v_domain)))
    vflux_k=np.median(np.poly1d(coeff_k)(np.log10(v_domain)))

    scalefactor = 10**(vflux-vflux_k)

    # scale flux to distance.
    #To convert to observed flux at Earth, multiply by a factor of (R/D)^2
    #where R is the stellar radius, and D is the distance to Earth. 
    #http://www.stsci.edu/hst/observatory/crds/k93models.html

    rbest=sqrt(scalefactor)*dist/6.955e10 # in R sun (Rsun = 6.955e10cm)
    
    return rbest,scalefactor

def get_lum(Av,mag,band):
    if band == 'V':
        coef = 1.
    elif band == 'R':
        coef = 0.751 #from Cardelli 1998
    elif band == 'K':
        coef = 0.056 #from Chini et al 1998, A&A, 329, 161

    Av_t = coef*Avbest
    Mv = mag + 5 - 5*np.log10(d)
    Mo_t = Mv - Av_t
    Mbol_t = Mo_t + BC
    L_t = 10**(0.4*(88.72 - Mbol_t))
    lum = np.log10(L_t/Lsun)
    M_0 = Mo_t

    return lum,M_0

####################################################################
#
#  get name of the object from terminal
#
####################################################################

parser = argparse.ArgumentParser()
 
parser.add_argument('-o', type=str)
parser.add_argument('-r', type=float)
 
objname = parser.parse_args().o
 
if not objname:
        objname = raw_input('Enter name of the object: ')
try:
    print objname
except IOError, msg:
    parser.error(str(msg))


####################################################################
#
#  DEFINE CONSTANTS AND ARRAYS
#
####################################################################
write_chisqFile = False
save_SEDfig = True

if objname == 'spec462':    
    d=700. #RCW36 462
elif objname == 'spec413':    
    d=810. #RCW36 463
elif objname == 'spec179':    
    d=620. #RCW36 179
elif objname == 'spec408':    
    d=700. #RCW36 408 not sure!!!
elif 'M17' in objname:
    d=1980. #distance to M17 in pc
else:
    print '!!! Distance set to 2 kpc !!!'
    d=2000.
    
dist=d*3.086e18 #distance to M17 in cm
Lsun = 3.84e33 #erg/s

Rv_array = np.arange(2.1,5.7,0.1)
Rv_array = [3.7]

Avmin=3
Avmax=20
ngrid=1000
Avgrid=np.linspace(Avmin, Avmax, ngrid)

Av_array = []
Av_array_ext = []
slopecoeff_array=[]
slopecoeff_array_ext=[]
scalecoeff_array=[]
scalecoeff_array_ext=[]
chisq=[]
chisq_ext=[]
scalefactor_array=[]
scalefactor_array_ext=[]
rstar_array=[]
rstar_array_ext=[]
Lum_array=[]
Lum_array_ext=[]
chisq_red = []
chisq_red_ext = []
Mo_array = []
Mo_array_ext = []



pathKurucz='/Users/macla/Dropbox/M17/SEDs/'
path='/Users/macla/Dropbox/M17/SEDs/'
# path='/Users/macla/Dropbox/M17/DIBs/RCW36_xshseds/'

temp,logg,tempK,loggK,SpT,BC = get_param(objname)
####################################################################
#
#  Read Kurucz models
#
####################################################################
 
 
kurucz=fits.open(pathKurucz+'Castelli-Kurucz/ckp00_'+str(temp)+'.fits')
 
wave_k = kurucz[1].data['WAVELENGTH']/10.
npix_k=len(wave_k)
 
if logg == 4.: #Fluxes tabulated in units of erg/s/cm^2/A
  flux_k = kurucz[1].data['g40'] #main sequence 
elif logg == 3.5:
  flux_k = kurucz[1].data['g35'] #giant
elif logg == 3.0:
  flux_k = kurucz[1].data['g30'] #
elif logg == 2.5:
  flux_k = kurucz[1].data['g25'] #
elif logg == 2.0:
  flux_k = kurucz[1].data['g20'] #
elif logg == 4.5:
  flux_k = kurucz[1].data['g45'] #
elif logg == 5:
  flux_k = kurucz[1].data['g50'] #giant


####################################################################
#
#  READ XSHOOTER DATA
#
####################################################################
w,f = np.loadtxt(path+'/chisq_fit/'+objname.split('M17-')[1]+'_FLUX.dat').T
# w,f = np.loadtxt(path+objname+'_FLUX.dat').T

####################################################################
#
#  READ MAGNITUDES FROM FILES (OBJNAME+.DAT)
#
# Flux zero points from Spitzer Telescope Handbook
# http://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzermission/missionoverview/spitzertelescopehandbook/19/
#http://casa.colorado.edu/~ginsbura/filtersets.htm
 
# UBVRI magnitudes from Neckel, T. & Chini, R. 1980
# JHK magnitudes from 2MASS Skrutskie et al. 2006
# 3.6 and 5.8 um from Spitzer GLIMSPE Benjamin et al. 2003
# N and Q mag from Nielbock et al. 2001
#
####################################################################
 
 
std_band, std_mag, std_err = np.loadtxt(path+objname+'.dat', dtype=str).T
nbands=len(std_band)

std_wave,std_flux = FluxZeroPoints_waves(nbands)

####################################################################
####################################################################
##############         Compute Dereddened flux        ##############
####################################################################
####################################################################
 
 
# define the photospheric domain
if objname == 'M17-B163':
    ph_k = np.logical_and(wave_k > 900, wave_k < 1000)
    ph = np.logical_and(w > 900, w < 1000)
elif objname == 'M17-B337' or objname == 'spec408':
    ph_k = np.logical_and(wave_k > 600, wave_k < 900)
    ph = np.logical_and(w > 600, w < 900)
elif 'spec' in objname:
    ph_k = np.logical_and(wave_k > 400, wave_k < 2000)
    ph = np.logical_and(w > 400, w < 2000)
else:
    ph_k = np.logical_and(wave_k > 400, wave_k < 820)
    ph = np.logical_and(w > 400, w < 820)
 
#find Kurucz model's slope and scale
coeff_k=np.polyfit(np.log10(wave_k[ph_k]),np.log10(wave_k[ph_k]*10*flux_k[ph_k]),1)
slope_k=coeff_k[0]
scale_k=coeff_k[1] 

 
#limit photospheric magnitudes of the magniitudes covered by Cardelli
w_phot = std_wave[std_wave < 4000]
f_phot = std_flux[std_wave < 4000]

#loop over Rv 
for Rv in Rv_array:
#Rv = 3.1

    print 'Rv = ',Rv

    wave, a, AlAv, AlAvarray = get_dered(Rv, len(f))
    
    AlAv=np.poly1d(a)(w)
    AlAv_phot=np.poly1d(a)(w_phot)
    # print 'AlAv_phot', AlAv_phot, len(AlAv_phot)
    AlAv_phot_ext = extinction.fitzpatrick99(w_phot*10, 1.0, Rv)
    AlAv_ext = extinction.fitzpatrick99(w*10, 1.0, Rv)
    # print 'Extinction', AlAv_phot_ext, len(AlAv_phot_ext)
    
    Avbest, bestslopecoeff,bestscalecoeff = get_Av(AlAv,Avgrid)
    Avbest_ext, bestslopecoeff_ext,bestscalecoeff_ext = get_Av(AlAv_ext,Avgrid)

    flux_der = (f*10**(.4*AlAv*Avbest))
    flux_der_ext = (f*10**(.4*AlAv_ext*Avbest_ext))

    f_phot_der=f_phot*10**(.4*AlAv_phot*Avbest)
    f_phot_der_ext=f_phot*10**(.4*AlAv_phot_ext*Avbest_ext)
    
    rbest, scalefactor = get_radius(bestslopecoeff,bestscalecoeff)
    rbest_ext, scalefactor_ext = get_radius(bestslopecoeff_ext,bestscalecoeff_ext)

    if objname == 'M17-B337':
        mag = float(std_mag[std_band == 'K'][0])
        band ='K'
    elif objname == 'M17-B163':
        mag = float(std_mag[std_band == 'R'][0])
        band ='R'
    else:
        mag = float(std_mag[std_band == 'V'][0])
        band ='V'


    lumbest, M_0best = get_lum(Avbest,mag,band)
    lumbest_ext, M_0best_ext = get_lum(Avbest_ext,mag,band)
	
## To fit all bossible photometry points:

    if objname == 'M17-B243' or objname == 'M17-B268' or objname == 'M17-B275' or objname == 'M17-B163':
        w_fit = w_phot[w_phot < 1300]
        f_fit = f_phot_der[w_phot < 1300]
        f_fit_ext = f_phot_der_ext[w_phot < 1300]
    elif objname == 'M17-B289' or objname == 'M17-B215':
        w_fit = w_phot[w_phot < 2000]
        f_fit = f_phot_der[w_phot < 2000]
        f_fit_ext = f_phot_der_ext[w_phot < 2000]
    else:
        w_fit = w_phot[w_phot < 4000]
        f_fit = f_phot_der[w_phot < 4000]
        f_fit_ext = f_phot_der_ext[w_phot < 4000]

## If you want to fit only B and V bands:
    # if objname == 'M17-B337' or objname == 'M17-B163':
    #     w_fit = w_phot[w_phot < 1300]
    #     f_fit = f_phot_der[w_phot < 1300]
    # else:
    #     w_fit = w_phot[np.logical_or(std_band == 'B', std_band == 'V')]
    #     f_fit = f_phot_der[np.logical_or(std_band == 'B', std_band == 'V')]

## To fit all bossible photometry points except V band:
    # if objname == 'M17-B243' or objname == 'M17-B268' or objname == 'M17-B275' or objname == 'M17-B163':
    #     w_fit = w_phot[np.logical_and(w_phot < 1300,w_phot!=550)]
    #     f_fit = f_phot_der[np.logical_and(w_phot < 1300,w_phot!=550)]
    # elif objname == 'M17-B289' or objname == 'M17-B215':
    #     w_fit = w_phot[np.logical_and(w_phot < 2000,w_phot!=550)]
    #     f_fit = f_phot_der[np.logical_and(w_phot < 2000,w_phot!=550)]
    # else:
    #     w_fit = w_phot[np.logical_and(w_phot < 4000,w_phot!=550)]
    #     f_fit = f_phot_der[np.logical_and(w_phot < 4000,w_phot!=550)]
    
    kurucz_w = np.array([min(wave_k, key=lambda w_fit:abs(w_fit-i)) for i in w_fit])
    ind = wave_k.searchsorted(kurucz_w)
    fnew_k = flux_k[ind]

    x_fit = np.log10(w_fit)
    y_fit = np.log10(w_fit*10*f_fit)
    y_fit_ext = np.log10(w_fit*10*f_fit_ext)

    x_kf = np.log10(kurucz_w)
    y_kf = np.log10(kurucz_w*10*scalefactor*fnew_k)
    y_kf_ext = np.log10(kurucz_w*10*scalefactor_ext*fnew_k)


    chisq_red.append(np.nansum((y_fit - y_kf)**2/(y_fit*0.1)**2)/(len(y_fit)-1))
    chisq_red_ext.append(np.nansum((y_fit_ext - y_kf_ext)**2/(y_fit_ext*0.1)**2)/(len(y_fit_ext)-1))
    chisq.append(np.nansum((y_fit - y_kf)**2/(y_fit*0.1)**2))
    chisq_ext.append(np.nansum((y_fit_ext - y_kf_ext)**2/(y_fit_ext*0.1)**2))
    Av_array.append(Avbest)
    Av_array_ext.append(Avbest_ext)
    slopecoeff_array.append(bestslopecoeff)
    slopecoeff_array_ext.append(bestslopecoeff_ext)
    scalecoeff_array.append(bestscalecoeff)
    scalecoeff_array_ext.append(bestscalecoeff_ext)
    scalefactor_array.append(scalefactor)
    scalefactor_array_ext.append(scalefactor_ext)
    rstar_array.append(rbest)
    rstar_array_ext.append(rbest_ext)
    Lum_array.append(lumbest)
    Lum_array_ext.append(lumbest_ext)
    Mo_array.append([M_0best])
    Mo_array_ext.append([M_0best_ext])
    
    # x = np.log10(w)
    # y = np.log10(w*10*f)
    # y_der = np.log10(w*10*flux_der)
    # # Smooth spectra
    # x_sm, y_sm = smoothSpec(20, x, y)
    # x_sm, y_der_sm = smoothSpec(20, x, y_der)
    #
    #
    # x_phot = np.log10(std_wave)
    # y_phot = np.log10(std_wave*10*std_flux)
    # x_phot_der = np.log10(w_phot)
    # y_phot_der = np.log10(w_phot*10*f_phot_der)
    #
    # x_k = np.log10(wave_k)
    # y_k = np.log10(wave_k*10*scalefactor*flux_k)
    #
    # plt.plot(x[x>np.log10(700)],y[x>np.log10(700)])
    # plt.plot(x[x>np.log10(700)],y_der[x>np.log10(700)])
    # plt.plot(x_k,y_k,'--')
    # plt.plot(x_kf,y_kf,'dr')
    # plt.plot(x_phot,y_phot,'ok')
    # plt.plot(x_phot_der,y_phot_der,'ob')
    # plt.plot(x_fit,y_fit,'or')
    # plt.show()
    

index_best = chisq.index(min(chisq))
bestslopecoeff = slopecoeff_array[index_best]
bestscalecoeff = scalecoeff_array[index_best]
Avbest = Av_array[index_best]
Rv = Rv_array[index_best]
scalefactor = scalefactor_array[index_best]
rstar = rstar_array[index_best]
lum = Lum_array[index_best]
Mo = Mo_array[index_best]

print 'Original code', chisq_red[index_best], Rv, Avbest

index_best_ext = chisq_ext.index(min(chisq_ext))
bestslopecoeff_ext = slopecoeff_array_ext[index_best_ext]
bestscalecoeff_ext = scalecoeff_array_ext[index_best_ext]
Avbest_ext = Av_array_ext[index_best_ext]
Rv_ext = Rv_array[index_best_ext]
scalefactor_ext = scalefactor_array_ext[index_best_ext]
rstar_ext = rstar_array_ext[index_best_ext]
lum_ext = Lum_array_ext[index_best_ext]
Mo_ext = Mo_array_ext[index_best_ext]

print 'With Extinction:', chisq_red_ext[index_best_ext], Rv_ext, Avbest_ext

wave, a, AlAv, AlAvarray = get_dered(Rv, len(f))

AlAv=np.poly1d(a)(w)
AlAv_phot=np.poly1d(a)(w_phot)

AlAv_ext = extinction.fitzpatrick99(w*10, 1.0, Rv)
AlAv_phot_ext = extinction.fitzpatrick99(w_phot*10, 1.0, Rv)

flux_der = (f*10**(.4*AlAv*Avbest))
flux_der_ext = (f*10**(.4*AlAv_ext*Avbest_ext))
f_phot_der=f_phot*10**(.4*AlAv_phot*Avbest)
f_phot_der_ext=f_phot*10**(.4*AlAv_phot_ext*Avbest_ext)

x = np.log10(w)
y = np.log10(w*10*f)
y_der = np.log10(w*10*flux_der)
y_der_ext = np.log10(w*10*flux_der_ext)
# Smooth spectra
x_sm, y_sm = smoothSpec(20, x, y) 
x_sm, y_der_sm = smoothSpec(20, x, y_der) 
x_sm, y_der_sm_ext = smoothSpec(20, x, y_der_ext) 

x_phot = np.log10(std_wave)
y_phot = np.log10(std_wave*10*std_flux)
x_phot_der = np.log10(w_phot)
y_phot_der = np.log10(w_phot*10*f_phot_der)
y_phot_der_ext = np.log10(w_phot*10*f_phot_der_ext)

x_k = np.log10(wave_k)
y_k = np.log10(wave_k*10*scalefactor*flux_k)

####################################################################
#
#  Write files
#
####################################################################

if write_chisqFile == True:
    out = file('/Users/macla/Dropbox/M17/SEDs/chisq_fit/chi_sq_'+objname+'.dat', 'w')
    out.write('# Best parameters for ' + objname +' (Number of points fitted: '+str(len(x_fit))+')'+ '\n')
    out.write('# Rv = '+str(Rv)+'\n')
    out.write('# Av = '+str(Avbest)+'\n')
    out.write('# Rstar = '+str(rstar)+'\n')
    out.write('# Lum = '+str(lum)+'\n')
    out.write('# M_0 = '+str(Mo)+'\n')
    out.write('# Rv \t \t  chi^2 \t \t chi_red^2 \t\t Av \t \t  Rstar \t \t Lum \t\t M_0 \n')
    for i in range(len(Rv_array)):
        out.write(str(Rv_array[i])+'\t'+str(chisq[i])+\
        '\t'+str(chisq_red[i])+'\t'+str(Av_array[i])+\
        '\t'+str(rstar_array[i])+'\t'+str(Lum_array[i])+\
        '\t' +str(Mo_array[i][0]) + '\n')

    out.close()

    out_ext = file('/Users/macla/Dropbox/M17/SEDs/chisq_fit/chi_sq_'+objname+'_extF99.dat', 'w')
    out_ext.write('# Best parameters for ' + objname +' (Number of points fitted: '+str(len(x_fit))+')'+ '\n')
    out_ext.write('# Rv = '+str(Rv_ext)+'\n')
    out_ext.write('# Av = '+str(Avbest_ext)+'\n')
    out_ext.write('# Rstar = '+str(rstar_ext)+'\n')
    out_ext.write('# Lum = '+str(lum_ext)+'\n')
    out_ext.write('# M_0 = '+str(Mo_ext)+'\n')
    out_ext.write('# Rv \t \t  chi^2 \t \t chi_red^2 \t\t Av \t \t  Rstar \t \t Lum \t\t M_0 \n')
    for i in range(len(Rv_array)):
        out_ext.write(str(Rv_array[i])+'\t'+str(chisq_ext[i])+\
        '\t'+str(chisq_red_ext[i])+'\t'+str(Av_array_ext[i])+\
        '\t'+str(rstar_array_ext[i])+'\t'+str(Lum_array_ext[i])+\
        '\t' +str(Mo_array_ext[i][0]) + '\n')
    out_ext.close()

####################################################################
#
#  PLOT
#
####################################################################

print '\n ------------- Plotting spectrum ------------- \n'

# Plot the spectrum
fig = figure(figsize=(5,4))
ax = fig.add_subplot(111)
if objname == 'M17-B163':
    ax.plot(x_sm[x_sm>np.log10(700)],y_sm[x_sm>np.log10(700)], '0.35')
elif objname == 'M17-B337':
    ax.plot(x_sm[x_sm>np.log10(600)],y_sm[x_sm>np.log10(600)], '0.35')
elif objname == 'M17-B331':
    ax.plot(x_sm[x_sm>np.log10(530)],y_sm[x_sm>np.log10(530)], '0.35')
else:
    ax.plot(x_sm,y_sm, '0.35')
# the dereddened spectrum
if objname == 'M17-B163':
    ax.plot(x_sm[x_sm>np.log10(700)],y_der_sm[x_sm>np.log10(700)], 'royalblue')
    # ax.plot(x_sm[x_sm>np.log10(700)],y_der_sm_ext[x_sm>np.log10(700)], 'royalblue',alpha=0.5)
elif objname == 'M17-B337':
    ax.plot(x_sm[x_sm>np.log10(600)],y_der_sm[x_sm>np.log10(600)], 'royalblue')
    # ax.plot(x_sm[x_sm>np.log10(600)],y_der_sm_ext[x_sm>np.log10(600)], 'royalblue',alpha=0.5)
elif objname == 'M17-B331':
    ax.plot(x_sm[x_sm>np.log10(530)],y_der_sm[x_sm>np.log10(530)], 'royalblue')
    # ax.plot(x_sm[x_sm>np.log10(530)],y_der_sm_ext[x_sm>np.log10(530)], 'royalblue',alpha=0.5)
else:
    ax.plot(x_sm,y_der_sm, 'royalblue')
    # ax.plot(x_sm,y_der_sm_ext, 'royalblue',alpha=0.5)
# the kurucz model
ax.plot(x_k,y_k, 'k--', alpha=0.5, linewidth=1)
# the photometry
ax.plot(x_phot,y_phot, 's', color='0.35', markersize=5, linewidth=3)# fillstyle='none',
#The dereddend photometry
if objname == 'M17-B331':
    ax.plot(x_phot_der[x_phot_der<3.5],y_phot_der[x_phot_der<3.5], 'd',color='royalblue', markersize=5, linewidth=2)#, fillstyle='none'
    # ax.plot(x_phot_der[x_phot_der<3.5],y_phot_der_ext[x_phot_der<3.5], 'd',color='royalblue', markersize=5, linewidth=2,alpha=0.5)
else:
    ax.plot(x_phot_der,y_phot_der, 'd',color='royalblue', markersize=5, linewidth=2)#, fillstyle='none'
    # ax.plot(x_phot_der,y_phot_der_ext, 'd',color='royalblue', markersize=5, linewidth=2,alpha=0.5)#, fillstyle='none'

print 'Kband dereddened flux: ', y_phot_der[std_band.tolist().index('K')]

plt.xticks(np.log10([500,1000,2000, 5000,10000, 20000]), ['0.5','1.0','2.0', '5.0', '10.0', '20.0'], fontsize = 12)
ax.set_xlim(np.log10(250),np.log10(23000))
yrange=[-15,-6]
ax.set_ylim(yrange)
ax.set_xlabel(r'$\lambda$ $\mathrm{(\mu m)}$', fontsize=14)
ax.set_ylabel(r'$\log\,\lambda F_{\lambda}$ ($\mathrm{erg}$ $\mathrm{s^{-1} cm^{-2}}$)', fontsize=14)
plt.yticks(fontsize = 12)


#The box with radius and Av
textstr = '$\mathrm{%3s}$ \n $T_{\mathrm{eff}}=%6d$ $\mathrm{K}$ \n $\log\,g=%.2f$ \n  \n\n\n \n $\mathrm{R_V=%.1f}$ \n$\mathrm{A_V=%.2f}$ \n $\mathrm{R_{\star}=%.2f R_{\odot}}$ \n $\mathrm{log\,L/L_{\odot}=%.2f}$'%(SpT, tempK, loggK, Rv, Avbest, rstar, lum)

ax.text(0.5, 0.5,textstr, ha='left', va='center', fontsize=15, transform=ax.transAxes)
plt.gcf().subplots_adjust(left=0.2, bottom=0.1)

if save_SEDfig == True:
    #outname = path+'00_newSEDs/'+objname+'_T_'+str(tempK)+'_Rv_' + str(Rv) + '_logg_' + str(loggK) + '_SED.pdf'
    #outname = '/Users/macla/Dropbox/M17/SEDs/chisq_fit/noV/'+objname+'_T_'+str(temp)+'_Rv_' + str(Rv) + '_logg_' + str(logg) + '_SED.pdf'
    # outname = '/Users/macla/Dropbox/M17/papers/M17_PMS/v2/figures/'+objname+'_T_'+str(temp)+'_Rv_' + str(Rv) + '_logg_' + str(logg) + '_SED.pdf'
    outname = '/Users/macla/Dropbox/PhDThesis_2/gfx/Paper1/'+objname+'_T_'+str(temp)+'_Rv_' + str(Rv) + '_logg_' + str(logg) + '_SED.pdf'
    print 'spectrum plotted in: ', outname+'\n'
    fig.savefig(outname, bbox_inches='tight')
elif save_SEDfig == False:
    plt.show()

