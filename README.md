# Program to compute the isotropic equivalent energy of GRBs

Latest revision: 1 March 2021

## Prerequisites

```
astropy, joblib, psutil
```

#### How to call it?

```
usage: compute_Eiso.py [-h] --file FILE [--output OUTPUT] [--H0 H0]
                       [--Omega_M OMEGA_M] [--nint NINT] [--nmc NMC]

Programme to do compute the isotropic energy of GRBs

optional arguments:
  -h, --help         show this help message and exit
  --file FILE        Input file
  --output OUTPUT    Prefix of output file (Default: out_$FILE)
  --H0 H0            Hubble constant (Default: 67.3 km/s/Mpc)
  --Omega_M OMEGA_M  Hubble constant (Default: 0.315)
  --nint NINT        Number of steps in numerical integration (Default: 100)
  --nmc NMC          Number of steps in MC simulation (Default: 5000)
```

#### Example

```
python compute_Eiso.py --file energy_input
```

### How to flag objects in the input file?

put a '#' in front of the GRB ID

### What are the output values?

The program returns the isotropic energy between 1 kev and 10000 keV. In addition it also computes the k-correction as defined in [https://ui.adsabs.harvard.edu/abs/2003ApJ...594..674B/abstract](Bloom et al.(2003)).