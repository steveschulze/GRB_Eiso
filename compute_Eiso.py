import  argparse
from    astropy.cosmology import FlatLambdaCDM
from    astropy.io import ascii
from    astropy import table
import  numpy as np
import  psutil
from    scipy import stats, integrate
from    joblib import Parallel, delayed

parser     = argparse.ArgumentParser(description='Programme to do compute the isotropic energy of GRBs')
parser.add_argument('--file',    type     = str,
                                 help     = 'Input file',
                                 required = True)

parser.add_argument('--output',  type     = str,   
                                 help     = 'Prefix of output file (Default: out_$FILE)',
                                 default  = 'out_')

parser.add_argument('--H0',      type     = float, 
                                 help     = 'Hubble constant (Default: 67.3 km/s/Mpc)',
                                 default  = 67.3)

parser.add_argument('--Omega_M', type     = float, 
                                 help     = 'Hubble constant (Default: 0.315)',         
                                 default  = 0.315)

parser.add_argument('--nint',    type     = int,   
                                 help     = 'Number of steps in numerical integration (Default: 100)',
                                 default  = 100)

parser.add_argument('--nmc',     type     = int,   
                                 help     = 'Number of steps in MC simulation (Default: 5000)',        
                                 default  = 5000)

args                     = parser.parse_args()

''' How to prepare input file?

Fluence is given in 10e-7 erg cm^-2

'''

# Energy

E1_REST                  = 0.1    # keV
E2_REST                  = 10000  # keV

def gamma_band(E, EPEAK, ALPHA, BETA):

    E0                   = EPEAK / (ALPHA + 2.0)

    if E <= (ALPHA - BETA) * E0:
        return E**ALPHA * np.exp(-E/E0)
    else:
        return ((ALPHA-BETA)*E0)**(ALPHA-BETA) * np.exp(BETA-ALPHA) * E**BETA

def gamma_cpl(E, EPEAK, ALPHA):
    E0                   = EPEAK / (ALPHA + 2.0)
    return E**ALPHA * np.exp(-E/E0)

def k_corr(E1_OBS, E2_OBS, E1_REST, E2_REST, REDSHIFT, ALPHA, BETA, EPEAK, NINT):

    energy_range_obs     = 10**np.linspace(np.log10(E1_OBS), np.log10(E2_OBS), NINT)
    energy_range_rest    = 10**np.linspace(np.log10(E1_REST / (1. + REDSHIFT)) , np.log10(E2_REST / (1. + REDSHIFT)), NINT)

    #print(energy_range_obs)

    if BETA != 0.:

        nominator        = [gamma_band(x, EPEAK, ALPHA, BETA) for x in energy_range_rest]
        denominator      = [gamma_band(x, EPEAK, ALPHA, BETA) for x in energy_range_obs]

    elif BETA == 0. and  EPEAK != 0.:

        nominator        = [gamma_cpl(x, EPEAK, ALPHA) for x in energy_range_rest]
        denominator      = [gamma_cpl(x, EPEAK, ALPHA) for x in energy_range_obs]

    else:
        nominator        = 0.
        denominator      = 1.

    return integrate.simps(nominator   * energy_range_rest, energy_range_rest) / \
           integrate.simps(denominator * energy_range_obs,  energy_range_obs)

def compute_Eiso(DATA):

    cosmo     = FlatLambdaCDM(H0=args.H0, Om0=args.Omega_M)

    E1_REST              = 1
    E2_REST              = 1E4
    NINT                 = args.nint
    NMC                  = args.nmc

    print("Processing: %s" %DATA['GRB'])

    # Resample data

    # Fluence

    for x in ['FLUENCE_ERRP', 'FLUENCE_ERRM', 'ALPHA_ERRP', 'ALPHA_ERRM', 'BETA_ERRP', 'BETA_ERRM', 'EPEAK_ERRP', 'EPEAK_ERRM']:
        DATA[x] = DATA[x] / 1.6

    if (DATA['FLUENCE_ERRP'] != 0.):
        u                = np.random.uniform(size = NMC)
        temp_fluence     = np.where(u < 0.5, stats.norm.ppf(u, DATA['FLUENCE'], DATA['FLUENCE_ERRM']),
                                             stats.norm.ppf(u, DATA['FLUENCE'], DATA['FLUENCE_ERRP']))

    else:
        temp_fluence     = np.zeros(NMC) + DATA['FLUENCE']

    # alpha

    u                    = np.random.uniform(size = NMC)

    if DATA['ALPHA_ERRM']!= 0:
        temp_alpha       = np.where(u < 0.5, stats.norm.ppf(u, DATA['ALPHA'], DATA['ALPHA_ERRM']),
                                             stats.norm.ppf(u, DATA['ALPHA'], DATA['ALPHA_ERRP']))

    else:
        temp_alpha       = np.zeros(NMC) + DATA['ALPHA']

    # beta

    if (DATA['BETA']     != 0.) and (DATA['BETA_ERRP'] != 0.):
        u                = np.random.uniform(size = NMC)
        temp_beta        = np.where(u < 0.5, stats.norm.ppf(u, DATA['BETA'], DATA['BETA_ERRM']),
                                             stats.norm.ppf(u, DATA['BETA'], DATA['BETA_ERRP']))

    elif (DATA['BETA']   != 0.) and (DATA['BETA_ERRP'] == 0.):
        temp_beta        = np.zeros(NMC) + DATA['BETA']

    else:
        temp_beta        = np.zeros(NMC)

    # Epeak

    if (DATA['EPEAK']    != 0.) and (DATA['EPEAK_ERRP'] != 0.):
        u                = np.random.uniform(size = NMC)
        temp_epeak       = np.where(u < 0.5, stats.norm.ppf(u, DATA['EPEAK'], DATA['EPEAK_ERRM']),
                                             stats.norm.ppf(u, DATA['EPEAK'], DATA['EPEAK_ERRP']))

    elif (DATA['EPEAK']  != 0.) and (DATA['EPEAK_ERRP'] == 0.):
        temp_epeak       = np.zeros(NMC) + DATA['EPEAK']

    else:
        temp_epeak       = np.zeros(NMC)

    # Redshift

    if DATA['REDSHIFT_ERRP'] != 0.:
        u                = np.random.uniform(size = NMC)
        temp_redshift    = np.where(u < 0.5, stats.norm.ppf(u, DATA['REDSHIFT'], DATA['REDSHIFT_ERRM']),
                                             stats.norm.ppf(u, DATA['REDSHIFT'], DATA['REDSHIFT_ERRP']))

    else:
        temp_redshift    = np.zeros(NMC) + DATA['REDSHIFT']

    #print(temp_redshift)
    #import pdb; pdb.set_trace()
    #print (temp_redshift)
    temp_dl              = cosmo.luminosity_distance(temp_redshift).value * 3.086e+24
    #temp_dl              = [cosmo.luminosity_distance(x).value for x in temp_redshift]
    #print(temp_dl)

    # Compute k-correction and Eiso
    # Speed up calculation with multiprocessing.

    k_correction         = np.array([k_corr(DATA['E1_OBS'], DATA['E2_OBS'], E1_REST, E2_REST, DATA['REDSHIFT'], \
                                     -temp_alpha[x], -temp_beta[x], temp_epeak[x], NINT) for x in range(NMC)])

    E_iso                = 4 * np.pi * temp_dl**2 / (1+temp_redshift) * temp_fluence * 1e-7 * k_correction

    # Remove nan's

    k_correction         = k_correction[~np.isnan(k_correction)]
    E_iso                = E_iso[~np.isnan(E_iso)]
    length               = len(E_iso)

    # Compute median value and 1 sigma error intervals

    k_correction         = np.array([np.percentile(k_correction, 50), \
                                     np.percentile(k_correction, 50+68.2/2.) - np.percentile(k_correction, 50),\
                                     np.percentile(k_correction, 50)         - np.percentile(k_correction, 50-68.2/2.),])

    E_iso                = np.array([np.percentile(E_iso, 50), \
                                     np.percentile(E_iso, 50+68.2/2.)        - np.percentile(E_iso, 50),\
                                     np.percentile(E_iso, 50)                - np.percentile(E_iso, 50-68.2/2.)])

    # Write to output table

    output               = table.Table(np.array([DATA['INDEX'], DATA['GRB'], \
                                                 DATA['REDSHIFT'], DATA['REDSHIFT_ERRP'], DATA['REDSHIFT_ERRM'], \
                                                 k_correction[0], k_correction[1], k_correction[2], \
                                                 np.log10(E_iso[0]), np.log10(E_iso[0]+E_iso[1])-np.log10(E_iso[0]), np.log10(E_iso[0]) - np.log10(E_iso[0]-E_iso[2]),
                                                 length]),\
                                        names = ('INDEX', 'GRB',
                                                'REDSHIFT', 'REDSHIFT_ERRP', 'REDSHIFT_ERRM',
                                                'KVAL', 'KVAL_ERRP', 'KVAL_ERRM',
                                                'LOGEISO', 'LOGEISO_ERRP', 'LOGEISO_ERRM',
                                                'NMC'),\
                                        dtype = ('i', 'S100',
                                                'f', 'f', 'f',
                                                'f', 'f', 'f',
                                                'f', 'f', 'f',
                                                'i'))

    # Some formatting

    for x in ['REDSHIFT', 'REDSHIFT_ERRP', 'REDSHIFT_ERRM', 'KVAL', 'KVAL_ERRP', 'KVAL_ERRM', 'LOGEISO', 'LOGEISO_ERRP', 'LOGEISO_ERRM']:
        output[x].format='4.3f'

    return output

# Data

data                     = ascii.read(args.file)
data['INDEX']            = range(len(data))

output                   = Parallel(n_jobs= int(psutil.cpu_count()/2), backend="multiprocessing")(delayed(compute_Eiso)(x) for x in data)
#output                   = [compute_Eiso(x) for x in data]
output                   = table.vstack(output)
output.sort('INDEX')
del output['INDEX']

ascii.write(output, args.output + args.file, overwrite=True, delimiter='\t')