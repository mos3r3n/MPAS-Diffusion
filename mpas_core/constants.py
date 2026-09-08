"""Physical/numerical constants shared across the package.

Values match the MPAS model's own framework constants module
(src/framework/mpas_constants.F in MPAS-Dev/MPAS-Model), not generic
textbook rounded values, so filtering/blending results are consistent
with what MPAS itself uses internally.
"""

R_EARTH = 6_371_229.0          # m, MPAS mpas_constants.F 'a'
RAD2DEG = 180.0 / 3.141592653589793
P0      = 100_000.0            # Pa, MPAS mpas_constants.F 'p0'

GRAVITY = 9.80616              # m s^-2, MPAS mpas_constants.F 'gravity'
RD      = 287.0                # J kg^-1 K^-1, dry-air gas constant, MPAS 'rgas'
RV      = 461.6                # J kg^-1 K^-1, water-vapor gas constant, MPAS 'rv'
CP      = 3.5 * RD             # J kg^-1 K^-1, specific heat at const p, MPAS 'cp' (=1004.5)
RD_CP   = RD / CP              # = 2/7 ~ 0.285714, MPAS 'rgas/cp' (kappa)
EPS     = RD / RV              # ~0.622, dry/moist gas-constant ratio
