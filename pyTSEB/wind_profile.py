# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:50:52 2016

@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main routines for estimating the wind profile above and within a canopy.
It requires the following package.

* :doc:`MO_similarity` for the estimation of adiabatic correctors.

Wind profile functions
----------------------
* :func:`calc_u_C` [Norman1995]_ canopy wind speed.
* :func:`calc_u_C_star` MOST canopy wind speed.
* :func:`calc_u_Goudriaan` [Goudriaan1977]_ wind speed profile below the canopy.
* :func:`calc_A_Goudriaan` [Goudriaan1977]_ wind attenuation coefficient below the canopy.
* :func:`calc_u_Massman` [Massman2007]_ wind speed profile below the canopy.
* :func:`calc_U_b` [Massman2017]_ logarithmic wind profile, dominant near the ground
* :func:`calc_U_t` [Massman20177]_ hyperbolic cosine wind profile, dominant near the top of the canopy.
* :func:`calc_u_star_ratio` [Massman2017]_ ratio of friction velocity and wind speed at the canopy height.
* :func:`cummulative_drag_area` [Massman2017]_ cummulative drag area below a normzalized height.
* :func:`drag_area_index` [Massman2017]_ cummulative drag area below a normzalized height.
* :func:`calc_cummulative_canopy_distribution` [Massman2017]_ non-dimensional cummulative canopy distribution at a normalized height.
* :func:`calc_canopy_distribution` [Massman2017]_ cnon-dimensional canopy distribution at a normalized height.
* :func:`canopy_shape` [Massman2017]_ asymmetrical Gaussian foliage distribution.
"""

import numpy as np

import pyTSEB.MO_similarity as MO  
#==============================================================================
# List of constants used in wind_profile
#==============================================================================

# Drag coefficient (Goudriaan 1977)
c_d = 0.2
# Relative turbulence intensity (Goudriaan 1977)
i_w = 0.5
# Von Karman constant
KARMAN = 0.4
# Size of the normalized height bins in Massman wind profile
BIN_SIZE = 0.0001

def calc_u_C(u_friction, h_C, d_0, z_0M):
    '''[Norman1995]_ wind speed at the canopy, reformulated to use u_friction

    Parameters
    ----------
    u_friction : float
        Wind friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    # The original equation below has been refolmulated to use u_friction:
    # u_C = u * log((h_C - d_0) / z_0M)/(log ((z_u  - d_0) / z_0M)- Psi_M)
    u_C = np.log((h_C - d_0) / z_0M) * u_friction / KARMAN
    return np.asarray(u_C)


def calc_u_C_star(u_friction, h_C, d_0, z_0M, L=float('inf')):
    ''' MOST wind speed at the canopy

    Parameters
    ----------
    u_friction : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    L : float, optional
        Monin-Obukhov length (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).
    '''

    Psi_M = MO.calc_Psi_M((h_C - d_0) / L)
    Psi_M0 = MO.calc_Psi_M(z_0M / L)

    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = (u_friction * (np.log((h_C - d_0) / z_0M) - Psi_M + Psi_M0)) / KARMAN
    return np.asarray(u_C)


def calc_u_Goudriaan(u_C, h_C, LAI, leaf_width, z):
    '''Estimates the wind speed at a given height below the canopy.

    Parameters
    ----------
    U_C : float
        Windspeed at the canopy interface (m s-1).
    h_C : float
        canopy height (m).
    LAI : float
        Efective Leaf (Plant) Area Index.
    leaf_width : float
        effective leaf width size (m).
    z : float
        heigh at which the windsped will be estimated.

    Returns
    -------
    u_z : float
        wind speed at height z (m s-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    .. [Goudriaan1977] Goudriaan (1977) Crop micrometeorology: a simulation study
 '''

    # extinction factor for wind speed
    a = calc_A_Goudriaan(h_C, LAI, leaf_width)
    del LAI, leaf_width
    u_z = u_C * np.exp(-a * (1.0 - (z / h_C)))  # Eq. 4.48 in Goudriaan 1977
    return np.asarray(u_z)


def calc_A_Goudriaan(h_C, LAI, leaf_width):
    ''' Estimates the extinction coefficient factor for wind speed

    Parameters
    ----------
    h_C : float
        canopy height (m)
    LAI : float
        Efective Leaf (Plant) Area Index
    leaf_width : float
        effective leaf width size (m)

    Returns
    -------
    a : float
        exctinction coefficient for wind speed through the canopy

    References
    ----------
    .. [Goudriaan1977] Goudriaan (1977) Crop micrometeorology: a simulation study
    '''

    # Equation in Norman et al. 1995
    k3_prime = 0.28
    a = k3_prime * LAI**(2. / 3.) * h_C**(1. / 3.) * leaf_width**(-1. / 3.)

    return np.asarray(a)


def calc_u_Massman(u_C, h_C, LAI, z, canopy_distribution, Xi_soil=0.0001, C_d=0.2):
    '''' Canopy wind speed. From Eq. 11 of [Massman2017]_.
    Parameters
    ----------
    u_C : float
        Wind speed at the top of the canopy.
    h_C : float
        canopy height
    LAI : float
        Leaf Area Index
    z : float
        height above the ground
    canopy_distribution : array_like
        relative cummulative canopy distribution function
    Xi_soil : float
        ground surface roughness length. Default = 0.00101m.
    C_d_equiv : float
        Equivalent drag coefficient of the individual foliage elements. Default = 0.2.

    Returns
    -------
    u_z : float
        Canopy wind speed.

    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''
    u_C, h_C, LAI, z = map(np.asarray, [u_C, h_C, LAI, z])
    U_b = calc_U_b(z, h_C, Xi_soil)  # Eq.6
    U_t = calc_U_t(z, LAI, h_C, canopy_distribution, Xi_soil, C_d_equiv=C_d)  # Eq. 7
    u_z = u_C * U_b * U_t
    return u_z


def calc_U_b(z, h_C, Xi_soil=0.0025):
    '''' Logarithmic wind profile. Dominant near the ground
    From Eq. 6 of [Massman2017]_.
    Parameters
    ----------
    z : float
        height above the ground
    h_C : float
        canopy height
    Xi_soil : float
        ground surface roughness length. Default = 0.00101m.

    Returns
    -------
    U_b : float
        Non dimensional logarithmic wind profile.

    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''
    z = np.asarray(z)
    h_C = np.asarray(h_C)
    z0_soil = Xi_soil * h_C
    U_b = np.zeros(z0_soil.shape)
    U_b[z > z0_soil] = (np.log(z[z > z0_soil] / z0_soil[z > z0_soil]) /
                        np.log(h_C[z > z0_soil] / z0_soil[z > z0_soil]))
    return U_b


def calc_U_t(z, LAI, h_C, canopy_distribution, Xi_soil=0.0025, C_d_equiv=0.2):
    '''' hyperbolic cosine wind profile. Dominant near the top of the canopy
    From Eq. 7 of [Massman2017]_.
    Parameters
    ----------
    z : float
        height above the ground
    LAI : float
        Leaf Area Index
    h_C : float
        canopy height
    canopy_distribution : array_like
        relative cummulative canopy distribution function
    Xi_soil : float
        ground surface roughness length. Default = 0.00101m.
    C_d_equiv : float
        Equivalent drag coefficient of the individual foliage elements. Default = 0.2.

    Returns
    -------
    U_t : float
        Non dimensional hyperbolic cosine wind profile.

    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''
    upper_limit = z / h_C
    Zeta_h = drag_area_index(LAI, C_d_equiv=C_d_equiv)
    Zeta_Xi = cummulative_drag_area(LAI, canopy_distribution, upper_limit, C_d_equiv=C_d_equiv)
    u_star_ratio = calc_u_star_ratio(Zeta_h, Xi_soil)
    C_surf = 2.0 * u_star_ratio ** 2  # Eq. 9
    N = Zeta_h / C_surf  # Eq. 8
    U_t = np.cosh(N * Zeta_Xi / Zeta_h) / np.cosh(N)
    return U_t


def calc_u_star_ratio(Zeta_h, Xi_0_soil):
    ''' Ratio of friction velocity and wind speed at the canopy height.
    From Eq. 10 of [Massman2017]_.

    Parameters
    ----------
    Zeta_h : float
        Drag area index
    Xi_soil : float
        ground surface roughness length. Default = 0.00101m.

    Returns
    -------
    u_star_ratio : float
        Ratio of friction velocity and wind speed at the canopy height.

    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''

    c1 = 0.38
    c3 = 15.0
    c2 = c1 + KARMAN / np.log(Xi_0_soil)
    u_star_ratio = c1 - c2 * np.exp(-c3 * Zeta_h)  # Eq 10.
    return u_star_ratio


def cummulative_drag_area(LAI, foliage_distribution, upper_limit, C_d_equiv=0.2):
    ''' Cummulative drag area below a normzalized height,
    from Eq. 4 or 5 in [Massman2017]_.

    Parameters
    ----------
    LAI : array_like
        Leaf Area Index
    foliage_distribution : array_like
        cummulative canopy distribution function
    upper_limit : array_like
        Upper heigh normalized value below which the drag area drag_area_distribution
        will be computed. Default=1, i.e. top of the canopy.
    C_d_equiv : float
        drag coefficient Cd described in Eq. 3 of [MassmanXX]_.
        Default 0.2, 0, 0
    Returns
    -------
    Zeta_Xi : float
        Cummulative drag area. By default returns the drag area index,
        see Eq. 4 of [MassmanXX]_.
    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''
    upper_limit = np.asarray(upper_limit)
    upper_limit[upper_limit < 0] = 0
    upper_limit[upper_limit > 1] = 1
    upper_limit = upper_limit / BIN_SIZE
    upper_limit = np.round(upper_limit).astype(np.int32)
    Zeta_Xi = np.zeros(upper_limit.shape)
    Zeta_Xi = LAI * C_d_equiv * foliage_distribution[upper_limit - 1]

    return np.asarray(Zeta_Xi)


def drag_area_index(LAI, C_d_equiv=0.2):
    ''' Cummulative drag area below a normzalized height,
    from Eq. 4 or 5 in [Massman2017]_.

    Parameters
    ----------
    LAI : float
        Leaf Area Index
    C_d_equiv : float
        drag coefficient Cd described in Eq. 3 of [MassmanXX]_.
        Default 0.2, 0, 0
    Returns
    -------
    Zeta_h : float
        Cummulative drag area. By default returns the drag area index,
        see Eq. 4 of [MassmanXX]_.
    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''

    Zeta_h = LAI * C_d_equiv
    return Zeta_h


def calc_cummulative_canopy_distribution(f_a):
    '''Calculates the non-dimensional cummulative canopy distribution.
    From Eq. 1 in [Massman2017]_.

    Parameters
    ----------
    f_a : float
        Non-dimensional canopy distribution at a normalized height.

    Returns
    -------
    f_a : float
        cummulative canopy density.
    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''

    f_a = np.asarray(f_a).reshape(-1)
    f_a_cum = np.zeros(f_a.shape)
    for i in range(np.size(f_a)):
        f_a_cum[i] = (np.sum(f_a[:i + 1]))

    return f_a_cum


def calc_canopy_distribution(Xi_max, sigma_u, sigma_l):
    '''Calculates the non-dimensional canopy distribution at a normalized height.
    From Eq. 1 in [Massman2017]_.

    Parameters
    ----------
    Xi_max : float
        Value of the peak distribution.
    sigma_u : float
        upper standard deviation.
    sigma_l : float
        lower standard deviation.


    Returns
    -------
    f_a : float
        non-dimensional foliage density at a normalized height Xi.
    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''


    f_a = assimetrical_gaussian_distribution(Xi_max, sigma_u, sigma_l)
    f_a = f_a / np.sum(f_a)
    return f_a


def assimetrical_gaussian_distribution(Xi_max, sigma_u, sigma_l, upper_Xi=1):
    ''' Double assimetrical Gaussian distribution function.
    From Eq. 2 of [Massman2017]_.

    Parameters
    ----------
    Xi_max : float
        Value of the peak distribution.
    sigma_u : float
        upper standard deviation.
    sigma_l : float
        lower standard deviation.
    upper_Xi : float
        Upper heigh normalized value below which the density distribution
        will be computed. Default=1, i.e. top of the canopy.

    Returns
    -------
    f_a : array
        Distribution function at equidisitant bins between 0 and upper_Xi.

    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''

    heights = np.arange(0, upper_Xi, BIN_SIZE)
    f_a = np.zeros(heights.shape)
    upper_dist = np.logical_and(heights >= Xi_max, heights <= 1)
    f_a[upper_dist] = np.exp(-(heights[upper_dist] - Xi_max) ** 2 / sigma_u ** 2)
    f_a[~upper_dist] = np.exp(-(Xi_max - heights[~upper_dist]) ** 2 / sigma_l ** 2)
    return f_a


def canopy_shape(h_c, h_b, h_max=0.5):
    ''' Asymmetrical Gaussian foliage distribution.

    Parameters
    ----------
    h_c : float
        Top of the canopy height
    h_b : float
        Height of the first living branch.
    h_max: float
        Relative position between h_c and h_b where the maximum foliage density occurs,
        default=0-5, i.e. peak occurs at the middle of the canopy (~spherical canopy)

    Returns
    -------
    Xi_max : float
        Value of the peak distribution
    sigma_u : float
        upper standard deviation.
    sigma_l : float
        lower standard deviation.
    References
    ----------
    .. [Massman2017] W. J. Massman, J. M. Forthofer, M. A. Finney. An Improved
        Canopy Wind Model for Predicting Wind Adjustment Factors
        and Wildland Fire Behavior, Canadian Journal of Forest Research,
        Pages 594-603, http://dx.doi.org/10.1139/cjfr-2016-0354
    '''

    # Use relative units
    h_b = h_b / h_c
    Xi_max = (h_max * (h_c - h_b) + h_b) / h_c

    # Lower standar deviation
    sigma_l = (Xi_max - h_b) / 2.0
    # Uper standar deviation
    sigma_u = (1.0 - Xi_max) / 2.0

    return Xi_max, sigma_u, sigma_l
