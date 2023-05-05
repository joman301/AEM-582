""" 
Writen by: 
    Joshua Vondracek 
    AEM 582 Space Systems (Spring 2023) 
    Aerospace Engineering & Mechanics 
    University of Alabama 
    jbvondracek@crimson.ua.edu
     
Created:        2023.4.13
Latest Rev.:    2023.5.04

TODO: CHANGE THIS STUFF
PROBLEM STATEMENT: Develop a model capable of calculating the 
needed delta-V, the time of flight (TOF), and the propellant mass fraction 
(pmf) for a Hohmann transfer maneuver from a user-specified 
initial orbital altitude to a target orbital altitude ranging from the 
lower end of Low Earth Orbit (LEO) to an orbit with the period of that  
associated with a geosynchronous orbit. Plot these values as a function  
of orbital radius divided by the initial orbital radius, indicating  
the user-specified initial orbital radius as a red circle. What conclusions  
can be drawn from these plots? For this problem, assume a specific  
impulse of 400 s and that the period of a geosynchronous orbit is 
86161.0 s. Do not perform a validation analysis for this problem. 

ASSUMPTIONS: 
- The initial and final orbits are circular. 
- There are no orbital perturbations 
- The Earth may be assumed to have a uniform mass distribution 
- The Earth may be approximated as a perfect sphere. 
- Lowest orbital altitude is 160 km AGL. 
- Convergence criterion for ndp was both del-V & pmf < 0.001 for R/Ri = 1. 
     
# GOVERNING EQUATIONS: 
- Orbital Speed: Circular Orbit 
- Orbital Speed: Elliptical Orbit 
- time of flight (TOF) 
- propellant mass fraction (from rocket equation) 
""" 


# Import necessary libraries to run this
import math
import numpy as np
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def ideal_SRP(S_0, c, r_0, r):
  ''' Calculate the Solar Radiation Pressure acting on the sail for
  a given distance away from sun (r), solar constant (S_0), and speed
  of light (c)
  '''
  return (S_0/c)*(r_0/r)**2
  

def ideal_SRP_force(P, A, alpha, normal_vector):
  ''' Calculate the ideal case Solar Radiation Pressure force
  based off of the calculated pressure (P), area of the sail (A),
  angle between the sun line and normal vector of the sail (alpha),
  and the normal vector of the solar sail
  '''
  return 2*P*A*math.cosd(alpha)**2*normal_vector

def NPR_a_c(P_0, A, m, a1, a2):
   return (2*P_0*A/m) * (a1 + a2)

# Integrate f(x) = x^2
def f1(x):
    return x**2

# ===================================================================
# Functions that run once to calculate constants                    |
# ===================================================================

def populate_dicts(dicts, r_0, v_0, angle_alpha):
  for dict in dicts:
    dicts[dict]["s"] = (1-dicts[dict]["d"])/dicts[dict]["rho"]
    dicts[dict]["B_f"] = 0.79                 # Lambertian constant for front of sail (ASSUMED VALUE FOR ALL)
    dicts[dict]["B_b"] = 0.55                 # Lambertian constant for back of sail (ASSUMED VALUE FOR ALL)
    rho_inf, s_inf, epsilon_finf, epsilon_b0, B_f0, B_b0 = calc_optical_coeffs_inf(dicts[dict]["deg_factor"],dicts[dict]["rho"], dicts[dict]["s"], dicts[dict]["e_f"], dicts[dict]["e_b"], dicts[dict]["B_f"], dicts[dict]["B_b"])
    dicts[dict]["rho_inf"] = rho_inf          # Solar reflectance as t -> inf
    dicts[dict]["s_inf"] = s_inf              # Specular reflection as t -> inf
    dicts[dict]["e_f_inf"] = epsilon_finf     # Epsilon for front of sail as t -> inf
    dicts[dict]["e_b_inf"] = epsilon_b0       # Epsilon for back of sail as t -> inf
    dicts[dict]["B_f_inf"] = B_f0             # Lambertian constant for front of sail as t -> inf
    dicts[dict]["B_b_inf"] = B_b0             # Lambertian constant for back of sail as t -> inf
    dicts[dict]["rho_over_t"] = [dicts[dict]["rho"]]
    dicts[dict]["s_over_t"] = [dicts[dict]["s"]]
    dicts[dict]["e_f_over_t"] = [dicts[dict]["e_f"]]
    dicts[dict]["p"] = [[r_0, 0]]             # Position vector [km]
    dicts[dict]["v"] = [[0, v_0]]             # Velocity vector [km/s]
    dicts[dict]["a"] = [[0, 0]]               # Acceleration vector [km/s/s]
    dicts[dict]["r"] = [r_0]                  # Radial dist from sun [km]
    dicts[dict]["angle_alpha"] = angle_alpha  # Angle of sail [deg]
  
  return None


def calc_optical_coeffs_inf(d, rho_0, s_0, epsilon_f0, epsilon_b0, B_f0, B_b0):
  ''' Calculates the optical coefficients as r goes to infinity (max degradation).
  "d" is the degradation constant that we set, range 0-1
  '''
  rho_inf = rho_0/(1+d)
  s_inf = s_0/(1+d)
  epsilon_finf = (1+d)*epsilon_f0
  return rho_inf, s_inf, epsilon_finf, epsilon_b0, B_f0, B_b0

def calc_solar_pressure(S_0, c, r_0, r):
  return (S_0/c) * (r_0/r)**2

def calc_characteristic_accleration(P, A, m, material_dict):
  '''Calculate the characteristic acceleration of the solar sail
  with the pressure (P), area of the sail (A), and the mass of
  the solar sail (m). Assuming we're using the NPR model, also
  include two more terms, a1 and a2
  '''
  ideal_a_c = 2*P*A/m
  rho, e_f, e_b, B_f, B_b, s = material_dict["rho"], material_dict["e_f"], material_dict["e_b"], material_dict["B_f"], material_dict["B_b"], material_dict["s"]
  a1 = 0.5 * (1 + s * rho)
  a2 = 0.5 * (B_f * (1 -s) * rho + (1-rho) * ((e_f*B_f - e_b*B_b)/(e_f+e_b)))
  NPR_a_c = ideal_a_c * (a1 + a2)
  return NPR_a_c

def calc_lightness_number(a_c, mu, r_0):
  '''Calculcate the lightness number by using characteristic acceleration
  of the solar sail (a_c), the gravational acceleration of the sun (mu/r_0**2)
  '''
  beta = a_c/(mu/r_0**2)
  return beta

# ====================================================================
# Functions that run in the loop to numerically integrate the system |
# ====================================================================

def calc_solar_constant(au_in_km, one_au_const, current_r):
  ''' Find the new solar constant based off the inverse square law,
  runs every time step to integrate numerically
  '''
  return (au_in_km/current_r**2) * one_au_const


def calc_optical_degradation(coeff, coeff_inf, gamma, sigma_t):
  ''' Calculating the new optical coefficient based off degradation,
  runs every time step to integrate numerically
  '''
  return coeff_inf + (coeff-coeff_inf)*math.exp(-gamma*sigma_t)


def calc_force_coeffs(s, rho, epsilon_f, epsilon_b, B_f, B_b):
  ''' Calculates the force coefficients b1, b2, and b3 as a function of
  optical coefficients
  '''
  b1 = 0.5*(1-s*rho)
  b2 = s*rho
  b3 = 0.5*(B_f * (1-s)*rho + (1-rho)*(epsilon_f*B_f - epsilon_b*B_b)/(epsilon_f+epsilon_b))
  return b1, b2, b3

def calc_accel(beta, b1, b2, b3, mu, r, alpha, e_r, n):
  ''' Calculate the acceleration of the solar sail
  '''
  alpha_deg = alpha * 180 / math.pi
  return [(beta/(b1+b2+b3))*(mu/r**2)*math.cos(alpha_deg)*(b1*e_r[i] + (b2*math.cos(alpha_deg))*n[i]) for i in range(len(n))]


if __name__ == "__main__":
  t0 = 0
  t_step = 0.01 # Time Step [days]
  s_in_d = 86400  # Number of seconds in a day [sec/day]
  m = 8 # Mass of sail + 6U cubesat [kg]
  A = 58.1  # Area of solar sail [m^2]
  v_0 = 29.77  # Initial y component velocity of satellite [km/s]
  alpha_angle = 45  # Angle of face of 
  m_sun = 1.9891e30  # Mass of sun [kg]
  r_sun = 696000  # Radius of sun [km]
  gamma = math.log(2) / 1368 / 2  # Half life solar radiation dose (half of 1368 constant)
  save_graph = True


  materials = {
    "Mylar" : {
      "alpha" : 0.08,     # Absorbance
      "rho" : 0.92,       # Solar reflectance
      "e_f" : 0.0218,     # Emittance divided by 11
      "e_b" : 0.24,       # Emittance
      "d" : 0.1043,       # Reflectance divided 8.816; diffusivity
      "s" : 0,            # Specular reflection factor - (1-d)/rho
      "deg_factor" : 0.01 # User-set degradation factor
    },

    "CP1" : {
      "alpha" : 0.09,     # Absorbance
      "rho" : 0.91,       # Solar reflectance
      "e_f" : 0.0245,     # Emittance divided by 11
      "e_b" : 0.27,       # Emittance
      "d" : 0.103,        # Reflectance divided 8.816; diffusivity
      "s" : 0,            # Specular reflection factor - (1-d)/rho
      "deg_factor" : 0.01 # User-set degradation factor
    },

    "Teonex" : {
      "alpha" : 0.08,     # Absorbance
      "rho" : 0.92,       # Solar reflectance
      "e_f" : 0.0227,     # Emittance divided by 11
      "e_b" : 0.25,       # Emittance
      "d" : 0.104,        # Reflectance divided 8.816; diffusivity
      "s" : 0,            # Specular reflection factor - (1-d)/rho
      "deg_factor" : 0.01 # User-set degradation factor
    },

    "Kapton" : {
      "alpha" : 0.1,      # Absorbance
      "rho" : 0.9,        # Solar reflectance
      "e_f" : 0.0436,     # Emittance divided by 11
      "e_b" : 0.48,       # Emittance
      "d" : 0.102,        # Reflectance divided 8.816; diffusivity
      "s" : 0,            # Specular reflection factor - (1-d)/rho
      "deg_factor" : 0.01 # User-set degradation factor
    },

    "APICAL AH" : {
      "alpha" : 0.344,    # Absorbance
      "rho" : 0.656,      # Solar reflectance
      "e_f" : 0.0725,     # Emittance divided by 11
      "e_b" : 0.798,      # Emittance
      "d" : 0.039,        # Reflectance divided 8.816; diffusivity
      "s" : 0,            # Specular reflection factor - (1-d)/rho
      "deg_factor" : 0.01 # User-set degradation factor
    },
  }


  S_0 = 1368  # Solar emittance constant? [W/m^2]
  au = 149597870.7  # Astronomical Unit [km]
  c = 299792458  # Speed of light [m/s]
  mu = 1.3271e20  # Sun gravitational parameter [m^3/s^2]
  crash = False

  r_0 = au
  r_mars = 1.5 * au
  r = [r_0]

  p = [[r_0, 0]]  # 2D Position Vector [km]
  v = [[0,10]]  # 2D Velocity Vector [km/s]
  a = [[0, 0]]  # 2D Acceleration Vector [km/s/s]
  
  # Create all shared constants/ calculated constants that aren't looped for materials
  populate_dicts(materials, r_0, v_0, alpha_angle)

  figure, axes = plt.subplots()
  

# Loop through all the materials with the same simulation
  for material in materials:
    P_0 = calc_solar_pressure(S_0, c, r_0, r_0)  # Calculating initial solar pressure (perpendicular to sun line)
    a_c = calc_characteristic_accleration(P_0, A, m, materials[material])  # Characteristic acceleration (assuming 1 AU, perpendicular, NPR model)
    beta = calc_lightness_number(a_c, mu, r_0)  # Calculating lightness number
    crash = False

    i = 0
    # Simulation loop that generates all of the data points, numerically solves the equations
    while materials[material]["r"][i] < r_mars:

      # Calculate the angle of the satellite wrt the x-axis
      sail_theta = np.arccos(materials[material]["p"][i][0]/materials[material]["r"][i])

      # Numerical integration section
      b1, b2, b3 = calc_force_coeffs(materials[material]["s_over_t"][i], materials[material]["rho_over_t"][i], materials[material]["e_f_over_t"][i], materials[material]["e_b_inf"], materials[material]["B_f_inf"], materials[material]["B_b_inf"])
      
      # Unit vector of sun-spacecraft line
      e_r = [math.cos(sail_theta), math.sin(sail_theta)]

      # Normal vector to solar sail
      n = [math.cos(sail_theta + materials[material]["angle_alpha"]), math.sin(sail_theta + materials[material]["angle_alpha"])]
      a = calc_accel(beta, b1, b2, b3, mu, materials[material]["r"][i], materials[material]["angle_alpha"], e_r, n)
      F_sail = [m*x for x in a]
      F_sun = [(6.67430e-11 * m * m_sun / (materials[material]["r"][i]*1000)**2)*x for x in e_r]

      F_total_mag = [(y - z)/1000 for y,z in zip(F_sail, F_sun)]
      #F_vector = [F_total_mag*math.cos(sail_theta), F_total_mag*math.sin(sail_theta)]
      materials[material]["a"].append([x/m for x in F_total_mag])
      materials[material]["v"].append([y + z for y,z in zip(materials[material]["v"][i],[x * t_step * s_in_d for x in materials[material]["a"][i]])])
      materials[material]["p"].append([y + z for y,z in zip(materials[material]["p"][i],[x * t_step * s_in_d for x in materials[material]["v"][i]])])
      
      # Calculate the new optical coefficients at the end of the time step
      solar_const_current = calc_solar_constant(au, S_0, materials[material]["r"][i])
      
      materials[material]["rho_over_t"].append(calc_optical_degradation(materials[material]["rho_over_t"][i], materials[material]["rho_inf"], gamma, solar_const_current))
      materials[material]["s_over_t"].append(calc_optical_degradation(materials[material]["s_over_t"][i], materials[material]["s_inf"], gamma, solar_const_current))
      materials[material]["e_f_over_t"].append(calc_optical_degradation(materials[material]["e_f_over_t"][i], materials[material]["e_f"], gamma, solar_const_current))
      materials[material]["r"].append(math.sqrt(materials[material]["p"][i+1][0]**2 + materials[material]["p"][i+1][1]**2))

      if materials[material]["p"][i+1][0]**2 + materials[material]["p"][i+1][1]**2 <= r_sun**2:
        print("You crashed into the sun on day " + str(i*t_step) + " with " + material)
        crash = True
        break 

      if i > 20000000:
        print("Broke")
        break
      
      i += 1

    if not crash:
      print("You made it to Mars on day " +  str(i*t_step) + " using " + material)

    x,y = zip(*materials[material]["p"])
    plt.plot(x, y, label = material)
    


  sun = plt.Circle(( 0 , 0 ), r_mars*0.025, color='y')
  start_point = plt.Circle((au, 0), r_mars*0.02, color='k')
  martian_orbit = plt.Circle((0,0), r_mars, color = 'k', fill = False)
  earth_orbit = plt.Circle((0,0), au, color = 'k', fill = False)
  
  axes.set_aspect(1)
  
  axes.add_artist(sun)
  axes.add_artist(start_point)
  axes.add_artist(martian_orbit)
  axes.add_artist(earth_orbit)
  plt.legend(loc="upper left")
  plt.title('Solar Sail Trajectories')
  plt.xlim(-1.1*r_mars,1.1*r_mars)
  plt.ylim(-1.1*r_mars,1.1*r_mars)   
  if save_graph:
    file_name_base = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_'
    file_name_end = ".png"
    orbit_plot = file_name_base + "orbit_alphang_" + str(alpha_angle) + "_sailsize_" + str(A) + "_m_" + str(m) + file_name_end
    plt.savefig(orbit_plot, bbox_inches = "tight", dpi = 300)
  plt.show()
  