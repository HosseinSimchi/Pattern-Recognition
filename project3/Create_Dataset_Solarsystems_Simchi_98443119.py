import numpy as np
import pandas as pd
from datetime import date
import math
import matplotlib.pyplot as plt
year_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="A")
year_v = year_v[1:-1]
year_v = year_v.values
year_v = year_v.flatten()

month_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="B")
month_v = month_v[1:-1]
month_v = month_v.values
month_v = month_v.flatten()

day_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="C")
day_v = day_v[1:-1]
day_v = day_v.values
day_v = day_v.flatten()

hour_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="D")
hour_v = hour_v[1:-1]
hour_v = hour_v.values
hour_v = hour_v.flatten()

minute_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="E")
minute_v = minute_v[1:-1]
minute_v = minute_v.values
minute_v = minute_v.flatten()

Lat_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="F")
Lat_v = Lat_v[1:-1]
Lat_v = Lat_v.values
Lat_v = Lat_v.flatten()

Long_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="G")
Long_v = Long_v[1:-1]
Long_v = Long_v.values
Long_v = Long_v.flatten()

Mag_v = pd.read_excel ('./data_base.xlsx', header=None, index_col=False, usecols="H")
Mag_v = Mag_v[1:-1]
Mag_v = Mag_v.values
Mag_v = Mag_v.flatten()

## Delta T (minutes)
day_0 = date(2021, 1, 15)
Delta_t = []
for n in range(0, np.size(year_v)):
    day_1 = date(year_v[n], month_v[n], day_v[n])
    Delta_d = day_0 - day_1
    Delta_time = Delta_d.days * 24 * 60 
    Delta_time = Delta_time - (hour_v[n]*60+minute_v[n]) + (4*60 + 45) 
    Delta_t.append(Delta_time)
    
Delta_t = np.array(Delta_t)

## Earth
D_earth = 149.6           # million km, distance to Sun
Omega_earth = 360/365.25  # degree/day, angular velocity
Omega_earth /= 24*60      # degree/minute, angular velocity
theta_0_earth = 114.2563  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_earth = -Omega_earth * Delta_t + theta_0_earth
theta_earth = theta_earth % 360
X_earth = D_earth * np.cos(math.pi * theta_earth / 180) # million km
Y_earth = D_earth * np.sin(math.pi * theta_earth / 180) # million km

## Mercury
D_mercury = 57.9            # million km, distance to Sun
Omega_mercury = 360/88      # degree/day, angular velocity
Omega_mercury /= 24*60      # degree/minute, angular velocity
theta_0_mercury = 356.5764  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_mercury = -Omega_mercury * Delta_t + theta_0_mercury
theta_mercury = theta_mercury % 360
X_mercury = D_mercury * np.cos(math.pi * theta_mercury / 180)               # million km
Y_mercury = D_mercury * np.sin(math.pi * theta_mercury / 180)               # million km
R_mercury_earth = np.sqrt(np.power(X_mercury-X_earth, 2) + np.power(Y_mercury-Y_earth, 2))  # million km 
alpha = abs(theta_mercury - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_mercury_earth = np.degrees(np.arcsin(D_mercury/R_mercury_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_mercury_earth)):
    if D_mercury > max(D_earth, R_mercury_earth[n]):
        theta_mercury_earth[n] = 180 - theta_mercury_earth[n]

## Venus
D_venus = 108.2           # million km, distance to Sun
Omega_venus = 360/224.7   # degree/day, angular velocity
Omega_venus /= 24*60      # degree/minute, angular velocity
theta_0_venus = 253.6944  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_venus = -Omega_venus * Delta_t + theta_0_venus
theta_venus = theta_venus % 360
X_venus = D_venus * np.cos(math.pi * theta_venus / 180)               # million km
Y_venus = D_venus * np.sin(math.pi * theta_venus / 180)               # million km
R_venus_earth = np.sqrt((X_venus-X_earth)**2 + (Y_venus-Y_earth)**2)  # million km 
alpha = abs(theta_venus - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_venus_earth = np.degrees(np.arcsin(D_venus/R_venus_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_venus_earth)):
    if D_venus > max(D_earth, R_venus_earth[n]):
        theta_venus_earth[n] = 180 - theta_venus_earth[n]
        
## Mars
D_mars = 227.9           # million km, distance to Sun
Omega_mars = 360/687     # degree/day, angular velocity
Omega_mars /= 24*60      # degree/minute, angular velocity
theta_0_mars = 75.0507   # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_mars = -Omega_mars * Delta_t + theta_0_mars
theta_mars = theta_mars % 360
X_mars = D_mars * np.cos(math.pi * theta_mars / 180)               # million km
Y_mars = D_mars * np.sin(math.pi * theta_mars / 180)               # million km
R_mars_earth = np.sqrt((X_mars-X_earth)**2 + (Y_mars-Y_earth)**2)  # million km 
alpha = abs(theta_mars - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_mars_earth = np.degrees(np.arcsin(D_mars/R_mars_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_mars_earth)):
    if D_mars > max(D_earth, R_mars_earth[n]):
        theta_mars_earth[n] = 180 - theta_mars_earth[n]

## Jupiter
D_jupiter = 778.6           # million km, distance to Sun
Omega_jupiter = 360/4331    # degree/day, angular velocity
Omega_jupiter /= 24*60      # degree/minute, angular velocity
theta_0_jupiter = 307.7578  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_jupiter = -Omega_jupiter * Delta_t + theta_0_jupiter
theta_jupiter = theta_jupiter % 360
X_jupiter = D_jupiter * np.cos(math.pi * theta_jupiter / 180) # million km
Y_jupiter = D_jupiter * np.sin(math.pi * theta_jupiter / 180) # million km
R_jupiter_earth = np.sqrt((X_jupiter-X_earth)**2 + (Y_jupiter-Y_earth)**2)  # million km 
alpha = abs(theta_jupiter - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_jupiter_earth = np.degrees(np.arcsin(D_jupiter/R_jupiter_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_jupiter_earth)):
    if D_jupiter > max(D_earth, R_jupiter_earth[n]):
        theta_jupiter_earth[n] = 180 - theta_jupiter_earth[n]
        
## Saturn
D_saturn = 1433.5          # million km, distance to Sun
Omega_saturn = 360/10747   # degree/day, angular velocity
Omega_saturn /= 24*60      # degree/minute, angular velocity
theta_0_saturn = 303.7680  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_saturn = -Omega_saturn * Delta_t + theta_0_saturn
theta_saturn = theta_saturn % 360
X_saturn = D_saturn * np.cos(math.pi * theta_saturn / 180) # million km
Y_saturn = D_saturn * np.sin(math.pi * theta_saturn / 180) # million km
R_saturn_earth = np.sqrt((X_saturn-X_earth)**2 + (Y_saturn-Y_earth)**2)  # million km 
alpha = abs(theta_saturn - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_saturn_earth = np.degrees(np.arcsin(D_saturn/R_saturn_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_saturn_earth)):
    if D_saturn > max(D_earth, R_saturn_earth[n]):
        theta_saturn_earth[n] = 180 - theta_saturn_earth[n]

## Uranus
D_uranus = 2872.5          # million km, distance to Sun
Omega_uranus = 360/30589   # degree/day, angular velocity
Omega_uranus /= 24*60      # degree/minute, angular velocity
theta_0_uranus = 39.2134   # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_uranus = -Omega_uranus * Delta_t + theta_0_uranus
theta_uranus = theta_uranus % 360
X_uranus = D_uranus * np.cos(math.pi * theta_uranus / 180) # million km
Y_uranus = D_uranus * np.sin(math.pi * theta_uranus / 180) # million km
R_uranus_earth = np.sqrt((X_uranus-X_earth)**2 + (Y_uranus-Y_earth)**2)  # million km 
alpha = abs(theta_uranus - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_uranus_earth = np.degrees(np.arcsin(D_uranus/R_uranus_earth*np.sin(math.pi * alpha / 180))) # degree
for n in range(0, np.size(theta_uranus_earth)):
    if D_uranus > max(D_earth, R_uranus_earth[n]):
        theta_uranus_earth[n] = 180 - theta_uranus_earth[n]
        
## Neptun
D_neptun = 4495.1          # million km, distance to Sun
Omega_neptun = 360/59800   # degree/day, angular velocity
Omega_neptun /= 24*60      # degree/minute, angular velocity
theta_0_neptun = 350.2058  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_neptun = -Omega_neptun * Delta_t + theta_0_neptun
theta_neptun = theta_neptun % 360
X_neptun = D_neptun * np.cos(math.pi * theta_neptun / 180) # million km
Y_neptun = D_neptun * np.sin(math.pi * theta_neptun / 180) # million km
R_neptun_earth = np.sqrt((X_neptun-X_earth)**2 + (Y_neptun-Y_earth)**2)  # million km 
alpha = abs(theta_neptun - theta_earth)
alpha = np.where(alpha > 180, 360 - alpha, alpha)

theta_neptun_earth = np.degrees(np.arcsin(D_neptun/R_neptun_earth*np.sin(math.pi * alpha / 180))) # degree 
for n in range(0, np.size(theta_neptun_earth)):
    if D_neptun > max(D_earth, R_neptun_earth[n]):
        theta_neptun_earth[n] = 180 - theta_neptun_earth[n]

## Moon
D_moon = 0.384            # million km, distance to Earth
Omega_moon = 360/29.53      # degree/day, angular velocity around Earth
Omega_moon /= 24*60      # degree/minute, angular velocity
theta_0_moon = 318.8875  # degree,     angle at Jan 15, 2021, 4:45 Tehran time
theta_moon = -Omega_moon * Delta_t + theta_0_moon # relative to Earth
theta_moon = theta_moon % 360
X_moon = X_earth + D_moon * np.cos(math.pi * theta_moon / 180)               # million km
Y_moon = Y_earth + D_moon * np.sin(math.pi * theta_moon / 180)               # million km

V_moon  = np.array([X_moon, Y_moon]) + np.array([-X_earth, -Y_earth])
V_sun = np.array([-X_earth, -Y_earth])
theta_moon_earth=[]

for n in range(0, np.size(X_moon)):
    cosine_angle = np.dot(V_moon[:,n], V_sun[:,n]) / (np.linalg.norm(V_moon[:,n]) * np.linalg.norm(V_sun[:,n]))
    angle = np.degrees(np.arccos(cosine_angle))    
    theta_moon_earth.append(angle)
    
theta_moon_earth = np.array(theta_moon_earth)

data = list(zip(year_v, month_v, day_v, hour_v, minute_v,\
                R_mercury_earth, theta_mercury_earth, R_venus_earth, theta_venus_earth, \
                R_mars_earth, theta_mars_earth, R_jupiter_earth, theta_jupiter_earth, \
                R_saturn_earth, theta_saturn_earth, R_uranus_earth, theta_uranus_earth, \
                R_neptun_earth, theta_neptun_earth, theta_moon_earth))
data = pd.DataFrame.from_dict(data)
data.to_csv('C:\\Users\\Lenovo\\Desktop\\data_base_3.csv')  
