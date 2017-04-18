# This python program calculates the magnon current j(x,y) in a ferromagnet
# and plot it using the data output from micromangetic simulations (mumax3).
# The data inputs are the position and magnetization data in .txt files
# (The .ovf output files can be parsed to get .txt files)
# The magnon current is defined based on the paper Nature Materials:
# Thermally driven rachet motion of a skyrmion microcrystal and topological 
# magnon Hall effect. 

import cmath
import numpy as np
import math, time, os
import matplotlib.pyplot as plt
from matplotlib import rc

# Pauli matrices
sigma_x = [[0, 1], [1, 0]]
sigma_y = [[0, -1j], [1j, 0]]

#Constatns
DbyJ = 0.27
Ms = 8e5
# Returns the dot product of two lists : each of two elements
def dot(list1, list2):
    if list1 == 0 or list2 == 0: return 0
    len1 = len(list1)
    len2 = len(list2)
    if len1 != len2: raise ValueError("Two z's have different length !")
    return sum([i*j for (i, j) in zip(list1, list2)])

# Returns a 2*1 list from a 2*2 matrix and 2*1 list multiplication
def matrix_by_list(matrix, list_):
    i_00, i_01 = matrix[0][0], matrix[0][1]
    i_10, i_11 = matrix[1][0], matrix[1][1]
    j_0, j_1 = list_[0], list_[1]
    a_list = [i_00*j_0 + i_01*j_1, i_10*j_0 + i_11*j_1]
    if len(a_list) != 2: raise ValueError("Error in matrix-list product !")
    return a_list

# Returns a table of position and current data for zero temperature case
def zero_temp_current(data_list):
    row = len(data_list)
    col = len(data_list[0])
    print("The row and col length of zero temp table are: ", row, col)
    list_of_r_thetaphi_and_z = []
    for i in range(row):
        θ_i = cmath.acos(data_list[i][5]/Ms)
        φ_i = cmath.acos(data_list[i][3]/(Ms*math.sqrt(1 - data_list[i][5]**2)))
        zup = cmath.cos(θ_i/2)
        zdown = cmath.exp(1j * φ_i)*cmath.sin(θ_i/2)
        z = [zup, zdown]
        zd = np.conjugate(z).tolist()
        r = [data_list[i][0],data_list[i][1]] # taking only x and y coordinates for position
        theta_phi = [θ_i, φ_i]
        list_of_r_thetaphi_and_z.append([r, theta_phi, z, zd])
    num = len(list_of_r_thetaphi_and_z)
    print("Just for the cross check (zero temp) - should be TRUE: ", row == num)
    my_list = list_of_r_thetaphi_and_z
    position_current_list = []
    for i in range(num-1):
        zi = my_list[i][2]
        zid =  my_list[i][3]
        delx = my_list[i+1][0][0] - my_list[i][0][0]
        dely = my_list[i+1][0][1] - my_list[i][0][1]
        delz= list(np.array(my_list[i+1][2]) - np.array(my_list[i][2]))
        delzd= list(np.array(my_list[i+1][2]) - np.array(my_list[i][2]))
        if delx != 0:
            delzdelx = list(map(lambda x: x/delx, delz))
            delzddelx = list(map(lambda x: x/delx, delzd))
        else:
            delzdelx = 0
            delzddelx = 0
        if dely != 0:
            delzdely = list(map(lambda x: x/dely, delz))
            delzddely = list(map(lambda x: x/dely, delzd))
        else:
            delzdely  = 0
            delzddely = 0
        ax = -1j*(dot(zid, delzdelx) - dot(delzddelx, zi))
        ay = -1j*(dot(zid, delzdely) - dot(delzddely, zi))
        position_current_list.append([my_list[i][0], [ax, ay]])
    return position_current_list

# Finding j_mu from the finite temperature mumax3 output files
def finite_temp_current(data_list):
    row = len(data_list)
    col = len(data_list[0])
    print("The row and col length of finite temp table  are: ", row, col)
    list_of_r_thetaphi_and_z = []
    for i in range(row):
        #if i <= 5: print("mx, my, mz are: ", data_list[i][0], data_list[i][1], data_list[i][2])
        θ_i = cmath.acos(data_list[i][5]/Ms)
        φ_i = cmath.acos(data_list[i][3]/(Ms*math.sqrt(1 - data_list[i][5]**2)))
        zup = cmath.cos(θ_i/2)
        zdown = cmath.exp(1j * φ_i)*cmath.sin(θ_i/2)
        z = [zup, zdown]
        zd = np.conjugate(z).tolist()
        r = [data_list[i][0],data_list[i][1], data_list[i][2]]
        theta_phi = [θ_i, φ_i]
        list_of_r_thetaphi_and_z.append([r, theta_phi, z, zd])
    num = len(list_of_r_thetaphi_and_z)
    print("Just for the cross check finite temp data input- should be TRUE: ", row == num)
    my_list = list_of_r_thetaphi_and_z
    position_current_list = []
    for i in range(num-1):
        zi = my_list[i][2]
        zid =  my_list[i][3]
        delx = (my_list[i+1][0][0] - my_list[i][0][0])
        dely = (my_list[i+1][0][1] - my_list[i][0][1])
        delz= list(np.array(my_list[i+1][2]) - np.array(my_list[i][2]))
        delzd= list(np.array(my_list[i+1][2]) - np.array(my_list[i][2]))
        if delx != 0:
            delzdelx = list(map(lambda x: x/delx, delz))
            delzddelx = list(map(lambda x: x/delx, delzd))
        else:
            delzdelx = 0
            delzddelx = 0
        if dely != 0:
            delzdely = list(map(lambda x: x/dely, delz))
            delzddely = list(map(lambda x: x/dely, delzd))
        else:
            delzdely  = 0
            delzddely = 0
        # from the finite temperature data: Calculating only finite temp part of current
        jx1 = -1j*dot(zid, delzdelx) - DbyJ*dot(zid, matrix_by_list(sigma_x, zi))
        jy1 = -1j*dot(zid, delzdely) - DbyJ*dot(zid, matrix_by_list(sigma_y, zi))
        position_current_list.append([my_list[i][0], [jx1, jy1]])
    return position_current_list

# Fining magnon-current from finite and zero temperature currents
def magnon_current(data_table_0, data_table_t):
    if len(data_table_0) != len(data_table_t): raise ValueError("Zero and finite temp tables have not same length !")
    print("Everything going great !")
    position_magnon_current_list = []
    for i in range(len(data_table_t)):
        position = data_table_t[i][0]
        jx_1 = data_table_t[i][1][0] - data_table_0[i][1][0]
        #jx = (abs(jx_1 + jx_1.conjugate()))/2
        jx = (jx_1 + jx_1.conjugate())/2
        jy_1 = data_table_t[i][1][0] - data_table_0[i][1][0]
        #jy = (abs(jy_1 + jy_1.conjugate()))/2
        jy = (jy_1 + jy_1.conjugate())/2
        position_magnon_current_list.append([position, [jx.real, jy.real]])
        if (i >= 25650 and i <= 25665): print("magnon-current is: ", [jx.real, jy.real])
    return position_magnon_current_list

#Plotting the position vs magnon-current vector plot
def Plot(data_table):
    x_position_list = []
    y_position_list = []
    jx_list = []
    jy_list = []
    jx0 = 0
    jy0 = 0
    for i in range(int((len(data_table) + 1)/3) - 1):
        x_position = data_table[3*i][0][0]*1e9   # converting position into nm
        y_position = data_table[3*i][0][1]*1e9   # converting position into nm
        x_center = 64
        y_center = 64
        radius = math.sqrt((x_position - x_center)**2 + (y_position - y_center)**2)
        if radius > 59:
            continue
        jx = data_table[3*i][1][0]
        jy = data_table[3*i][1][1]
        x_position_list.append(x_position)
        y_position_list.append(y_position)
        jx_list.append(jx)
        jy_list.append(jy)
    #cmap=plt.cm.jet
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.quiver(x_position_list, y_position_list, jx_list, jy_list, color='red')
    plt.title(r"$\vec{j} = (j_x, j_y)$ at each lattice points", fontsize=14)
    plt.xlabel(r"$x-position (nm)$", fontsize=14)
    plt.ylabel(r"$y-position (nm)$", fontsize=14)
    plt.xlim(0, 128)
    plt.ylim(0, 128)
    plt.show()

# Loading mumax3 ouput file
zero_temp_mumax_table = np.loadtxt("zero_temp_mag_texture.txt", skiprows=5)
finite_temp_mumax_table = np.loadtxt("finite_temp_mag_texture.txt", skiprows=5)
# Calling the functions 
data_table_0 = zero_temp_current(zero_temp_mumax_table)
data_table_t = finite_temp_current(finite_temp_mumax_table)
input_data = magnon_current(data_table_0, data_table_t)
Plot(input_data)

