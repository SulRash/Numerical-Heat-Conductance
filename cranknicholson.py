from node import Node

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# Constant
kappa = float(0.835)

# An equation used for groups of nodes with the left being a boundary condition.
def solving_equation1(known: list[Node], deltax: int, deltat: int) -> Tuple[list[int], list[float]]:

    # Calculates known term.
    T_iminus1_j = known[0].value
    T_i_j = known[1].value
    T_iplus1_j = known[2].value
    T_iminus1_jplus1 = known[3].value

    gamma = ( ( 2 * deltax * deltax ) / ( kappa * deltat ) )

    known_term = ( ( T_iminus1_j / ( gamma + 2 ) ) + ( ( T_i_j * ( gamma - 2 ) ) / ( gamma + 2 ) ) + ( T_iplus1_j / (gamma + 2) ) + ( T_iminus1_jplus1 / ( gamma + 2 ) ) )

    # Calculates coefficient of unknown terms.
    coef_of_T_iplus1_jplus1 = ( -1 * ( 1 / ( gamma + 2 ) ) )
    unknown_coef = [1, coef_of_T_iplus1_jplus1]

    return known_term, unknown_coef

# An equation used for groups of nodes with no boundary conditions present.
def solving_equation2(known: list[Node], deltax: int, deltat: int) -> Tuple[list[int], list[float]]:

    # Calculates known term.
    T_iminus1_j = known[0].value
    T_i_j = known[1].value
    T_iplus1_j = known[2].value

    gamma = ( ( 2 * deltax * deltax ) / ( kappa * deltat ) )

    known_term = ( ( T_iminus1_j / ( gamma + 2 ) ) + ( ( T_i_j * ( gamma - 2 ) ) / ( gamma + 2 ) ) + ( T_iplus1_j / (gamma + 2) ) )

    # Calculates coefficient of unknown terms.
    coef_of_T_iminus1_jplus1 = ( -1 * ( 1 / ( gamma + 2 ) ) )
    coef_of_T_iplus1_jplus1 = ( -1 * ( 1 / ( gamma + 2 ) ) )
    unknown_coef = [coef_of_T_iminus1_jplus1, 1, coef_of_T_iplus1_jplus1]

    return known_term, unknown_coef

# An equation used for groups of nodes with the right being a boundary condition.
def solving_equation3(known: list[Node], deltax: int, deltat: int) -> Tuple[list[int], list[float]]:

    # Calculates known term.
    T_iminus1_j = known[0].value
    T_i_j = known[1].value
    T_iplus1_j = known[2].value
    T_iplus1_jplus1 = known[3].value

    gamma = ( ( 2 * deltax * deltax ) / ( kappa * deltat ) )

    known_term = ( ( T_iminus1_j / ( gamma + 2 ) ) + ( ( T_i_j * ( gamma - 2 ) ) / ( gamma + 2 ) ) + ( T_iplus1_j / (gamma + 2) ) + ( T_iplus1_jplus1 / ( gamma + 2 ) ) )

    # Calculates coefficient of unknown terms.
    coef_of_T_iminus1_jplus1 = ( -1 * ( 1 / ( gamma + 2 ) ) )
    unknown_coef = [coef_of_T_iminus1_jplus1, 1]

    return known_term, unknown_coef

def solve_triagonal_matrix(known_terms: list[int], unknown_coefs: list[list[float]]):

    # An array of arrays set at 0 initially is used as a sparse matrix to solve for.
    tri_matrix = np.array([[0]*len(unknown_coefs)]*len(unknown_coefs), dtype=float)

    # Triagonal matrix is populated by coefficients of the unknown nodes.
    z = 0
    for i in range(len(unknown_coefs)):
        for j in range(len(unknown_coefs[i])):
            if i == 0:
                tri_matrix[i][j] = unknown_coefs[i][j]
            elif i == len(unknown_coefs) - 1:
                tri_matrix[i][len(unknown_coefs)-2 + j ] = unknown_coefs[i][j]
            else:
                tri_matrix[i][j+z] = unknown_coefs[i][j]
        if i >= 1:
            z += 1

    # The right hand side of the triagonal matrix with known values.
    tri_matrix_rhs = np.array(known_terms, dtype=float)
    
    # Solution is generated use scipy.
    solution = spsolve(tri_matrix, tri_matrix_rhs)
    return solution

def solve_nodes(deltax: int, deltat: int, t: int, d: int, constant_initial: bool) -> list[Node]:

    known_nodes = []

    # Generating default nodes.
    for j in range(int(t/deltat)+1):
        known_nodes.append(Node(i=0, j=j, value=float(0)))

    for j in range(int(t/deltat)+1):
        known_nodes.append(Node(i=int(d/deltax), j=j, value=float(0)))

    # If temperature is initially 500 unifromly around the bar.
    if constant_initial == True:
        for i in range(1,int(100/deltax)):
            known_nodes.append(Node(i=i, j=0, value=float(500), known=True))

    # If temperature is distributed different across bar.
    else:

        for i in range(1,int(d/deltax)):

            # Using equation given as T(x,t) = -xg(x - L) + c where c = 400 and g = 0.1
            temperature = 400 + ( ( i * deltax ) * (0.1) * ( d - ( i * deltax ) ) )
            known_nodes.append(Node(i=i, j=0, value=float(temperature)))

    # This loop focuses on moving up vertically in the node graph.
    for y in range(0,int(t/deltat)):

        # Known node array is initialised.
        knowns = []

        for _ in range(1, int(d/deltax)):
            knowns.append([])

        # Known terms and unknown coefficients are initialised for triagonal matrix.
        known_terms = []
        unknown_coefs = []

        # This loop focuses on moving right horizontally in the node graph.
        for x in range(0,(int(d/deltax)-1)):

            # This loop is responsible for checking j and j+1 nodes.
            for j in range(0+y,2+y):

                # This loop is responsible for checking i-1, i, and i+1 nodes.
                for i in range(0+x,3+x):

                    max = 2+x

                    # Checks if node already exists, to isolate known and unknown nodes.
                    exists = any(node.code == "T{i}{j}".format(i=i, j=j) for node in known_nodes)
                    
                    if exists:

                        # If node exists, the node is searched for to append to the list of knowns.
                        for _, node in enumerate(known_nodes):
                            if node.i == i and node.j == j:
                                break

                        # Node is appended to the list in knowns that corresponds to which group of nodes we are looking at.
                        for z in range(len(knowns)):
                            if max < 3+z:
                                knowns[z].append(node)
                                break
        
        # Three separate equations are used depending on where the group of nodes evaluated is in relation to boundary conditions.
        for z in range(len(knowns)):

            if z == 0:
                known_term, unknown_coef = solving_equation1(knowns[z], deltax, deltat)
            elif z == len(knowns) - 1:
                known_term, unknown_coef = solving_equation3(knowns[z], deltax, deltat)
            else:
                known_term, unknown_coef = solving_equation2(knowns[z], deltax, deltat)
            
            known_terms.append(known_term)
            unknown_coefs.append(unknown_coef)
        
        # Solution is of triagonal matrix is generated to find temperature of nodes, then appended to the list of known nodes.
        solution = solve_triagonal_matrix(known_terms, unknown_coefs)

        i = 0
        for z in range(len(solution)):
            i += 1
            known_nodes.append(Node(i=i, j=y+1, value=solution[z]))

    return known_nodes

def temperature_distribution_plot(nodes, deltat):


    # Generates all six separate lines.
    l0 = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []

    for node in nodes:
        if node.j == 0:
            l0.append(node.value)
    
    for node in nodes:
        if node.j == 1:
            l1.append(node.value)

    for node in nodes:
        if node.j == 2:
            l2.append(node.value)

    for node in nodes:
        if node.j == 3:
            l3.append(node.value)

    for node in nodes:
        if node.j == 4:
            l4.append(node.value)

    for node in nodes:
        if node.j == 5:
            l5.append(node.value)

    for node in nodes:
        if node.j == 6:
            l6.append(node.value)
    
    l0.append(l0.pop(0))
    l1.append(l1.pop(0))
    l2.append(l2.pop(0))
    l3.append(l3.pop(0))
    l4.append(l4.pop(0))
    l5.append(l5.pop(0))
    l6.append(l6.pop(0))

    # x coordinates.
    bar_length = np.array([0, 20, 40, 60, 80, 100])

    # Plots graph
    plt.plot(bar_length, l0, label="0s")
    plt.plot(bar_length, l1, label="{t1}s".format(t1=(1*deltat)))
    plt.plot(bar_length, l2, label="{t2}s".format(t2=(2*deltat)))
    plt.plot(bar_length, l3, label="{t3}s".format(t3=(3*deltat)))
    plt.plot(bar_length, l4, label="{t4}s".format(t4=(4*deltat)))
    plt.plot(bar_length, l5, label="{t5}s".format(t5=(5*deltat)))
    plt.plot(bar_length, l6, label="{t6}s".format(t6=(6*deltat)))

    plt.xlabel("Location in rod (cm)")
    plt.ylabel("Temperature (celsius)")
    plt.legend()
    plt.show()

def temperature_evolution(nodes1, nodes2, nodes3, deltax, deltat, t):

    # Generates line for deltat = 10, deltax = 20
    l1  = []

    # Generates line for deltat = 5, deltax = 20
    l2 = []

    # Generates line for deltat = 10, deltax = 10
    l3 = []

    for node in nodes1:
        if node.i == 1:
            l1.append(node.value)

    for node in nodes2:
        if node.i == 1:
            l2.append(node.value)
    
    for node in nodes3:
        if node.i == 2:
            l3.append(node.value)

    # Generates x coordinates necessary.
    time = np.linspace(0, t, (int(t/deltat)+1))
    time_halfstep = np.linspace(0, t, (int((2*t)/deltat) + 1))

    # Plots graph
    plt.plot(time, l1, "b-p" ,label="Delta X = {deltax}cm, Delta t = {deltat}s".format(deltax=deltax, deltat=deltat))
    plt.plot(time_halfstep, l2, "r-" ,label="Delta X = {deltax}cm, Delta t = {deltat}s".format(deltax=deltax, deltat=int(deltat/2)))
    plt.plot(time, l3, "k-x" ,label="Delta X = {deltax}cm, Delta t = {deltat}s".format(deltax=int(deltax/2), deltat=deltat))
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (celsius)")
    plt.legend()
    plt.show()

deltat = 10
deltax = 20
t = 60
d = 100
constant_initial = False

nodes1 = solve_nodes(deltax, deltat, t, d, constant_initial)
nodes2 = solve_nodes(deltax, int(deltat/2), t, d, constant_initial)
nodes3 = solve_nodes(int(deltax/2), deltat, t, d, constant_initial)
temperature_distribution_plot(nodes1, deltat)
temperature_evolution(nodes1, nodes2, nodes3, deltax, deltat, t)
