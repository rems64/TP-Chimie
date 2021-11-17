# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:19:00 2021

@author: remy.fayet
"""

import numpy as np
import matplotlib.pyplot as plt

# Valeurs expérimentales
x = np.array([19, 26, 32, 34.5, 39])
ux = np.array([0.2/np.sqrt(3) for k in x])

y = np.array([170, 106, 66, 57, 41])
uy = np.array([2/np.sqrt(3) for k in y])

# On convertit en kelvin
x+=273.2    

def tirages(N):
    # Listes pour les résultats de la régression pour chaque expérience
    ai = []
    bj = []

    # On réalise N expériences aléatoires en distribution uniforme    
    tX = np.random.uniform(x-ux,x+ux,(N,len(x)))
    tY = np.random.uniform(y-uy,y+uy,(N,len(x)))
    
    # On prépare la régression : y=ax+b <=> ln(ti) = Ea/RT + b
    X = 1/tX
    Y = np.log(tY)
    
    # Pour chaque expérience virtuelle
    for i in range(len(X)):
        # Régression linéaire
        coef = np.polyfit(X[i],Y[i],1)
        # On ajoute les résultats à chaque liste
        ai.append(coef[0])
        bj.append(coef[1])
    
    # On calcule maintenant la moyenne des coefficients directeurs et des
    #                                               ordonnées à l'origine
    a = np.mean(ai)
    b = np.mean(bj)
    
    # On calcule les incertitudes
    ua = np.std(ai,ddof=1)
    ub = np.std(bj, ddof=1)
    
    # Moyennes pour Monte-Carlo
    unSurX = np.mean(1/tX, 0)
    logY = np.mean(np.log(tY), 0)
    
    # Axe des abscisses avec 2% de marge
    tx = np.linspace(unSurX.min()*0.98,unSurX.max()*1.02,10)
    plt.plot(tx,[a*k+b for k in tx], label="modélisation")
    
    # A partir de Ea/R=a on obtient Ea = a*R
    eA = a*8.314E-3
    
    zscore = abs(eA - 54)/(ua*8.314E-3)
    
    # Description pour la console et le graphe
    descript = ("a:"+str(round(a, 2))+"\nua:"+str(round(ua, 2))+
                "\n\nb:"+str(round(b, 2))+"\nub:"+str(round(ub, 2)))
    
    # Affichage console
    print(descript)
    print("\n")
    print("E =", a*8.314E-3, "kJ/mol")
    print("Z-score =", zscore)
    
    # Graphe
    plt.figtext(-0.2, 0.38, descript, fontsize=14)
    plt.errorbar(unSurX,logY, yerr=np.std(logY, 0, ddof=1),
                 xerr=np.std(unSurX, 0, ddof=1), label="Points expérimentaux")
    plt.legend()
    plt.ylabel("ln(ti)")
    plt.xlabel("1/T (en K^-1)")
    plt.title("Détermination énergie d'activation\nE(exp)="+
              str(round(eA, 1))+"kJ/mol       z-score ="+str(round(zscore, 2)))
    plt.show()


# On réalise 10000 tirages
tirages(10000)
