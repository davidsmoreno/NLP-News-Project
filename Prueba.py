# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:35:59 2023

@author: David
"""

#Punto 1



def pares(x):
    if x%2==0:
        return True
    else:
        return False

def pares_e_impares(N,M):
    suma_pares=0
    suma_impares=0
    for i in range(M,N+1):
        if pares(i)==True:
            suma_pares=suma_pares+i
        else:
            suma_impares=suma_impares+i
            
    return suma_pares,suma_impares
    
print(pares_e_impares(10,5))



def moda(arr):
    contador={}
    for i in arr:
        if i in contador:
            contador[i]+=1
        else:
            contador[i]=1
    moda=None
    max_frec=0
    for i, frec in contador.items():
        if frec>max_frec:
            moda=i
            max_frec=frec
    return moda
            
arr=[10,4,6,3,5,4,1,63,-5,8]


def mediana(arr):
    arr.sort()
    n=len(arr)
    if n%2!=0:
        med=arr[n//2]
    else:
        med=(arr[n//2-1]+arr[n//2])/2
    return med
        


print(moda(arr))
print(mediana(arr))


        