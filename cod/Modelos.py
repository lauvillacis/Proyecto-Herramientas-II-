#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 08:35:56 2024

@author: lauravillacis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
#Se importa de la librería sklearn la clase que realiza el modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from cod.CargarDatos import CargarDatos


class Modelos(CargarDatos):
    def __init__(self, base, covariables, variable_predecir, covariables_train, covariables_test, predecir_train, predecir_test):
        CargarDatos.__init__(self, base)
        self.__covariables = covariables
        self.__variable_predecir = variable_predecir
        self.__covariables_train = covariables_train
        self.__covariables_test = covariables_test
        self.__predecir_train = predecir_train
        self.__predecir_test = predecir_test
        
        
    def sets(self):
        covariables = self.base.iloc[:, :-1] 
        self.__covariables = covariables
        predecir = self.base.iloc[:, -1]
        self.__variable_predecir = predecir
        covariables_train, covariables_test, predecir_train, predecir_test = train_test_split(covariables, predecir, test_size=0.3, random_state=10)
        self.__covariables_train = covariables_train
        self.__covariables_test = covariables_test
        self.__predecir_train = predecir_train
        self.__predecir_test = predecir_test
        print(covariables_train.shape)
        
    def naive_bayes(self, validacion_cruzada = False):
        naive_bayes = GaussianNB()
        if validacion_cruzada:
            predicciones = cross_val_predict(naive_bayes,self.__covariables , self.__variable_predecir, cv=5)
        else: 
            naive_bayes.fit(self.__covariables_train, self.__predecir_train)
            predicciones = naive_bayes.predict(self.__covariables_test)
            
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        
        return resultados  
    
    def regresion_logistica(self, validacion_cruzada = False):
        regresion_log = LogisticRegression()
        if validacion_cruzada:
            predicciones = cross_val_predict(regresion_log, self.__covariables , self.__variable_predecir, cv=5)
        else: 
            regresion_log = LogisticRegression()
            regresion_log.fit(self.__covariables_train, self.__predecir_train)
            predicciones = regresion_log.predict(self.__covariables_test)
            
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        
        return resultados 
    
    def k_vecinos_cercanos(self,k_vecinos, validacion_cruzada = False): #cambiar el nombre de la funcion y del modelo??
        k_vecinos_cercanos = KNeighborsClassifier(n_neighbors= k_vecinos) #Aquí se define el numeor de vecinos
        if validacion_cruzada:
            scaler = StandardScaler()
            covariables_estandar = scaler.fit_transform(self.__covariales)
            predicciones = cross_val_predict(k_vecinos_cercanos, covariables_estandar , self.__variable_predecir, cv=5)
        else:
            k_vecinos_cercanos.fit(self.__covariables_train, self.__predecir_train)
            predicciones = k_vecinos_cercanos.predict(self.__covariables_test)
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        
        return resultados 
    
    def arbol_de_decision(self, validacion_cruzada = False):
        arbol = DecisionTreeClassifier()
        if validacion_cruzada:
            predicciones = cross_val_predict(arbol, self.__covariables , self.__variable_predecir, cv=5)
        else: 
            arbol.fit(self.__covariables_train, self.__predecir_train)
            predicciones = arbol.predict(self.__covariables_test)
            
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        
        return resultados 
        
        
        
        
        
        
        
        
        
        
        