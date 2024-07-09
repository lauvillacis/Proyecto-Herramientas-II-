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

    def __init__(self, base=None):
        CargarDatos.__init__(self, base)
        self.__covariables = pd.DataFrame()
        self.__variable_predecir =  pd.DataFrame()
        self.__covariables_train =  pd.DataFrame()
        self.__covariables_test =  pd.DataFrame()
        self.__predecir_train =  pd.DataFrame()
        self.__predecir_test =  pd.DataFrame()
        
        
    def sets(self):
        self.__covariables = self.base.iloc[:, :-1] 
        self.__variable_predecir = self.base.iloc[:, -1]
        self.__covariables_train, self.__covariables_test, self.__predecir_train, self.__predecir_test = train_test_split(self.__covariables, self.__variable_predecir, test_size=0.3, random_state=10)
        
        
    def naive_bayes(self):
        naive_bayes = GaussianNB()
        escala = StandardScaler()
        covariables_train = escala.fit_transform(self.__covariables_train)
        naive_bayes.fit(covariables_train, self.__predecir_train)
        covariables_test = escala.fit_transform(self.__covariables_test)
        predicciones = naive_bayes.predict(covariables_test)
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados  
    
    def regresion_logistica(self):
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
        escala = StandardScaler()
        covariables_train = escala.fit_transform(self.__covariables_train)
        k_vecinos_cercanos.fit(covariables_train, self.__predecir_train)
        covariables_test = escala.fit_transform(self.__covariables_test)
        predicciones = k_vecinos_cercanos.predict(covariables_test)
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados 
    
    def arbol_de_decision(self):
        arbol = DecisionTreeClassifier()
        arbol.fit(self.__covariables_train, self.__predecir_train)
        predicciones = arbol.predict(self.__covariables_test)  
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados 
    
        
        
        
        
        
        
        
        
        
        
        