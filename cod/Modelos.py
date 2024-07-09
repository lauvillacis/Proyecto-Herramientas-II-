#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 08:35:56 2024

@author: lauravillacis
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from cod.CargarDatos import CargarDatos


class Modelos(CargarDatos):

    def __init__(self, base=None):
        '''
        Es el metodo constructor de los objetos de la clase
        
        Parametros
        -------
        base : data frame
            Los datos que se quieren utilizar con los modelos

        Returns
        -------
        No devuelve nada
        '''
        CargarDatos.__init__(self, base)
        self.__covariables = pd.DataFrame()
        self.__variable_predecir =  pd.DataFrame()
        self.__covariables_train =  pd.DataFrame()
        self.__covariables_test =  pd.DataFrame()
        self.__predecir_train =  pd.DataFrame()
        self.__predecir_test =  pd.DataFrame()
        
    #Métodos get
    @property
    def covariables(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'covariables'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        covariables: dataframe
            Es el valor del atributo 'covariables'
        '''
        return self.__covariables
    
    @property
    def variable_predecir(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'variable_predecir'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        variable_predecir: dataframe
            Es el valor del atributo 'variable_predecir'
        '''
        return self.__variable_predecir
    
    @property
    def covariables_train(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'covariables_train'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        covariables_train: dataframe
            Es el valor del atributo 'covariables_train'
        '''
        return self.__covariables_train
    
    @property
    def covariables_test(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'covariables_test'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        covariables_test: dataframe
            Es el valor del atributo 'covariables_test'
        '''
        return self.__covariables_test
    
    @property
    def predecir_train(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'predecir_train'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        predecir_train: dataframe
            Es el valor del atributo 'predecir_train'
        '''
        return self.__predecir_train
    
    @property
    def predecir_test(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'predecir_test'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        predecir_test: dataframe
            Es el valor del atributo 'predecir_test'
        '''
        return self.__predecir_test
    
    #Métodos set
    @covariables.setter
    def covariables(self, nuevas_covariables):
        '''
        Es el metodo set, para establecer el valor del atributo 'covariables'
        
        Parametros
        -------
        nuevas_covariables : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'covariables' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__covariables = nuevas_covariables
        
    @variable_predecir.setter
    def variable_predecir(self, nueva_variable_predecir):
        '''
        Es el metodo set, para establecer el valor del atributo 'variable_predecir'
        
        Parametros
        -------
        nueva_variable_predecir : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'variable_predecir' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__variable_predecir = nueva_variable_predecir
        
    @covariables_train.setter
    def covariables_train(self, nuevas_covariables_train):
        '''
        Es el metodo set, para establecer el valor del atributo 'covariables_train'
        
        Parametros
        -------
        nuevas_covariables_train : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'covariables_train' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__covariables_train = nuevas_covariables_train
    
    @covariables_test.setter
    def covariables_test(self, nuevas_covariables_test):
        '''
        Es el metodo set, para establecer el valor del atributo 'covariables_test'
        
        Parametros
        -------
        nuevas_covariables_test : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'covariables_test' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__covariables_test = nuevas_covariables_test
    
    @predecir_train.setter
    def predecir_train(self, nueva_predecir_train):
        '''
        Es el metodo set, para establecer el valor del atributo 'predecir_train'
        
        Parametros
        -------
        nueva_predecir_train : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'predecir_train' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__predecir_train = nueva_predecir_train
        
    @predecir_test.setter
    def predecir_test(self, nueva_predecir_test):
        '''
        Es el metodo set, para establecer el valor del atributo 'predecir_test'
        
        Parametros
        -------
        nueva_predecir_test : dataframe
            Esto es el  dataframe que se quiere asignar para modificar 
            el atributo 'predecir_test' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__predecir_test = nueva_predecir_test
        
            
    def sets(self):
        '''
        Parte la base de datos en covariables y la variable a predecir. 
        Además establece los conjuntos para entrenar y probar el modelo.
        
        Parametros
        -------
        No lleva parámetros explícitos
    
        Returns
        -------
        No devuelve nada, cambia los atributos de la clase
        '''
        self.__covariables = self.base.iloc[:, :-1] 
        self.__variable_predecir = self.base.iloc[:, -1]
        self.__covariables_train, self.__covariables_test, self.__predecir_train, self.__predecir_test = train_test_split(self.__covariables, self.__variable_predecir, test_size=0.3, random_state=10)
        
        
    def naive_bayes(self):
        '''
        Aplica el modelo llamado Naive bayes.
        
        Parametros
        -------
        No lleva parámetros explícitos
    
        Returns
        -------
        resultados: dataframe
            La primera columna son los valores reales del conjunto de testeo y 
            la segunda son los valores predichos por el modelo en cuestión.
        '''
        naive_bayes = GaussianNB()
        escala = StandardScaler()
        covariables_train = escala.fit_transform(self.__covariables_train)
        naive_bayes.fit(covariables_train, self.__predecir_train)
        covariables_test = escala.transform(self.__covariables_test)
        predicciones = naive_bayes.predict(covariables_test)
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados  
    
    def regresion_logistica(self):
        '''
        Aplica el modelo llamado Regresión logistica.
        
        Parametros
        -------
        No lleva parámetros explícitos
    
        Returns
        -------
        resultados: dataframe
            La primera columna son los valores reales del conjunto de testeo y 
            la segunda son los valores predichos por el modelo en cuestión.
        '''
        regresion_log = LogisticRegression()
        regresion_log.fit(self.__covariables_train, self.__predecir_train)
        predicciones = regresion_log.predict(self.__covariables_test)    
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados 
    
    def k_vecinos_cercanos(self,k_vecinos, validacion_cruzada = False):
        '''
        Aplica el modelo llamado K nearest neighbors.
        
        Parametros
        -------
        No lleva parámetros explícitos
    
        Returns
        -------
        resultados: dataframe
            La primera columna son los valores reales del conjunto de testeo y 
            la segunda son los valores predichos por el modelo en cuestión.
        '''
        k_vecinos_cerca = KNeighborsClassifier(n_neighbors= k_vecinos) #Aquí se define el numeor de vecinos
        escala = StandardScaler()
        covariables_train = escala.fit_transform(self.__covariables_train)
        k_vecinos_cerca.fit(covariables_train, self.__predecir_train)
        covariables_test = escala.fit_transform(self.__covariables_test)
        predicciones = k_vecinos_cerca.predict(covariables_test)
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados 
    
    def arbol_de_decision(self):
        '''
        Aplica el modelo llamado Arbol de decision.
        
        Parametros
        -------
        No lleva parámetros explícitos
    
        Returns
        -------
        resultados: dataframe
            La primera columna son los valores reales del conjunto de testeo y 
            la segunda son los valores predichos por el modelo en cuestión.
        '''
        arbol = DecisionTreeClassifier()
        arbol.fit(self.__covariables_train, self.__predecir_train)
        predicciones = arbol.predict(self.__covariables_test)  
        resultados = {
            'valor_real' : list(self.__predecir_test),
            'valor_predicho': list(predicciones)
        }
        resultados = pd.DataFrame(resultados)
        return resultados 
    
        
        
        
        
        
        
        
        
        
        
        