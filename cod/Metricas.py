#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:16:29 2024

@author: lauravillacis
"""
from cod.CargarDatos import CargarDatos
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
#Se importa de la librería sklearn la clase que realiza el modelo Naive Bayes
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings("ignore")


class Metricas():
    def __init__(self, base):
        self.__base = base
        
    def matriz_de_confusion(self, valores_reales, valores_predichos):
        matriz_confusion = confusion_matrix(valores_reales, valores_predichos)
        return matriz_confusion
    
    def puntaje_de_exactitud(self,valores_reales, valores_predichos):
        '''
        Devuelve la medida  de la proporción de predicciones correctas

        Parameters
        ----------
        valores_reales : TYPE
            DESCRIPTION.
        valores_predichos : TYPE
            DESCRIPTION.

        Returns
        -------
        exactitud : TYPE
            DESCRIPTION.

        '''
        exactitud = accuracy_score(valores_reales, valores_predichos)
        return exactitud

   
    def puntaje_de_precision(self, valores_reales, valores_predichos):
        '''
        Devuelve 

        Parameters
        ----------
        valores_reales : TYPE
            DESCRIPTION.
        valores_predichos : TYPE
            DESCRIPTION.
        etiquetas : TYPE
            DESCRIPTION.

        Returns
        -------
        precision : TYPE
            DESCRIPTION.

        '''
        precision = precision_score(valores_reales, valores_predichos, average = None)
        return precision
    
    def puntaje_F1(self, valores_reales, valores_predichos):
        puntaje_f1 = f1_score(valores_reales, valores_predichos, average = None) 
        return puntaje_f1
    
    def puntaje_recall(self, valores_reales, valores_predichos):
        puntaje_recall = recall_score(valores_reales, valores_predichos, average = None) 
        return puntaje_recall
    
    
    def reporte_de_clasificacion(self, valores_reales, valores_predichos, etiquetas):
        reporte = classification_report(valores_reales, valores_predichos, target_names = etiquetas)
        return reporte

       
    def puntaje_validacion_cruzada(self, modelo, k_vecinos = 0):
        covariables = self.__base.iloc[:, :-1] 
        variable_predecir = self.__base.iloc[:, -1]
        escala = StandardScaler()
        if modelo == 'naive bayes':
            covariables = escala.fit_transform(covariables)
            estimador = GaussianNB()
        elif modelo == 'regresion logistica':
            estimador = LogisticRegression()
        elif modelo == 'k vecinos cercanos':
            covariables = escala.fit_transform(covariables)
            estimador = KNeighborsClassifier(n_neighbors= k_vecinos)
        elif modelo == 'arbol de decision':
            estimador = DecisionTreeClassifier()
        else:
            print("Modelo no identificado")
        cortes = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 12)
        f1_puntaje = make_scorer(f1_score, average='binary')
        puntajes = cross_val_score(estimador, covariables, variable_predecir, cv = cortes, scoring = f1_puntaje) 
        puntjaes = [round(num, 3) for num in cv_scores]
        print('Se obtienen los siguientes coeficientes de determinación:')
        print(cv_scores, '\n')
        print('El promedio es: {:.3f}'.format(np.mean(cv_scores)))

        
        
        
        
        
        
        
        
        