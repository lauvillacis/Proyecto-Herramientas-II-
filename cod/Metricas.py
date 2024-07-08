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
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


class Metricas():
    def __init__(self, valores_reales, valores_predichos):
        self.__valores_reales = valores_reales
        self.__valores_predichos = valores_predichos
        
    def matriz_de_confusion(self):
        matriz_confusion = confusion_matrix(self.__valores_reales, self.__valores_predichos)
        return matriz_confusion
    
    def puntaje_de_exactitud(self):
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
        exactitud = accuracy_score(self.__valores_reales, self.__valores_predichos)
        return exactitud
    
    def puntaje_de_precision(self, etiquetas):
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
        precision = precision_score(self.__valores_reales, self.__valores_predichos, etiquetas, average = None)
        return precision
    
    def reporte_de_clasificacion(self, etiquetas):
        reporte = classification_report(self.__valores_reales, self.__valores_predichos, etiquetas)
        return reporte
        
        
        
        
        
        
        
        
        
        
        
        
        