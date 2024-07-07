#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:16:29 2024

@author: lauravillacis
"""
from CargarDatos import CargarDatos
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


class Metricas(CargarDatos):
    def __init__(self, base):
        CargarDatos.__init__(self, base)
        
    def matriz_de_confusion(self, valores_reales, valores_predichos):
        matriz_confusion = confusion_matrix(valores_reales, valores_predichos)
        return matriz_confusion
    
    def puntaje_de_exactitud(self, valores_reales, valores_predichos):
        '''
        Devuelve la medida  de la proporciooo2ón de predicciones correctas

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
    
    def puntaje_de_precision(self, valores_reales, valores_predichos, etiquetas):
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
        precision = precision_score(valores_reales, valores_predichos, etiquetas, average = None)
        return precision
    
    def reporte_de_clasificacion(self, valores_reales, valores_predichos, etiquetas):
        reporte = classification_report(valores_reales, valores_predichos, etiquetas)
        return reporte
        
        
        
        
        
        
        
        
        
        
        
        
        