#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:26:39 2024

@author: marcoguardia
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cod.CargarDatos import CargarDatos

class Graficar(CargarDatos):
    
    def __init__ (self, base=None):
        CargarDatos.__init__(self, base)
        
        
        
    def matriz_correlacion(self):
        #Matriz de correlación
        base_correlacion=self.base.drop(['Time', 'Class'], axis=1)
        correlation_matrix = base_correlacion.corr()

        plt.figure(figsize=(10, 8))
        mapa=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

        mapa.xaxis.set_ticks_position('none')

        plt.show() 
    
    def tabla_frecuencia(self, variable):
        tabla_frecuencia = self.base['Class'].value_counts()
        return(tabla_frecuencia)
    
    
    def distribucion(self, variable, bindwidth=1000, color='skyblue', edgecolor='black', fill=False):
        #Distribución de Amount
        binwidth = 1000
        bins = np.arange(min(self.base[variable]), max(self.base[variable]) + binwidth, binwidth)

        plt.hist(self.base[variable], bins=bins, color=color, edgecolor=edgecolor)

        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de la variable {variable}')


        plt.show()
        
    
    def valores_na(self):
        #Verificación de valores NA
        valores_nulos_por_columna = self.base.isnull().sum()
        return(valores_nulos_por_columna)
        
    def comparacion_densidades(self, cant_variables, predecir1, predecir0, color1='red', color0='green', fill=False):
        
        
        #Lista de variables a graficar 
        variables = [self.base.columns[i] for i in range(cant_variables)]  

        # Configuración de la cuadrícula de gráficos
        num_vars = len(variables)
        cols = 6  
        rows = (num_vars + cols - 1) // cols  

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

        # Generar los gráficos de densidad
        for i, var in enumerate(variables):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            #Densidad filtrada por predecir1
            sns.kdeplot(self.base[self.base['Class'] == 1][var], label=f'{var} - {predecir1}', fill=fill, color=color1, ax=ax)
            
            #Densidad filtrada por predecir0
            sns.kdeplot(self.base[self.base['Class'] == 0][var], label=f'{var} - {predecir0}', fill=fill, color=color0, ax=ax)
    
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            ax.set_title(f'Densidad de {var}')
            ax.legend()

        # Eliminar ejes vacíos si hay un número impar de gráficos
        if num_vars % cols != 0:
            for j in range(num_vars, rows * cols):
                if j >= num_vars:
                    fig.delaxes(axes.flat[j])

        plt.tight_layout()
        plt.show()


    