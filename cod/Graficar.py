#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:26:39 2024

@author: marcoguardia
"""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from cod.CargarDatos import CargarDatos

class Graficar(CargarDatos):
    
    def __init__ (self, base=None):
        '''
        Es el metodo constructor de los objetos de la clase
        
        Parametros
        -------
        base : data frame
            Los datos que se quieren utilizar para graficar

        Returns
        -------
        No devuelve nada
        '''
        CargarDatos.__init__(self, base)
        
        
        
    def matriz_correlacion(self, variable_predecir, filtrar=False):
        """
        Parameters
        ----------
        variable_predecir : string
            variable a predecir de la base de datos
        filtrar : bool, optional
            Si es true, se separa la variable a predecir en sus 
            dos valores 0,1 para calcular la matriz de correlación para cada uno.

        Returns
        -------
        None.

        """
        if(filtrar):
            correlation_matrix_0 = (self.base[self.base[variable_predecir]==0]).drop([variable_predecir], axis=1).corr()
            correlation_matrix_1 = (self.base[self.base[variable_predecir]==1]).drop([variable_predecir], axis=1).corr()
            
            plt.figure(figsize=(10, 8))
            mapa=sns.heatmap(correlation_matrix_0, annot=False, cmap='coolwarm', linewidths=0.5)
        
            plt.xticks(rotation=45, ha='right', fontsize=8)  # Reducir el tamaño de la fuente
            plt.yticks(fontsize=8)
            plt.title('Correlación en transacciones reales')

            # Mover las etiquetas del eje X a la parte inferior
            mapa.xaxis.tick_bottom()

            # Desactivar las etiquetas de ticks en la parte superior
            mapa.xaxis.set_ticks_position('bottom')
            mapa.xaxis.set_label_position('bottom')

            # Asegurarse de que las etiquetas del eje Y no se repitan
            mapa.yaxis.set_ticks_position('left')

            plt.show()
            
            plt.figure(figsize=(10, 8))
            mapa=sns.heatmap(correlation_matrix_1, annot=False, cmap='coolwarm', linewidths=0.5)
        
            plt.xticks(rotation=45, ha='right', fontsize=8)  # Reducir el tamaño de la fuente
            plt.yticks(fontsize=8)
            plt.title('Correlación en transacciones fraudulentas')

            # Mover las etiquetas del eje X a la parte inferior
            mapa.xaxis.tick_bottom()

            # Desactivar las etiquetas de ticks en la parte superior
            mapa.xaxis.set_ticks_position('bottom')
            mapa.xaxis.set_label_position('bottom')

            # Asegurarse de que las etiquetas del eje Y no se repitan
            mapa.yaxis.set_ticks_position('left')

            plt.show()
            
            
        else:   
            #Matriz de correlación
            correlation_matrix = self.base.drop([variable_predecir], axis=1).corr()

            plt.figure(figsize=(10, 8))
            mapa=sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        
            plt.xticks(rotation=45, ha='right', fontsize=8)  # Reducir el tamaño de la fuente
            plt.yticks(fontsize=8)

            # Mover las etiquetas del eje X a la parte inferior
            mapa.xaxis.tick_bottom()

            # Desactivar las etiquetas de ticks en la parte superior
            mapa.xaxis.set_ticks_position('bottom')
            mapa.xaxis.set_label_position('bottom')

            # Asegurarse de que las etiquetas del eje Y no se repitan
            mapa.yaxis.set_ticks_position('left')

            plt.show()
        
    def tabla_frecuencia(self, variable):
        """
        Método para generar una tabla de frecuencias de una variable de la base
        Parameters
        ----------
        variable : String
            Variable para hacer la tabla de frecuencia 

        Returns
        Series: frecuencia por clase dada una variable.
        None.

        """
        tabla_frecuencia = self.base[variable].value_counts()
        return(tabla_frecuencia)
    
    
    def distribucion(self, variable, binwidth=1000, color='skyblue', edgecolor='black', fill=False):
        """
        Método que genera la distribución empírica de una variable mediante un histograma.
    
        Parameters
        ----------
        variable : string
            variable para hacer la distribución
        binwidth : int, optional
            Inicializado en 1000 por default
        color : string, optional
            Color del histograma. Por default es 'skyblue'.
        edgecolor : string, optional
            Borde del histograma de distribución. Por default es 'black'.
        fill : bool, optional
             Se inicializa en false por default.

        Returns
        -------
        None.

        """
        #Distribución de Amount
        bins = np.arange(min(self.base[variable]), max(self.base[variable]) + binwidth, binwidth)

        plt.hist(self.base[variable], bins=bins, color=color, edgecolor=edgecolor)

        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de la variable {variable}')


        plt.show()
        
    
    def valores_na(self):
        """
        Método para obtener la cantidad de valores nulos por columna.

        Returns
        -------
        Series: Devuelve los valores nulos por columna

        """
        #Verificación de valores NA
        valores_nulos_por_columna = self.base.isnull().sum()
        return(valores_nulos_por_columna)
        
    def comparacion_densidades(self, cant_variables, variable_predecir, predecir1, predecir0, color1='red', color0='green', fill=False):
        """
        Método para realizar un gráfico de densidad para todas las covariables,
        asumiendo que la variable a predecir toma valores 0,1.
        Compara las densidades entre observaciones de valor 1 en la variable a 
        predecir y observaciones de valor 0 en la variable a predecir.

        Parameters
        ----------
        cant_variables : int
            cantidad de covariables de la base.
        variable_predecir : string
            variable a predecir
        predecir1 : string
            Clase 1 de la variable a predecir.
        predecir0 : string
            Clase 0 de la variable a predecir.
        color1 : string, optional
            Color de clase 1. El default es 'red'.
        color0 : string, optional
            Color de clase 0. El default es 'green'.
        fill : bool, optional
             El default es False.

        Returns
        -------
        None.

        """
        
        
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
            sns.kdeplot(self.base[self.base[variable_predecir] == 1][var], label=f'{var} - {predecir1}', fill=fill, color=color1, ax=ax)
            
            #Densidad filtrada por predecir0
            sns.kdeplot(self.base[self.base[variable_predecir] == 0][var], label=f'{var} - {predecir0}', fill=fill, color=color0, ax=ax)
    
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


    