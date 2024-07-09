#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:11:56 2024

@author: lauravillacis
"""
import pandas as pd

class CargarDatos():
    
    def __init__(self, base=None):
        """
        Parameters
        ----------
        base : pd.dataframe, 
        
        Returns
        -------
        None.
        """
        
        if(base is None):
            self.__base = pd.DataFrame()
        else:
            self.__base=base
        
    @property
    def base(self):
        '''
        Es el metodo get, para devolver el valor del atributo 'base'
        
        Parametros
        -------
        No lleva parametros explicitos pero se usa el objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        return self.__base
    
    
    #Set
    @base.setter
    def base(self, nueva_base):
        '''
        Es el metodo set, para establecer el valor del atributo 'base'
        
        Parametros
        -------
        nueva_base : dataframe
            Esto es el  dataframe que se quiere asignar/ para modificar 
            el atributo 'base' del objeto de la clase
        
        Returns
        -------
        No devuelve nada
        '''
        self.__base = nueva_base
        
    def view(self):
        if self.__base is not None:
            datos = self.__base.info()
            print(datos)
            return datos
        else:
            print("El objeto 'base' es None")
            return None
        
        
    def cargar_base(self, ruta):
        self.__base = pd.read_excel(ruta)