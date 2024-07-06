#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:11:56 2024

@author: lauravillacis
"""
import pandas as pd

class CargarDatos():
    
    def __init__(self, base):
        self.__base = base
        
    def cargar_base(self, ruta):
        self.__base = pd.read_excel(ruta)