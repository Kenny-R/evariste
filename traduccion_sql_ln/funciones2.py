import sys
sys.path.append("..")
import pandas as pd
from .funciones import traducir_proyecciones, obtener_literal
from sqlglot import Expression
from sqlglot.expressions import Column
from parser_SQL.parser_SQL_clases import *

# Funciones nuevas
def traducir_miniconsulta_scan(consulta: miniconsulta_sql) -> str:
    proyeccion: list[str] = traducir_proyecciones(consulta.proyecciones)
    tabla: str = consulta.tabla
    if tabla[-1] == 'y':
        tabla = tabla[:-1]
        tabla += "ies"
    else:
        tabla += "s"

    return "Give me the " + ", ".join(proyeccion) + " of the " + tabla

def obtener_columna(condicion: Expression) -> tuple[str, bool]:
    columna_izq: Union[str, None] = str(condicion.args.get('this').args.get('this')) if isinstance(condicion.args.get('this'),  Column) and condicion.args.get('this').args.get('table') is not None else None
    columna_der: Union[str, None] = str(condicion.args.get('expression').args.get('this')) if isinstance(condicion.args.get('expression'),  Column) and condicion.args.get('expression').args.get('table') is not None else None
    reverso: bool = columna_izq is None
    
    return (columna_izq if not reverso else columna_der), reverso, columna_der

def filtrar_condicion(condicion: Expression, tupla) -> str:
    columna, reverso, literal = obtener_columna(condicion)
    match condicion.key:
        case "eq":
            return f"Has {tupla} {obtener_literal(condicion, literal)} {columna}?" 
        case "is":
            return f"Has {tupla} their {columna} equal to {obtener_literal(condicion, literal)}?"
        case "neq":
            return f"Has {tupla} their {columna} not equal to {obtener_literal(condicion, literal)}?"
        case "not":
            return f"Has {tupla} their {columna} not equal to {obtener_literal(condicion, literal)}?"
        case "gt":
            return f"Has {tupla} more than {obtener_literal(condicion, literal)} {columna}?" if not reverso else f"Has {tupla} less than {obtener_literal(condicion, literal)} {columna}?"
        case "gte":
            return f"Has {tupla} more than or equal to {obtener_literal(condicion, literal)} {columna}?" if not reverso else f"Has {tupla} less than or equal to {obtener_literal(condicion, literal)} {columna}?"
        case "lt":
            return f"Has {tupla} less than {obtener_literal(condicion, literal)} {columna}?" if not reverso else f"Has {tupla} more than {obtener_literal(condicion, literal)} {columna}?"
        case "lte":
            return f"Has {tupla} less than or equal to {obtener_literal(condicion, literal)} {columna}?" if not reverso else f"Has {tupla} more than or queal to {obtener_literal(condicion, literal)} {columna}?"
        
def obtener_columna_join(condicion: Expression, tabla: str) -> str:
    columna_izq: Union[str, None] = str(condicion.args.get('this').args.get('this')) if tabla == str(condicion.args.get('this').args.get('table')) else None
    columna_der: Union[str, None] = str(condicion.args.get('expression').args.get('this')) if tabla == str(condicion.args.get('expression').args.get('table')) else None
    
    return (columna_izq if columna_izq is not None else columna_der)

def columnas_join(condicion: Expression, tabla: str, fila: str) -> tuple[str, str]:
    columna_join: str = obtener_columna_join(condicion, tabla)
    return f"Give me the {columna_join} which corresponds with te following data {fila}", columna_join


def traducir_miniconsulta_anidada_scan(consulta: miniconsulta_sql_anidadas) -> str:
    tabla: str = list(consulta.tablas_aliases.values())[0]
    proyeccion: list[str] = traducir_proyecciones(consulta.proyecciones[list(consulta.tablas_aliases.keys())[0]])
    return "Give me the " + ", ".join(proyeccion) + " of the " + tabla 


def  filtrar_anidamiento(subconsulta: dict, tupla: tuple, columna: str) -> str:
    match subconsulta.get('operacion'):
        case "eq":
            return f"Has {tupla} {subconsulta.get('subconsulta').resultado[0].tolist()} {columna}?" 
        case "neq":
            return f"Has {tupla} their {columna} not equal to {subconsulta.get('subconsulta').resultado[0].tolist()}?" 
        case "gt":
            return f"Has {tupla} more than {subconsulta.get('subconsulta').resultado[0].tolist()} {columna}?" 
        case "gte":
            return f"Has {tupla} more than or equal to {subconsulta.get('subconsulta').resultado[0].tolist()} {columna}?"
        case "lt":
            return f"Has {tupla} less than {subconsulta.get('subconsulta').resultado[0].tolist()} {columna}?" 
        case "lte":
            return f"Has {tupla} less than or equal to {subconsulta.get('subconsulta').resultado[0].tolist()} {columna}?" 
    return f"Has {tupla} their {columna} {subconsulta.get('operacion')} {subconsulta.get('subconsulta').resultado[0].tolist()}?"