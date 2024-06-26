import sys
sys.path.append("..")
import pandas as pd

from sqlglot import Expression
from sqlglot.expressions import Paren, Column, Literal
from parser_SQL.parser_SQL_clases import *

def obtener_operador(operacion: str, reverso: bool = False) -> str:
    match operacion:
        case "eq":
            return " is "
        case "is":
            return " is "
        case "in":
            return " is "
        case "neq":
            return " is not "
        case "not":
            return " is not "
        case "not in":
            return " is not "
        case "gt":
            return " is greater than " if not reverso else " is less than "
        case "gte":
            return " is greater than or equal to " if not reverso else " is less than or equal to "
        case "lt":
            return " is less than " if not reverso else " is greater than "
        case "lte":
            return " is less than or equal to " if not reverso else " is greater than or equal to "
        case "in":
            return " is not "
        
def obtener_literal(condicion: Expression, columna: Expression) -> str:
    if condicion.args.get('this').args.get('table') is not None and condicion.args.get('expression').args.get('table') is not None:
        return columna

    return str(condicion.args.get('expression')) if isinstance(condicion.args.get('expression'), Literal) or isinstance(condicion.args.get('expression'), Column) else str(condicion.args.get('this'))

def procesar_condicion_aux(condicion: Expression) -> str:
    columna_izq: Union[str, None] = str(condicion.args.get('this').args.get('this')) if isinstance(condicion.args.get('this'),  Column) and condicion.args.get('this').args.get('table') is not None else None
    columna_der: Union[str, None] = str(condicion.args.get('expression').args.get('this')) if isinstance(condicion.args.get('expression'),  Column) and condicion.args.get('expression').args.get('table') is not None else None
    reverso: bool = columna_izq is None
    
    return (columna_izq if not reverso else columna_der) + obtener_operador(condicion.key, reverso) + obtener_literal(condicion, columna_der)

def procesar_condiciones(condiciones: list[Expression]) -> str:
    condiciones_str: str = ""

    for i, condicion in enumerate(condiciones):
        if i != 0: 
            condiciones_str += " and "
        
        if not (isinstance(condicion, Paren)):
            condiciones_str += procesar_condicion_aux(condicion)
            continue

        parte_izq: Expression = condicion.args.get('this').args.get('this')
        parte_der: Expression = condicion.args.get('this').args.get('expression')
        condiciones_str += procesar_condicion_aux(parte_izq) + " or "
        condiciones_str += procesar_condicion_aux(parte_der)

    return condiciones_str

def traducir_proyecciones(proyecciones: list[Expression]):
    proyecciones_traducidas: set[int] = set()

    for proyeccion in proyecciones:
        proyecciones_traducidas.add(proyeccion.args.get('this').args.get('this'))
    
    return list(proyecciones_traducidas)  

def traducir_dataframe(df: pd.DataFrame, 
                       columna_dependiente: str, 
                       columna_independiente: str,
                       operacion: str = " is ") -> str :
    columna_list: list = []

    if columna_independiente in df.columns:
        columna_list = df[columna_independiente].tolist()
    
    columna_str: str = " where " + columna_dependiente + operacion + "in ("

    for i, item in enumerate(columna_list):
        if i != 0:
            columna_str += ", "
        columna_str += str(item)
    
    columna_str += ")"

    return columna_str

def procesar_condiciones_join(consulta: miniconsulta_sql) -> str:
    condiciones_join_str: str = ""
        
    for i, condicion in enumerate(consulta.condiciones_join):
        if i != 0: 
            condiciones_join_str += " and "

        tabla_consulta: Expression
        tabla_join: Expression 

        if str(condicion.args.get('this').args.get('table')) == consulta.alias:
            tabla_consulta = condicion.args.get('this')
            tabla_join = condicion.args.get('expression')
        else:
            tabla_consulta = condicion.args.get('expression')
            tabla_join = condicion.args.get('this')

        # Por que puede darse el caso de que hayan varias dependencias        
        for dependencia in consulta.dependencias:
            if dependencia.alias == str(tabla_join.args.get('table')):
                # print("AHHHHAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
                # print(dependencia.resultado)
                # print("AHHHHAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
                
                if dependencia.resultado.empty:
                    break

                condiciones_join_str += traducir_dataframe(dependencia.resultado, str(tabla_consulta.args.get('this')), str(tabla_join.args.get('this')))

    return condiciones_join_str
   
def traducir_miniconsulta_sql(consulta: miniconsulta_sql, tiene_dependencia: bool) -> str:
    proyeccion: list[str] = traducir_proyecciones(consulta.proyecciones)
    tabla: str = consulta.tabla
    condicion: str = procesar_condiciones(consulta.condiciones)
    return "Give me the " + ", ".join(proyeccion) + " of the " + tabla + (" where " + condicion  if condicion != "" else "") \
        + (procesar_condiciones_join(consulta) if tiene_dependencia else "")

def obtener_columnas_condicion_aux(condicion: Expression) -> list[str]:
    if condicion is None:
        return []
    
    operations: list[str] = ['eq', 'neq', 'gt', 'gte', 'lt', 'lte']
    if condicion.key in operations:
        return ([str(condicion.args.get('this').args.get('this'))] if isinstance(condicion.args.get('this'),  Column) and condicion.args.get('this').args.get('table') is not None else []) + \
            ([str(condicion.args.get('expression').args.get('this'))] if isinstance(condicion.args.get('expression'),  Column) and condicion.args.get('expression').args.get('table') is not None else [])
    
    return obtener_columnas_condicion_aux(condicion.args.get('this')) + obtener_columnas_condicion_aux(condicion.args.get('expression'))

def obtener_columnas_condicion(condiciones: list[Expression]) -> list[str]:
    columnas: list[str] = []
    for cond in condiciones:
        columnas += obtener_columnas_condicion_aux(cond)
    return list(set(columnas))

def procesar_anidamientos(lista_anidamientos: list[dict[str, str | Expression]]) -> str:
    anidamiento_str: str = ""

    for info_subconsulta in lista_anidamientos:
        proyeccion_sub: str 
        if len(info_subconsulta['subconsulta'].miniconsultas_dependientes) != 0:
            proyeccion_sub = str(info_subconsulta['subconsulta'].miniconsultas_dependientes[0].proyecciones[0])
        else:
            proyeccion_sub = str(info_subconsulta['subconsulta'].miniconsultas_independientes[0].proyecciones[0])
        
        print(info_subconsulta['subconsulta'].resultado)
        print(proyeccion_sub)
        anidamiento_str += traducir_dataframe(info_subconsulta['subconsulta'].resultado, 
                               str(info_subconsulta['columna']), 
                               proyeccion_sub, obtener_operador(info_subconsulta['operacion']))
        
    return anidamiento_str

def traducir_miniconsulta_sql_anidada(consulta: miniconsulta_sql_anidadas) -> str:
    tabla: str = list(consulta.tablas_aliases.values())[0]
    proyeccion: list[str] = traducir_proyecciones(consulta.proyecciones[list(consulta.tablas_aliases.keys())[0]])
    condicion: str = procesar_condiciones(consulta.condiciones)
    return "Give me the " + ", ".join(proyeccion) + " of the " + tabla + (" where " + condicion  if condicion != "" else "") \
        + (procesar_anidamientos(consulta.subconsultas) if consulta.subconsultas is not None else "")