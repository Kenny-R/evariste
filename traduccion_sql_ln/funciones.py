import sys
sys.path.append("..")
import pandas as pd

from sqlglot import Expression
from sqlglot.expressions import Paren, Column, Literal
from parser_SQL.parser_SQL_clases import *

def obtener_operador(condicion: Expression, reverso: bool) -> str:
    match condicion.key:
        case "eq":
            return " is "
        case "neq":
            return " is not "
        case "gt":
            return " is greater than " if not reverso else " is less than "
        case "gte":
            return " is greater than or equal to " if not reverso else " is less than or equal to "
        case "lt":
            return " is less than " if not reverso else " is greater than "
        case "lte":
            return " is less than or equal to " if not reverso else " is greater than or equal to "
        
def obtener_literal(condicion: Expression, columna) -> str:
    if isinstance(condicion.args.get('this'), Column) and isinstance(condicion.args.get('expression'), Column):
        return columna
    
    return str(condicion.args.get('expression')) if isinstance(condicion.args.get('expression'), Literal) else str(condicion.args.get('this'))

def procesar_condicion_aux(condicion: Expression) -> str:
    columna_izq: Union[str, None] = str(condicion.args.get('this').args.get('this')) if isinstance(condicion.args.get('this'),  Column) else None
    columna_der: Union[str, None] = str(condicion.args.get('expression').args.get('this')) if isinstance(condicion.args.get('expression'),  Column) else None
    reverso: bool = columna_izq is None
    return (columna_izq if not reverso else columna_der) + obtener_operador(condicion, reverso) + obtener_literal(condicion, columna_der)

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

def traducir_proyecciones(consulta: miniconsulta_sql):
    proyecciones_traducidas: set[int] = set()

    for proyeccion in consulta.proyecciones:
        proyecciones_traducidas.add(proyeccion.args.get('this').args.get('this'))
    
    return list(proyecciones_traducidas)  

def traducir_dataframe(df: pd.DataFrame, columna_dependiente: str, columna_independiente: str) -> str :
    columna_list: list = df[columna_independiente].tolist()
    columna_str: str = " whose " + columna_dependiente + " is on ("

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
                if dependencia.resultado.empty:
                    break

                condiciones_join_str += traducir_dataframe(dependencia.resultado, str(tabla_consulta.args.get('this')), str(tabla_join.args.get('this')))

    return condiciones_join_str
   
def traducir_miniconsulta_sql(consulta: miniconsulta_sql, tiene_dependencia: bool) -> str:
    proyecciones: list[str] = traducir_proyecciones(consulta)
    tabla: str = consulta.tabla
    condicion: str = procesar_condiciones(consulta.condiciones)

    proyecciones_juntas = ""

    if len(proyecciones) == 1:
        proyecciones_juntas = proyecciones[0]
    elif len(proyecciones) > 1:
        proyecciones_juntas = ", ".join(proyecciones[:-1]) + f" and {proyecciones[-1]}"

    return "Give me the " + proyecciones_juntas + " of the " + tabla + (" where " + condicion  if condicion != "" else "") \
        + (procesar_condiciones_join(consulta) if tiene_dependencia else "")

def obtener_columnas_condicion_aux(condicion: Expression) -> list[str]:
    if condicion is None:
        return []
    
    operations: list[str] = ['eq', 'neq', 'gt', 'gte', 'lt', 'lte']
    if condicion.key in operations:
        return ([str(condicion.args.get('this').args.get('this'))] if isinstance(condicion.args.get('this'),  Column) and condicion.args.get('this').args.get('table') is not None else []) + \
            ([str(condicion.args.get('expression').args.get('this'))] if isinstance(condicion.args.get('expression'),  Column) and condicion.args.get('expression').args.get('table') is not None else [])
    
    return obtener_columnas_condicion_aux(condicion.args.get('this')) + obtener_columnas_condicion_aux(condicion.args.get('expression'))

def obtener_columnas_condicion(consulta: miniconsulta_sql) -> list[str]:
    columnas: list[str] = []
    for cond in consulta.condiciones:
        columnas += obtener_columnas_condicion_aux(cond)
    return list(set(columnas))