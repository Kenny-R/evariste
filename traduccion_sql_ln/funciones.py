import sys
sys.path.append("..")

from sqlglot import Expression
from sqlglot.expressions import Paren, Column, Literal
from parser_SQL.parser_SQL_clases import *

def obtener_operador(condicion: Expression, reverso: bool) -> str:
    match condicion.key:
        case "eq":
            return " is equal to "
        case "neq":
            return " is not equal to "
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
    proyecciones_traducidas = []

    for proyeccion in consulta.proyecciones:
        proyecciones_traducidas.append(proyeccion.args.get('this').args.get('this'))
    
    return proyecciones_traducidas

def traducir_miniconsulta_sql(consulta: miniconsulta_sql):
    proyeccion: str = traducir_proyecciones(consulta)
    tabla: str = consulta.tabla
    condicion: str = procesar_condiciones(consulta.condiciones)

    return "Give me the " + proyeccion + " of the " + tabla + " where " + condicion