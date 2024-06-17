from __future__ import annotations

import re
import json
import asyncio
import pandas as pd
from sqlglot import Expression
from typing import Optional, Union
from sqlglot.expressions import In, Binary, Not, Subquery

configuraciones = json.load(open("./configuraciones.json"))

STATUS = configuraciones['miniconsultas_status']
OPERACIONES_CONJUNTOS = configuraciones['miniconsultas_operaciones']
FUNCIONES_AGREGACION = configuraciones['miniconsultas_funciones_agregacion']

class miniconsulta_sql:
    """
        Clase que controla la ejecucion de una consulta 'simple' de SQL usando un LLM.

        Tenga en cuenta que estas consultas solo trabajan sobre una unica tabla. Si
        En la condiciones de la consulta se hace referencia a otra tabla quiere decir
        que esta consulta depende de otra, y por lo tanto el atributo dependencia 
        debe ser distinto de None

        atributos
        ------------

        tabla: Un string que almacena el nombre original de la tabla SQL.

        alias: Un string que almacena el alias relacionado a la tabla SQL.

        proyecciones: Una lista de expresiones las cuales son todas las proyecciones
                      del select de la consulta SQL.

        condiciones: Una lista de expresiones las cuales son todas las condiciones
                     del where de la consulta SQL.
                     
        condiciones_join: Una lista de expresiones las cuales son condiciones que 
                          estaban en algun ON de un JOIN en la consulta original 
                          SQL, estas condiciones deben ir en el where de esta
                          miniconsulta, e indica la forma en la que se debe
                          juntar el resultado de esta consulta con otra.
        
        status: Un string que indica el estado de ejecicion de la peticion a SQL.

        dependencia: Una miniconsulta de la cual depende esta miniconsulta.

        metodos
        ----------
        crear_prompt: Funcion que usando los datos disponibles crea una 
                      version en lenguaje natural de la peticion SQL
        
        _crear_representacion_SQL: Usando los datos disponibles
                                   devuelve un string con la miniconsulta
                                   en sintaxis SQL de Postgres
                                   
    """
    # Toda la información necesaria para construir la 
    # consulta SQL
    tabla:str
    alias: str
    proyecciones: list[Expression]
    condiciones: list[Expression]
    condiciones_join: Optional[list[Expression]]
    
    # Status disponibles: En espera, Ejecutando, Finalizado
    status: str
    dependencias: Optional[list[miniconsulta_sql]]

    # Resultado final

    resultado: pd.DataFrame

    def __init__(self, 
                tabla: str, 
                proyecciones: list[Expression],
                condiciones: list[Expression],
                alias:str = '',
                condiciones_join: Optional[list[Expression]] = None,
                dependencias: Optional[list[miniconsulta_sql]] = []):
        
        self.tabla = tabla
        self.alias = alias
        self.proyecciones = proyecciones
        self.condiciones = condiciones
        self.condiciones_join = condiciones_join
        self.dependencias = dependencias
        self.status = STATUS[0]
        self.resultado = pd.DataFrame()

    def crear_prompt(self):
        import traduccion_sql_ln

        traduccion = traduccion_sql_ln.traducir_miniconsulta_sql(self, self.dependencias is not None)
        proyecciones = traduccion_sql_ln.traducir_proyecciones(self)
        lista_columnas_condiciones = traduccion_sql_ln.obtener_columnas_condicion(self)

        return (traduccion, proyecciones, lista_columnas_condiciones)

    async def ejecutar(self):
        from ejecutar_LLM import hacer_consulta
        
        async def procesar(consulta_procesar: miniconsulta_sql):
            if consulta_procesar.status == STATUS[0]:
                traduccion, proyecciones, lista_columnas_condiciones = consulta_procesar.crear_prompt()
                consulta_procesar.status = STATUS[1]
                columnas = proyecciones + [columna for columna in lista_columnas_condiciones if columna not in proyecciones]
                consulta_procesar.resultado = await hacer_consulta(traduccion, columnas)
                consulta_procesar.status = STATUS[2]

            elif consulta_procesar.status == STATUS[1]:
                while consulta_procesar.status != STATUS[2]:
                    asyncio.sleep(5) 

        if self.dependencias != None:
            tareas = [procesar(dep) for dep in self.dependencias]
            await asyncio.gather(*tareas)
        
        await procesar(self)

    def imprimir_datos(self, nivel:int) -> str:
        lista_dependencias_str = "["
        if self.dependencias != []:
            for dependencia in self.dependencias:
                lista_dependencias_str+= dependencia.imprimir_datos(nivel+2) + "\n"
        
        lista_dependencias_str += "]"
        return f"""
{nivel*'    '}MINI CONSULTA
{(nivel+1)*'    '}tabla: {self.tabla}
{(nivel+1)*'    '}alias: {self.alias}
{(nivel+1)*'    '}proyecciones: {[i.sql() for i in self.proyecciones]}
{(nivel+1)*'    '}condiciones: {[i.sql() for i in self.condiciones]}
{(nivel+1)*'    '}condiciones_join: {[i.sql() for i in self.condiciones_join]}
{(nivel+1)*'    '}dependencias: {lista_dependencias_str}
{(nivel+1)*'    '}status: {self.status}
{(nivel+1)*'    '}"""

    def __str__(self) -> str:
        return self.imprimir_datos(0)
    
    def __repr__(self) -> str:
        return self.imprimir_datos(0)
    
class join_miniconsultas_sql:
    """
        Clase que controla todos los datos necesarios para realizar
        una consulta que originalmente era un join, utilizando
        miniconsultas de complejidad menor

        atributos
        ------------        
        miniconsultas_dependientes: Lista con todas las miniconsultas
                                    que dependen del resultado de otra
                                    para ser ejecutada con exito 

        miniconsultas_independientes: Lista con todas las miniconsultas
                                      que no dependen del resultado
                                      de ninguna otra para ser ejecutada
                                      con exito

        condiciones_having_or: Lista con todas las disyunciones que existen
                               en el HAVING del JOIN

        lista_agregaciones: Lista con todos las funciones de agregación que 
                            esta en el SELECT del JOIN
        
        lista_group_by: Lista con todas las columnas del GROUP BY

        lista_order_by: Lista con todas las columnas del ORDER BY

        condiciones_join: Una lista con las distintas condiciones
                          utilizadas en los joins de la consulta
                          SQL original
        
        condiciones_or: Lista con todas las disyunciones que existen
                        en el WHERE del JOIN
        
        condiciones_having: Las condiciones del HAVING que no son disyunciones
        
        resultado: El resultado de la ejecucion de este join

        limite: Un entero que indica si el JOIN tiene un LIMIT o no (Si tiene
                un -1 quiere decir que no tienen LIMIT)

        metodos
        -------------
        ejecutar: Funcion que realizara todos los joins utilizando las
                  miniconsultas disponibles
    """
    miniconsultas_independientes: list[miniconsulta_sql]
    miniconsultas_dependientes: list[miniconsulta_sql]
    condiciones_having_or: list[Expression]
    lista_agregaciones: list[Expression]
    condiciones_having: list[Expression]
    lista_group_by: list[dict[str,str]]
    lista_order_by: list[dict[str,str]]
    condiciones_join: list[Expression]
    condiciones_or: list[Expression]
    resultado: pd.DataFrame
    limite: int

    def __init__(self, 
                 condiciones_join: list[Expression],
                 miniconsultas_dependientes: list[miniconsulta_sql],
                 miniconsultas_independientes: list[miniconsulta_sql],
                 lista_group_by: list[dict[str,str]] = [],
                 lista_order_by: list[dict[str,str]] = [],
                 limite: int = -1,
                 condiciones_or: list[Expression] = [],
                 condiciones_having_or: list[Expression] = [],
                 condiciones_having: list[Expression] = [],
                 lista_agregaciones: list[Expression] = []
                 ):
        
        self.condiciones_join = condiciones_join
        self.miniconsultas_dependientes = miniconsultas_dependientes
        self.miniconsultas_independientes = miniconsultas_independientes
        self.lista_group_by = lista_group_by
        self.lista_order_by = lista_order_by
        self.limite = limite
        self.condiciones_or = condiciones_or
        self.condiciones_having_or = condiciones_having_or
        self.condiciones_having = condiciones_having
        self.lista_agregaciones = lista_agregaciones
        self.resultado = pd.DataFrame()

    
    def _juntar_resultados(self):

        # juntamos los resultados
        resultado_final = pd.DataFrame()

        # Construimos un diccionario para buscar rapidamente las miniconsultas
        miniconsultas = {mc.alias : mc for mc in self.miniconsultas_independientes + self.miniconsultas_dependientes}
        hay_match = True

        # tomamos el resultado de una de las consultas independientes para tener una base con la que empezar 
        # a hacer merge. OJO: Esta parte podria no estar bien tienes que revisar eso         
        consulta_base = self.miniconsultas_independientes[0]

        resultado_final = consulta_base.resultado.copy()
        resultado_final.columns = [f"{consulta_base.alias}.{columna.strip()}" for columna in resultado_final.columns]

        tablas_procesadas = [consulta_base.alias]
        condiciones_por_procesar = type(self.condiciones_join)(self.condiciones_join
                                                               )
        while len(condiciones_por_procesar) != 0:
            if hay_match == False:
                resultado_final = pd.DataFrame()
                break

            # Buscamos la siguiente condicion a procesar, Esta sera el que tenga alguna
            # tabla procesada en su condicion
            condicion_a_eliminar = -1
            for i, condicion in enumerate(condiciones_por_procesar):
                tabla1 = condicion.args.get('this').args.get('table').this
                tabla2 = condicion.args.get('expression').args.get('table').this

                if tabla1 in tablas_procesadas or tabla2 in tablas_procesadas:
                    if tabla1 not in tablas_procesadas:
                        resultados_agregar = miniconsultas[tabla1].resultado
                        columna_en_resultado_final = condicion.args.get('expression')
                        columna_en_resultados_agregar = condicion.args.get('this')
                    else:
                        resultados_agregar = miniconsultas[tabla2].resultado
                        columna_en_resultado_final = condicion.args.get('this')
                        columna_en_resultados_agregar = condicion.args.get('expression')
                    
                    condicion_a_eliminar = i
                    break
            
            if tabla1 in tablas_procesadas and tabla2 in tablas_procesadas:
                raise Exception("Es posible que se vuelvan a procesar las mismas tablas??")

            if condicion_a_eliminar == -1:
                raise Exception("Algo raro paso buscando la condicion para hacer merge")
            else:
                condiciones_por_procesar.pop(condicion_a_eliminar)

            # Vamos a hacer merge de la tabla 1 a la tabla 2

            # Revisamos si existe la columna objetivo en la tabla 1 y la tabla 2
            # o si alguno de las tablas resultados estan vacios

            if (resultado_final.empty or 
                resultados_agregar.empty or 
                str(columna_en_resultado_final).strip() not in resultado_final.columns or
                str(columna_en_resultados_agregar.this).strip() not in resultados_agregar.columns):
                
                # print(f'resultado_final.empty: {resultado_final.empty}')
                # print(f'resultados_agregar.empty: {resultados_agregar.empty}')
                # print(f'str(columna_en_resultado_final).strip(): {str(columna_en_resultado_final).strip()}')
                # print(f'resultado_final.columns: {resultado_final.columns}')
                # print(f'str(columna_en_resultado_final).strip() not in resultado_final.columns: {str(columna_en_resultado_final).strip() not in resultado_final.columns}')
                # print(f'str(columna_en_resultados_agregar.this).strip(): {str(columna_en_resultados_agregar.this)}')
                # print(f'resultados_agregar.column: {resultados_agregar.columns}')
                # print(f'str(columna_en_resultados_agregar.this).strip() not in resultados_agregar.columns): {str(columna_en_resultados_agregar.this) not in resultados_agregar.columns}')
                
                hay_match = False
                continue
            
            columnas_anteriores = list(resultado_final.columns)
            nuevas_columnas = [f"{columna_en_resultados_agregar.table}.{columna.strip()}" for columna in resultados_agregar.columns]
            resultado_final = resultado_final.merge(resultados_agregar, right_on=str(columna_en_resultados_agregar.this).strip(), left_on=str(columna_en_resultado_final).strip())
            resultado_final.columns = columnas_anteriores + nuevas_columnas       
            tablas_procesadas.append(str(columna_en_resultados_agregar.table))
        self.resultado = resultado_final

    def _ordenar_resultados(self):
        columnas = []
        ascendente = []

        for datos_order_by in self.lista_order_by:
            tabla, columna, tipo = datos_order_by.values()
            columnas.append(f"{tabla}.{columna}".strip())
            ascendente.append(tipo == "ASC")

        self.resultado.sort_values(by = columnas, ascending= ascendente, inplace=True)

    def _transformar_abreviaciones(self, cantidad:str):
        exponente = 0
        cantidad_procesada = cantidad
        
        unidades = {"thousand": 3,"million": 6, "billion": 9, "trillion": 12}
        abreviaciones = {"K":3, "M":6, "B":9, "T":12}

        for unidad in unidades.keys():
            if re.findall(unidad,cantidad_procesada, re.I) != []:
                exponente += unidades[unidad] * len(re.findall(unidad,cantidad_procesada))

                cantidad_procesada = re.sub(unidad, "",cantidad_procesada,flags=re.I)
        
        for abreviacion in abreviaciones.keys():
            if re.findall(unidad,cantidad_procesada, re.I) != []:
                exponente += abreviaciones[abreviacion] * len(re.findall(abreviacion,cantidad_procesada))

                cantidad_procesada = re.sub(abreviacion, "",cantidad_procesada,flags=re.I)
        
        return (exponente, cantidad_procesada)

    def _procesar_cantidades(self, cantidad: str):
        # procesamos las unidades (Millones, Billones, Miles, M, B, K)
        exponente,cantidad_procesada = self._transformar_abreviaciones(cantidad)

        # Quitamos todo lo que no sea un numero o un punto
        cantidad_procesada = re.sub("[^\.1-9]","", cantidad_procesada)

        #pasamos a flotante
        try:
            cantidad_final= float(cantidad_procesada)
        except:
            raise Exception("No se pudo transformar el numero a flotante")
        
        cantidad_final *= 10**(exponente)

        return cantidad_final

    def _hacer_agregaciones(self):
        resultado = pd.DataFrame()
        for agregacion in self.lista_agregaciones:
            columna = agregacion.this.sql()
            
            if columna not in self.resultado.columns:
                continue
            
            datos = self.resultado[columna]

            if agregacion.key == "count":
                columnas_resultado = len(resultado.columns)
                resultado.insert(columnas_resultado, agregacion.sql(), [len(datos)])

                continue
            
            datos = datos.apply(self._procesar_cantidades)
            
            if agregacion.key == "min":
                columnas_resultado = len(resultado.columns)
                resultado.insert(columnas_resultado, agregacion.sql(), datos.min())

            elif agregacion.key == "max":
                columnas_resultado = len(resultado.columns)
                resultado.insert(columnas_resultado, agregacion.sql(), datos.max()) 
            
            elif agregacion.key == "avg":
                columnas_resultado = len(resultado.columns)
                resultado.insert(columnas_resultado, agregacion.sql(), datos.mean())
            
            elif agregacion.key == "sum":
                columnas_resultado = len(resultado.columns)
                resultado.insert(columnas_resultado, agregacion.sql(), datos.sum())
            
            
        
        self.resultado = resultado


    def ejecutar(self):
        """
            Aqui es donde se hara el o los joins usando los resultados
            de las miniconsultas.

            Ten en cuenta que este ejecutar debe ser llamado despues de 
            haber realizado todas las peticiones al LLM y las miniconsultas
            deben haber sido ejecutadas antes de ejecutar esta funcion

            Aqui probablemente le pasemos distintas estrategias para ejecutar un join
            por lo que ten presente que seguramente debamos pasarle de alguna forma
            la manera en la que vamos a ejecutar
        """
        # Ejecutamos las miniconsultas independientes y luego las dependientes
        for mc in self.miniconsultas_independientes:
            asyncio.run(mc.ejecutar())

        for mc in self.miniconsultas_dependientes:
            asyncio.run(mc.ejecutar())

        
        self._juntar_resultados()

        if self.limite > 0:
            self.resultado = self.resultado.iloc[:self.limite]
        
        self._ordenar_resultados()
      
        self._hacer_agregaciones()

    def imprimir_datos(self, nivel: int) -> str:
        return f"""
{nivel*'    '}CONSULTA JOIN
{(nivel + 1)*'    '}miniconsultas_independientes: {self.miniconsultas_independientes}
{(nivel + 1)*'    '}miniconsultas_dependientes: {self.miniconsultas_dependientes}
{(nivel + 1)*'    '}condiciones_having_or: {[i.sql() for i in self.condiciones_having_or]}
{(nivel + 1)*'    '}condiciones_having: {[i.sql() for i in self.condiciones_having]}
{(nivel + 1)*'    '}lista_group_by: {self.lista_group_by}
{(nivel + 1)*'    '}lista_order_by: {self.lista_order_by}
{(nivel + 1)*'    '}condiciones_join: {[i.sql() for i in self.condiciones_join]}
{(nivel + 1)*'    '}condiciones_or: {[i.sql() for i in self.condiciones_or]},
{(nivel + 1)*'    '}lista_agregaciones: {[i.sql() for i in self.lista_agregaciones]}
{(nivel + 1)*'    '}limite: {self.limite}
              """

    def __str__(self) -> str:
        return self.imprimir_datos(0)
    
class operacion_miniconsultas_sql:
    """
        Clase que controla todos los datos necesarios para realizar
        una consulta que originalmente era una operacion de conjuntos,
        utilizando miniconsultas de complejidad menor

        atributos
        ------------
        operacion: Un string con el tipo de operacion que se quiere realizar

        parte_derecha: El ejecutor necesario para procesar la consulta que esta
                       del lado derecho de la operacion
        
        parte_izquierda: El ejecutor necesario para procesar la consulta que esta
                         del lado izquierdo de la operacion

        resultado: El resultado de la ejecucion de esta operación

        metodos
        -------------
        ejecutar: Funcion que realiza la operacion indicada en los datos
    """
    operacion: str
    parte_derecha: Union[miniconsulta_sql, join_miniconsultas_sql]
    parte_izquierda: Union[miniconsulta_sql, join_miniconsultas_sql, operacion_miniconsultas_sql]
    restultado: str

    def __init__(self, 
                 operacion:str, 
                 parte_derecha: Union[miniconsulta_sql, join_miniconsultas_sql], 
                 parte_izquierda:Union[miniconsulta_sql, join_miniconsultas_sql, operacion_miniconsultas_sql]):
        
        self.operacion = operacion
        self.parte_derecha = parte_derecha
        self.parte_izquierda = parte_izquierda
    
    def ejecutar(self):
        """
            Aqui es donde se hara la operacion usando los resultados
            de las miniconsultas.

            Ten en cuenta que este ejecutar debe ser llamado despues de 
            haber realizado todas las peticiones al LLM y las miniconsultas
            deben haber sido ejecutadas antes de ejecutar esta funcion
        """
        raise Exception("Por implementar!!!")


    def imprimir_datos(self, nivel: int) -> str:
        return f"""
{nivel*'    '}CONSULTA OPERACION
{(nivel + 1)*'    '}operacion: {self.operacion}
{(nivel + 1)*'    '}parte_derecha: 
{self.parte_derecha.imprimir_datos(nivel+2)}
{(nivel + 1)*'    '}parte_izquierda: 
{self.parte_izquierda.imprimir_datos(nivel+2)}
              """
    def __str__(self) -> str:
          return self.imprimir_datos(0)

class miniconsulta_sql_anidadas:
    """
        Clase que controla todos los datos necesarios para realizar
        una consulta que originalmente era una consulta anidada, utilizando
        miniconsultas de complejidad menor

        atributos
        ------------        
        proyecciones: Lista con todas las proyecciones en el SELECT

        agregaciones: Lista con todas las agregaciones en el SELECT

        aliases: Lista con todos los alias o nombre de Tablas utilizadas
                 en la consulta
        
        tablas_aliases: Diccionario con el nombre de los aliases utilizados
        
        condiciones_having_or: Lista con todas las disyunciones que existen
                               en el HAVING del JOIN

        lista_agregaciones: Lista con todos las funciones de agregación que 
                            esta en el SELECT del JOIN
        
        lista_group_by: Lista con todas las columnas del GROUP BY

        lista_order_by: Lista con todas las columnas del ORDER BY

        condiciones_join: Una lista con las distintas condiciones
                          utilizadas en los joins de la consulta
                          SQL original
        
        condiciones_or: Lista con todas las disyunciones que existen
                        en el WHERE del JOIN
        
        condiciones_having: Las condiciones del HAVING que no son disyunciones
        
        resultado: El resultado de la ejecucion de este join

        limite: Un entero que indica si el JOIN tiene un LIMIT o no (Si tiene
                un -1 quiere decir que no tienen LIMIT)
        
        subconsultas: Una lista de diccionarios con toda la información sobre 
                      todas las condiciones anidadas dentro de la consulta
    """

    proyecciones: list[Expression]
    agregaciones: list[Expression]
    aliases: list[str]
    tablas_aliases: dict[str, str]
    condiciones_join: list[Expression]
    condiciones: list[Expression]
    condiciones_or: list[Expression]
    condiciones_having: list[Expression]
    condiciones_having_or: list[Expression]
    limite: int
    resultado: str
    lista_order_by: list[str]
    lista_group_by: list[dict[str,str]]
    subconsultas: list[dict[str, str | Expression]]

    def __init__(self, 
                 proyecciones: list[Expression],
                 agregaciones: list[Expression],
                 aliases: list[str],
                 tablas_aliases: dict[str, str],
                 condiciones: list[Expression],
                 condiciones_or: list[Expression],
                 condiciones_having: list[Expression],
                 condiciones_having_or: list[Expression],
                 condiciones_join: list[Expression] = [],
                 limite: int = -1,
                 lista_order_by: list[str] = [],
                 lista_group_by: list[dict[str,str]] = []) -> None:
        self.proyecciones = proyecciones
        self.agregaciones = agregaciones
        self.aliases = aliases
        self.tablas_aliases = tablas_aliases
        self.condiciones = condiciones
        self.condiciones_or = condiciones_or
        self.condiciones_having =  condiciones_having
        self.condiciones_having_or =  condiciones_having_or
        self.condiciones_join = condiciones_join
        self.limite = limite
        self.lista_order_by = lista_order_by
        self.lista_group_by = lista_group_by
        self.subconsultas, self.condiciones = self.__obtener_subconsultas(condiciones)
            
    def __construir_miniconsulta(self, 
                                 consulta_sql_ast: Expression) -> (miniconsulta_sql 
                                                                   | join_miniconsultas_sql):
        raise Exception("Por implementar!!!")
    
    def __obtener_subconsultas(self,
                               condiciones: list[Expression]) -> tuple[list[dict[str, str | Expression]], list[Expression]]:
    
        # Para evitar las importaciones circulares
        from .parser_SQL_funciones import obtener_ejecutor

        consultas: list[dict[str, str | Expression]] = []
        indices: list[int] = []
    
        for i, cond in enumerate(condiciones):
            if (isinstance(cond, Binary)): 
                if (isinstance(cond.args.get('this'), Subquery)): 
                    indices.append(i)
                    consultas.append({'operacion': cond.key,
                                'tabla': cond.args.get('expression').args.get('table').args.get('this'),
                                'columna': cond.args.get('expression').args.get('this').args.get('this'),
                                'subquery': obtener_ejecutor(cond.args.get('this').args.get('this').sql())})

                elif (isinstance(cond.args.get('expression'), Subquery)):
                    indices.append(i)
                    consultas.append({'operacion': cond.key,
                                'tabla': cond.args.get('this').args.get('table').args.get('this'),
                                'columna': cond.args.get('this').args.get('this').args.get('this'),
                                'subquery': obtener_ejecutor(cond.args.get('expression').args.get('this').sql())})
            
            elif (isinstance(cond, Not)):
                indices.append(i)
                consultas.append({'operacion': f'{cond.key} {cond.args.get("this").key}',
                                'tabla': cond.args.get('this').args.get('this').args.get('table').args.get('this'),
                                'columna': cond.args.get('this').args.get('this').args.get('this').args.get('this'),
                                'subquery': obtener_ejecutor(cond.args.get('this').args.get('query').args.get('this').sql())})
                
            elif (isinstance(cond, In)):
                indices.append(i)
                consultas.append({'operacion': cond.key,
                            'tabla': cond.args.get('this').args.get('table').args.get('this'),
                            'columna': cond.args.get('this').args.get('this').args.get('this'),
                            'subquery': obtener_ejecutor(cond.args.get('query').args.get('this').sql())})

        return (consultas, [condiciones[i] for i in range(len(condiciones)) if i not in indices])

    def imprimir_datos(self, nivel: int) -> str:
        proyecciones_imprimir = []
        for i in self.proyecciones.values():
            for j in i:
                proyecciones_imprimir.append(j.sql())
        return f"""
{nivel*'    '}CONSULTA ANIDADA
{(nivel + 1)*'    '}proyecciones: {proyecciones_imprimir}
{(nivel + 1)*'    '}agregaciones: {[i.sql() for i in self.agregaciones]}
{(nivel + 1)*'    '}aliases: {self.aliases}
{(nivel + 1)*'    '}tablas_aliases: {self.tablas_aliases}
{(nivel + 1)*'    '}condiciones_join: {[i.sql() for i in self.condiciones_join]}
{(nivel + 1)*'    '}condiciones: {[i.sql() for i in self.condiciones]}
{(nivel + 1)*'    '}condiciones_or: {[i.sql() for i in self.condiciones_or]}
{(nivel + 1)*'    '}condiciones_having: {self.condiciones_having}
{(nivel + 1)*'    '}condiciones_having_or: {[i.sql() for i in self.condiciones_having_or]}
{(nivel + 1)*'    '}limite: {self.limite}
{(nivel + 1)*'    '}lista_order_by: {self.lista_order_by}
{(nivel + 1)*'    '}lista_group_by: {self.lista_group_by}
{(nivel + 1)*'    '}subconsultas: {self.subconsultas}
{(nivel + 1)*'    '}"""

    def __str__(self) -> str:
        return self.imprimir_datos(0)