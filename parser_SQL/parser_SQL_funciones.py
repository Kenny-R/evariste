import json
from typing import Any
from .parser_SQL_clases import *
from sqlglot import Expression, column, parse_one
from sqlglot.expressions import In, Binary, Not, Subquery, Ordered

configuraciones = json.load(open("./configuraciones.json"))
OPERACIONES_CONJUNTOS = configuraciones['miniconsultas_operaciones']
FUNCIONES_AGREGACION = configuraciones['miniconsultas_funciones_agregacion']

def obtener_tablas(consulta_sql_ast: Expression) -> tuple[list[str], dict[str, str]]:
    """
        Dada un ast de una consulta SQL de postgres obtiene todas las tablas 
        de la consulta. Esta función tiene en cuenta el from y los joins. 
        Ademas tiene en cuenta los alias

        Parametros
        --------------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        --------------
            Una lista con el nombre de todas las tablas de la consulta.

            Un diccionario cuyos key son los alias de cada tabla y los 
            valores son el nombre original de la tabla
    """

    tablas: list[str] = []
    tablas_alias: dict[str, str] = {}

    if consulta_sql_ast.key != 'select':
        raise Exception('La consulta SQL necesita tener un "SELECT"')

    if consulta_sql_ast.args.get('from') == None:
        raise Exception('La consulta SQL necesita tener un "FORM"')

    # Obtenemos la tabla que esta en el from
    elementos_a_revisar = [consulta_sql_ast.args['from']]

    # Si tiene joins tenemos en cuenta esas tablas
    if consulta_sql_ast.args.get('joins') != None:
        elementos_a_revisar += consulta_sql_ast.args['joins']

    # Conseguimos los nombres originales de las tablas y sus alias
    # si es que tienen

    for elemento in elementos_a_revisar:
        if elemento.key == 'from' or elemento.key == 'join':

            nombre_tabla = elemento.this.this.this
            alias_tabla = elemento.this.alias

            if nombre_tabla not in tablas:
                tablas.append(nombre_tabla)

            if alias_tabla != '':
                tablas_alias[alias_tabla] = nombre_tabla

    return tablas, tablas_alias


def obtener_proyecciones_agregaciones(consulta_sql_ast: Expression,
                                      tablas: list[str],
                                      tablas_alias: dict[str, str]) -> tuple[dict[str, list[column]], list[Expression]]:
    """
        Dada un ast de una consulta SQL de postgres obtiene todas las proyecciones y 
        funciones de agregación que hay en el SELECT de la consulta.

        Tenga en cuenta que si la consulta tiene un * como unica proyeccion
        la funcion lanzara un error.

        Parametros
        --------------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        tabla: Una lista con el nombre de todas las tablas de la consulta

        tablas_alias: Un diccionario cuyos key son los alias de cada tabla y los 
            valores son el nombre original de la tabla 

        Retorna
        --------------
            Una tupla cuya primera componente es un diccionario cuyas key son la tabla 
            (o alias de tablas) con la que esta relacionado una o varias proyecciones en
            el select. Y los valores son una lista de columnas de la tabla. En la 
            segunda componente se tiene un a lista con las funciones de agregacion
            que estan en el select.
    """
    proyecciones: dict[str, list[column]] = {}
    agregaciones = []

    # Revisamos si la unica proyeccion es un *
    if (len(consulta_sql_ast.args['expressions']) == 1 and
            consulta_sql_ast.args['expressions'][0].key == 'star'):
        raise Exception('Debe haber al menos una proyección en el Select')

    for elemento_a_revisar in consulta_sql_ast.args['expressions']:

        if elemento_a_revisar.key in FUNCIONES_AGREGACION:
            parametro_de_agregacion = elemento_a_revisar.this

            if (parametro_de_agregacion.key == 'column' and
                parametro_de_agregacion.table not in tablas and
                    tablas_alias.get(parametro_de_agregacion.table) == None):
                raise Exception(
                    f'No existe la tabla o alias de tabla "{parametro_de_agregacion.table}"')

            agregaciones.append(elemento_a_revisar)

        elif elemento_a_revisar.key == 'column':
            if (elemento_a_revisar.table not in tablas and
                    tablas_alias.get(elemento_a_revisar.table) == None):
                raise Exception(
                    f'No existe la tabla o alias de tabla "{elemento_a_revisar.table}"')

            if proyecciones.get(elemento_a_revisar.table) == None:
                proyecciones[elemento_a_revisar.table] = []

            proyecciones[elemento_a_revisar.table].append(elemento_a_revisar)

    return proyecciones, agregaciones


def obtener_tablas_condiciones(condicion: Expression) -> tuple[str, str]:
    """
    Dada una condicion obtiene cuales son las tablas implicadas en esa condicion.
    Puede darse el caso que hayan 0, 1 o 2 tablas en una condicion.

    Parametros
    --------------
    condicion: Una expresion de sqlglot que representa una condicion

    Retorna
    --------------
        Una tupla con las tablas implicadas en la condición    
    """

    tabla_izquierda = ''
    tabla_derecha = ''

    # caso en el que estamos trabajando con condiciones binarias >, <, <=, >=, LIKE
    if isinstance(condicion, Binary):

        nodo_izquierdo = condicion.this

        if nodo_izquierdo.key == 'column':
            tabla_izquierda = nodo_izquierdo.table

        nodo_derecho = condicion.args['expression']

        if nodo_derecho.key == 'column':
            tabla_derecha = nodo_derecho.table

    # caso en el que estamos trabajando con un NOT IN
    elif isinstance(condicion, Not) and isinstance(condicion.this, In):

        nodo_izquierdo = condicion.this.this

        if nodo_izquierdo.key == 'column':
            tabla_izquierda = nodo_izquierdo.table

    elif isinstance(condicion, In):

        nodo_izquierdo = condicion.this

        if nodo_izquierdo.key == 'column':
            tabla_izquierda = nodo_izquierdo.table

    return tabla_izquierda, tabla_derecha


def obtener_condiciones(conector_inicial: Expression, tipo_conectores: str = 'and') -> list[Expression]:
    """
    Dado un conector de sqlglot (Casi siempre un AND) obtiene todas las 
    condiciones existentes en ese conector y todos los que esten contenidos
    en el.

    Parametros
    --------------
    conector_inicial: Un conector de sqlglot (Casi siempre un AND)

    tipo_conectores: Un string con el tipo de conectores que se espera
                     que tenga las condiciones. Es decir que se espera
                     que si hay mas de un conector se espera que todos
                     sean ands o or.

    Retorna
    --------------
    Una lista con todas condiciones dentro del contector  
    """
    conectores = [conector_inicial]
    condiciones = []

    # Pasamos recursivamente por todos los operadores del WHERE
    # Y obtenemos todas las condiciones
    while conectores != []:
        conector_actual = conectores.pop(0)

        # Caso base
        if conector_actual.key != tipo_conectores:
            condiciones.append(conector_actual)
            break

        # Revisamos la parte izquierda del and
        if conector_actual.this.key != tipo_conectores:
            condiciones.append(conector_actual.this)
        else:
            conectores.append(conector_actual.this)

        # Agregamos la parte derecha del and
        if conector_actual.args['expression'].key != tipo_conectores:
            condiciones.append(conector_actual.args['expression'])

    return condiciones


def obtener_condiciones_where(consulta_sql_ast: Expression) -> dict[list[Expression], dict[str, list[Expression]]]:
    """
        Dada un ast de una consulta SQL de postgres obtiene todas las condiciones
        del WHERE de la consulta

        Tenga en cuenta que esta funcion espera que en el WHERE solo hayan operadores
        AND.

        Si hay condiciones de OR deben estar entre parentesis, los conectores principales
        deben ser ANDs

        Parametros
        --------------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        --------------
            Una lista con todas las condiciones del WHERE
    """

    # Obtenemos todas las condiciones
    # Ten en cuenta que el and asocia a izquierda esta vez
    if consulta_sql_ast.args.get('where') == None:
        raise Exception(
            f'La consulta debe tener un WHERE. la consulta es: \n {consulta_sql_ast.sql()}')

    condiciones_totales = obtener_condiciones(
        consulta_sql_ast.args['where'].this)

    condiciones_or = []
    condiciones = []

    for i in range(len(condiciones_totales)):
        if condiciones_totales[i].key == 'or' or (condiciones_totales[i].key == 'paren' and condiciones_totales[i].this.key == 'or'):
            condiciones_or.append(condiciones_totales[i])
        else:
            condiciones.append(condiciones_totales[i])

    return {'condiciones': condiciones, 'condiciones or': condiciones_or}


def obtener_condiciones_having(consulta_sql_ast: Expression) -> dict[str, list[Expression]]:
    """
       Dada un ast de una consulta SQL de postgres obtiene todas las condiciones
       del HAVING de la consulta

       Tenga en cuenta que esta funcion espera que en el HAVING solo hayan operadores
       AND.

       Si hay condiciones de OR deben estar entre parentesis, los conectores principales
       deben ser ANDs

       Parametros
       --------------
       consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                         ast de una consulta SQL

       Retorna
       --------------
           Una lista con todas las condiciones del HAVING
   """

    # Obtenemos todas las condiciones
    # Ten en cuenta que el and asocia a izquierda esta vez
    if consulta_sql_ast.args.get('having') == None:
        raise Exception('La consulta debe tener un HAVING')

    condiciones = obtener_condiciones(consulta_sql_ast.args['having'].this)

    condiciones_or = []

    for i in range(len(condiciones)):
        if condiciones[i].key == 'or':
            condiciones_or.append(condiciones.pop(i))

    return {'condiciones': condiciones, 'condiciones or': condiciones_or}


def clasificar_condiciones_where(condiciones: list[Expression],
                                 tablas: list[str],
                                 tablas_alias: dict[str, str]) -> dict[str, list[Expression]]:
    """
        Toma una lista de condiciones, una lista de tablas y un diccionario con los
        alises de las tablas. Clasifica cada condicion dependiendo de la tabla 
        a la que haga referencia        

        Parametros
        --------------
        condiciones: Una lista de condiciones

        tabla: Una lista con el nombre de todas las tablas de la consulta

        tablas_alias: Un diccionario cuyos key son los alias de cada tabla y los 
            valores son el nombre original de la tabla 

        Retorna
        --------------
            Un diccionario cuyas key son la tabla (o alias de tablas) con la que 
            esta relacionado una o varias condicones en el WHERE. Y los valores
            son una lista de dichas condiciones.
    """

    # Si una condicion depende de dos tablas lo clasificaremos con la tabla de la
    # izquierda
    condiciones_por_tablas = {}
    for condicion in condiciones:
        tabla_izquierda, tabla_derecha = obtener_tablas_condiciones(condicion)

        if tabla_derecha == '' and tabla_izquierda == '':
            raise Exception(f'La condicion {condicion} no es valida')

        # verificamos que las tablas del lado izquierdo y derecho existan
        if (tabla_izquierda != '' and
            tabla_izquierda not in tablas and
                tablas_alias.get(tabla_izquierda) == None):
            raise Exception(
                f'No existe la tabla o alias de tabla "{tabla_izquierda}"')

        if (tabla_derecha != '' and
            tabla_derecha not in tablas and
                tablas_alias.get(tabla_derecha) == None):
            raise Exception(
                f'No existe la tabla o alias de tabla "{tabla_derecha}"')

        if tabla_izquierda != '':
            if condiciones_por_tablas.get(tabla_izquierda) == None:
                condiciones_por_tablas[tabla_izquierda] = []

            condiciones_por_tablas[tabla_izquierda].append(condicion)
            continue

        if tabla_derecha != '':
            if condiciones_por_tablas.get(tabla_derecha) == None:
                condiciones_por_tablas[tabla_derecha] = []

            condiciones_por_tablas[tabla_derecha].append(condicion)
            continue

    return condiciones_por_tablas


def obtener_condiciones_joins(consulta_sql_ast: Expression,
                              tablas: list[str],
                              tablas_alias: dict[str, str],
                              condiciones: dict[str, Expression]) -> dict[str, list[Expression]]:
    """
        Dada un ast de una consulta SQL de postgres obtiene todas las condiciones
        de los distintos JOINs

        Esta función solo tiene en cuenta las condiciones que 
        son de igualdad

        Tenga en cuenta que esta funcion tiene en cuenta el numero de 
        condiciones del WHERE que este relacionado a una tabla para 
        saber a que tabla debe asignar la condicion del JOIN. 

        Esta funcion asigna la condicion del JOIN a la tabla que tenga
        menos condiciones (contando las condiciones del WHERE o del JOIN si 
        ya se el asigno alguna)

        TODO:
            - Manejar los casos donde las condiciones no son de igualdad

        Parametros
        --------------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        tabla: Una lista con el nombre de todas las tablas de la consulta

        tablas_alias: Un diccionario cuyos key son los alias de cada tabla y los 
            valores son el nombre original de la tabla 

        condiciones: Un diccionario cuyas key son la tabla (o alias de tabla) con la que 
                     esta relacionado una o varias condicones en el WHERE. Y los valores
                     son una lista de dichas condiciones.

        Retorna
        --------------
            Un diccionario cuyas key son la tabla (o alias de tablas) con la que 
            esta relacionado una o varias condiciones en los JOINs. Y los valores
            son una lista de dichas condiciones.
    """

    elementos_a_revisar = []

    if consulta_sql_ast.args.get('joins') != None:
        elementos_a_revisar = consulta_sql_ast.args['joins']

    condiciones_joins = []
    for elemento in elementos_a_revisar:
        if elemento.args.get('on') == None:
            raise Exception('Todo JOIN debe tener un ON')

        condiciones_joins.append(elemento.args['on'])

    condiciones_por_tablas = {}

    for condicion in condiciones_joins:
        for nodo in [condicion.this, condicion.args['expression']]:
            if nodo.key != 'column':
                raise Exception(
                    f'La condicion de JOIN {condicion} debe involucrar dos tablas')

            if nodo.table == '':
                raise Exception(f'La condicion {condicion} no es valida')

            if (nodo.table not in tablas and
                    tablas_alias.get(nodo.table) == None):
                raise Exception(
                    f'No existe la tabla o alias de tabla "{nodo.table}"')

        # Calculamos cuantas condiciones estan relacionada con cada una de las tablas
        # que estan involucaradas en la condicion

        numero_condiciones_tabla_izquierda = 0
        if condiciones.get(condicion.this.table) != None:
            numero_condiciones_tabla_izquierda += len(
                condiciones[condicion.this.table])

        if condiciones_por_tablas.get(condicion.this.table) != None:
            numero_condiciones_tabla_izquierda += len(
                condiciones_por_tablas[condicion.this.table])

        numero_condiciones_tabla_derecha = 0
        if condiciones.get(condicion.args['expression'].table) != None:
            numero_condiciones_tabla_derecha += len(
                condiciones[condicion.args['expression'].table])

        if condiciones_por_tablas.get(condicion.args['expression'].table) != None:
            numero_condiciones_tabla_derecha += len(
                condiciones_por_tablas[condicion.args['expression'].table])

        # Le añadimos la condicion a la tabla que tenga menos condiciones, para asi
        # acotar mas el dominio de la consulta

        if numero_condiciones_tabla_izquierda < numero_condiciones_tabla_derecha:
            if condiciones_por_tablas.get(condicion.this.table) == None:
                condiciones_por_tablas[condicion.this.table] = []

            condiciones_por_tablas[condicion.this.table].append(condicion)
        else:
            if condiciones_por_tablas.get(condicion.args['expression'].table) == None:
                condiciones_por_tablas[condicion.args['expression'].table] = []

            condiciones_por_tablas[condicion.args['expression'].table].append(
                condicion)

    return condiciones_por_tablas


def obtener_tablas_joins(consulta_sql_ast: Expression,
                         tablas: list[str],
                         tablas_alias: dict[str, str]) -> dict[str, list[Expression]]:
    """
        Dada un ast de una consulta SQL de postgres obtiene todas las condiciones
        de los distintos JOINs y devuelve las columnas de las tablas utilizadas
        en alguna de estas condiciones

        Parametros
        ------------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        tabla: Una lista con el nombre de todas las tablas de la consulta

        tablas_alias: Un diccionario cuyos key son los alias de cada tabla y los 
                      valores son el nombre original de la tabla 

        Retorna
        -----------

        Un diccionario cuya key son las distintas tablas utilizadas en alguna 
        condicion de un JOIN. Y sus valores son una lista con las distintas 
        columnas de dicha tabla los cuales fueron utilizados en alguna condición
        de JOIN
    """
    elementos_a_revisar = []

    if consulta_sql_ast.args.get('joins') != None:
        elementos_a_revisar = consulta_sql_ast.args['joins']

    condiciones_joins = []
    for elemento in elementos_a_revisar:
        if elemento.args.get('on') == None:
            raise Exception('Todo JOIN debe tener un ON')

        condiciones_joins.append(elemento.args['on'])

    proyecciones_por_tablas = {}

    for condicion in condiciones_joins:
        for nodo in [condicion.this, condicion.args['expression']]:
            if nodo.key != 'column':
                raise Exception(
                    f'La condicion de JOIN {condicion} debe involucrar dos tablas')

            if nodo.table == '':
                raise Exception(f'La condicion {condicion} no es valida')

            if (nodo.table not in tablas and
                    tablas_alias.get(nodo.table) == None):
                raise Exception(
                    f'No existe la tabla o alias de tabla "{nodo.table}"')

            tabla = nodo.table

            if proyecciones_por_tablas.get(tabla) == None:
                proyecciones_por_tablas[tabla] = []

            if nodo not in proyecciones_por_tablas[tabla]:
                proyecciones_por_tablas[tabla].append(nodo)

    return proyecciones_por_tablas


def obtener_dependencia(tabla: str, condiciones: list[Expression]):
    """
        Dada una tabla y una lista de condiciones revisa si existe 
        alguna condicion donde se relacione a esta tabla con otra. Lo
        que quiere decir que la primera tabla depende de otra.

        Tenga en cuenta que esta funcion supone que una tabla X solo
        puede depender de otra tabla Y. No es posible (por el momento)
        que X depende de Y y de otra tabla Z al mismo tiempo.

        TODO
        -----------------
        Modificar esta funcion para que maneje el caso de que una tabla
        X pueda depender de una tabla Y y otra Z

        Parametros
        ------------
        tabla: Un string que el alias (o nombre original) de la tabla la cual 
               se quiere verificar si depende de otra.

        condiciones: Una lista de expresiones de sqlglot. Estas expresiones 
                     representan condiciones

        Retorna
        ---------

        Un string vacio si la tabla no depende de ninguna otra tabla. O Un string
        con el nombre de la tabla de la que depende.
    """
    dependencia = ''
    for condicion in condiciones:
        for nodo in [condicion.this, condicion.args['expression']]:
            if nodo.key == 'column' and nodo.table != tabla:
                dependencia = nodo.table

    return dependencia


def obtener_group_by(consulta_sql_ast: Expression) -> list[dict[str, str]]:
    """
    Dada un ast de una consulta SQL de postgres el cual tiene uno GROUP BY
    obtiene todas las columnas necesarias para realizar la agrupacion

    Parametros
    -----------------

    consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                      ast de una consulta SQL

    Retorna
    ------------

    Una lista de diccionarios que almacenan las tablas y columnas necesarias 
    para realizar la agrupacion
    """
    if consulta_sql_ast.args.get('group') == None:
        raise Exception('La consulta SQL debe tener un group by')

    group_by_ast = consulta_sql_ast.args['group']

    return [{'tabla': i.args['table'].this, 'columna': i.args['this'].this} for i in group_by_ast.args['expressions']]


def obtener_order_by(consulta_sql_ast: Expression) -> list[dict[str, str]]:
    """
        Dada una consulta QSL, extrae la instruccion ORDER BY que indica el 
        orden solicitado para mostrar los registros a devolver en la consulta.

        parametros
        -----------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        retorna
        --------
            Una lista de diccionarios que indican las propiedades del orden
            solicitado
    """
    if consulta_sql_ast.args.get('order') == None:
        raise Exception('La consulta SQL debe tener un order by')

    ordenes: list[Ordered] = consulta_sql_ast.args.get(
        'order').args.get('expressions')

    return [{'tabla': i.this.table,
             'columna': i.this.this.this,
             'tipo': "DESC" if (i.args.get('desc') is not None) else "ASC"} for i in ordenes if i is not None]


def obtener_limit(consulta_sql_ast: Expression) -> int:
    """
        Dada una consulta QSL, extrae la instruccion que indica el limite
        de registros a devolver en la consulta.

        parametros
        -----------
        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        retorna
        --------
            Una entero que indica el limite
    """

    return int(str(consulta_sql_ast.args.get('limit').args.get('expression')))


def dividir_joins(consulta_sql_ast: Expression) -> dict[str, dict[str, Any]]:
    """
        Dada un ast de una consulta SQL de postgres el cual tiene cero o mas joins   
        obtiene la informacion suficiente para crear una o mas consultas con 
        complejidad igual o menor.

        Parametros
        -----------------

        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        ------------

        Un diccionario cuya claves son el alias (o nombre) de una tabla y los valores
        son otros diccionarios cuya claves son el nombre de la informacion de esa tabla
        y los valores son la informacion necesaria.
    """

    tablas, tablas_alias = obtener_tablas(consulta_sql_ast)

    proyecciones, agregaciones = obtener_proyecciones_agregaciones(
        consulta_sql_ast, tablas, tablas_alias)

    # Pasamos por todas las funciones de agregación y si trabajan sobre una columna de alguna tabla
    # la agregamos a las proyecciones

    for agregacion in agregaciones:
        if agregacion.this.key == "column":
            tabla = agregacion.this.args['table'].this

            if (tabla not in tablas and
                    tablas_alias.get(tabla) == None):
                raise Exception(
                    f'No existe la tabla o alias de tabla "{tabla}"')

            if proyecciones.get(tabla) == None:
                proyecciones[tabla] = []

            if agregacion.this not in proyecciones[tabla]:
                proyecciones[tabla].append(agregacion.this)

    condiciones, condiciones_or = obtener_condiciones_where(
        consulta_sql_ast).values()
    condiciones_por_tablas = clasificar_condiciones_where(
        condiciones, tablas, tablas_alias)

    condiciones_or_mantener = []

    for condicion_or in condiciones_or:
        mover = True
        tabla = None

        condicion_or_sin_parentesis = condicion_or

        if condicion_or_sin_parentesis.key == 'paren':
            condicion_or_sin_parentesis = condicion_or_sin_parentesis.this

        for condicion in obtener_condiciones(condicion_or_sin_parentesis, 'or'):
            tabla_1, tabla_2 = obtener_tablas_condiciones(condicion)

            if tabla_1 == '' and tabla_2 == '':
                mover = False
                break

            if tabla_1 != '' and tabla_2 != '' and tabla_1 != tabla_2:
                mover = False
                break

            if tabla == None:
                if tabla_1 != '':
                    tabla = tabla_1
                    continue

                if tabla_2 != '':
                    tabla = tabla_2
            else:
                if tabla_1 != '' and tabla_1 != tabla:
                    mover = False
                    break

                if tabla_2 != '' and tabla_2 != tabla:
                    mover = False
                    break

        if mover and tabla != None:
            if condiciones_por_tablas.get(tabla) == None:
                condiciones_por_tablas[tabla] = []

            condiciones_por_tablas[tabla].append(condicion_or)
        else:
            condiciones_or_mantener.append(condicion_or)

    condiciones_or = condiciones_or_mantener

    condiciones_having = []

    condiciones_having_or = []

    if consulta_sql_ast.args.get('having') != None:
        condiciones_having, condiciones_having_or = obtener_condiciones_having(
            consulta_sql_ast).values()

    condiciones_joins_por_tablas = obtener_condiciones_joins(
        consulta_sql_ast, tablas, tablas_alias, condiciones_por_tablas)

    proyecciones_joins = obtener_tablas_joins(
        consulta_sql_ast, tablas, tablas_alias)

    aliases = tablas_alias.keys()

    datos_miniconsultas = {}
    for alias in aliases:
        datos_miniconsultas[alias] = {'tabla': tablas_alias[alias]}

        if proyecciones.get(alias) != None:
            datos_miniconsultas[alias]['proyecciones'] = proyecciones[alias]
        else:
            datos_miniconsultas[alias]['proyecciones'] = []

        if proyecciones_joins.get(alias) != None:
            datos_miniconsultas[alias]['proyecciones'] += [i for i in proyecciones_joins[alias]
                                                           if i not in datos_miniconsultas[alias]['proyecciones']]

        if condiciones_por_tablas.get(alias) != None:
            datos_miniconsultas[alias]['condiciones'] = condiciones_por_tablas[alias]
        else:
            datos_miniconsultas[alias]['condiciones'] = []

        if condiciones_joins_por_tablas.get(alias) != None:
            datos_miniconsultas[alias]['condiciones_joins'] = condiciones_joins_por_tablas[alias]
        else:
            datos_miniconsultas[alias]['condiciones_joins'] = []

    resultado = {'datos miniconsultas': datos_miniconsultas,
                 'datos globales': {'proyecciones': proyecciones,
                                    'agregaciones': agregaciones,
                                    'order by': [],
                                    'limite': -1,
                                    'group by': [],
                                    'condiciones or': condiciones_or,
                                    'condiciones having': condiciones_having,
                                    'condiciones having or': condiciones_having_or}}

    if consulta_sql_ast.args.get('order') != None:
        resultado['datos globales']['order by'] = obtener_order_by(
            consulta_sql_ast)

    if consulta_sql_ast.args.get('group') != None:
        resultado['datos globales']['group by'] = obtener_group_by(
            consulta_sql_ast)

    if consulta_sql_ast.args.get('limit') != None:
        resultado['datos globales']['limite'] = obtener_limit(consulta_sql_ast)

    return resultado


def obtener_miniconsultas_join(consulta_sql_ast: Expression) -> dict[str, list[miniconsulta_sql]]:
    """
        Dada un ast de una consulta SQL de postgres con cero o mas joins lo divide en consultas mas 
        simples de forma tal que despues se pueda usar la informacion de estas 
        nuevas consultas mas pequeñas para hacerle preguntas a algun LLM.

        Parametros
        -----------------

        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        ------------

        Un diccionario con las miniconsultas que son dependientes (necesitan del 
        resultado de otra miniconsulta) y las independientes.
    """

    datos_divididos_joins = dividir_joins(consulta_sql_ast)
    datos_miniconsultas = datos_divididos_joins['datos miniconsultas']
    datos_globales = datos_divididos_joins['datos globales']
    dependencias = {}

    aliases = datos_miniconsultas.keys()

    miniconsultas_independientes = {}
    miniconsultas_dependientes = {}

    for alias in aliases:
        dependencia = obtener_dependencia(
            alias, datos_miniconsultas[alias]['condiciones_joins'])

        if dependencia != '':
            dependencias[alias] = dependencia
            miniconsultas_dependientes[alias] = miniconsulta_sql(tabla=datos_miniconsultas[alias]['tabla'],
                                                                 alias=alias,
                                                                 proyecciones=datos_miniconsultas[alias]['proyecciones'],
                                                                 condiciones=datos_miniconsultas[alias]['condiciones'],
                                                                 condiciones_join=datos_miniconsultas[alias]['condiciones_joins'])
        else:
            miniconsultas_independientes[alias] = miniconsulta_sql(tabla=datos_miniconsultas[alias]['tabla'],
                                                                   alias=alias,
                                                                   proyecciones=datos_miniconsultas[
                                                                       alias]['proyecciones'],
                                                                   condiciones=datos_miniconsultas[alias]['condiciones'],
                                                                   condiciones_join=datos_miniconsultas[alias]['condiciones_joins'])

    for alias, dependencia_mc in dependencias.items():
        if miniconsultas_independientes.get(dependencia_mc) != None:
            dependencia = miniconsultas_independientes[dependencia_mc]
        else:
            dependencia = miniconsultas_dependientes[dependencia_mc]

        if len(miniconsultas_dependientes[alias].dependencias) == 0:
            miniconsultas_dependientes[alias].dependencias = [dependencia]
        else:
            miniconsultas_dependientes[alias].dependencias.append(dependencia)

    lista_condiciones_join = []
    for miniconsulta in list(miniconsultas_dependientes.values()) + list(miniconsultas_independientes.values()):
        lista_condiciones_join += miniconsulta.condiciones_join

    return {'ejecutor': join_miniconsultas_sql(proyecciones=datos_globales['proyecciones'],
                                               condiciones_join=lista_condiciones_join,
                                               miniconsultas_dependientes=list(
                                                   miniconsultas_dependientes.values()),
                                               miniconsultas_independientes=list(
                                                   miniconsultas_independientes.values()),
                                               lista_group_by=datos_globales['group by'],
                                               lista_order_by=datos_globales['order by'],
                                               limite=datos_globales['limite'],
                                               condiciones_or=datos_globales['condiciones or'],
                                               condiciones_having=datos_globales['condiciones having'],
                                               condiciones_having_or=datos_globales['condiciones having or'],
                                               lista_agregaciones=datos_globales['agregaciones']),
            'dependientes': list(miniconsultas_dependientes.values()),
            'independientes': list(miniconsultas_independientes.values())}


def obtener_miniconsultas_operacion(consulta_sql_ast: Expression) -> dict[str, list[miniconsulta_sql]]:
    """
        Dada un ast de una consulta SQL de postgres con una o mas operaciones de conjuntos, lo divide  
        en consultas mas simples de forma tal que despues se pueda usar la  
        informacion de estas nuevas consultas mas pequeñas para hacerle preguntas 
        a algun LLM.

        Ten en cuenta que sqlglot asocia a 'izquierda' o mejor dicho asocia 
        hacia arriba

        Parametros
        -----------------

        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        ------------

        Un diccionario con las miniconsultas que son dependientes (necesitan del 
        resultado de otra miniconsulta) y las independientes.
    """
    if consulta_sql_ast.key not in OPERACIONES_CONJUNTOS:
        raise Exception(
            "Para ejecutar esta funcion la consulta SQL debe tener al menos una operacion")

    # la parte derecha de una operacion siempre sera una consulta que no es una operacion

    miniconsultas_derecha = obtener_miniconsultas(
        consulta_sql_ast.args['expression'].sql())

    miniconsultas_izquierda = obtener_miniconsultas(
        consulta_sql_ast.this.sql())

    miniconsultas_totales = {'dependientes': miniconsultas_izquierda['dependientes'] + miniconsultas_derecha['dependientes'],
                             'independientes': miniconsultas_izquierda['independientes'] + miniconsultas_derecha['independientes']}

    miniconsultas_totales['ejecutor'] = operacion_miniconsultas_sql(
        consulta_sql_ast.key, miniconsultas_derecha['ejecutor'], miniconsultas_izquierda['ejecutor'])
    return miniconsultas_totales


def obtener_miniconsultas(consulta_sql: str) -> dict[str, list[miniconsulta_sql]]:
    """
        Divide una consulta SQL en miniconsultas de menor complejidad
        y devuelve una lista con las distintas miniconsultas a ejecutar

        parametros
        -----------
        consulta_sql: Un string con la consulta SQL en sintaxis de Postgres

        retorna
        --------
            Una lista con las distintas miniconsultas a ejecutar
    """
    miniconsultas = {}
    consulta_sql_ast = parse_one(consulta_sql, dialect='postgres')

    # Caso donde la consulta es una operacion de conjuntos
    if consulta_sql_ast.key in OPERACIONES_CONJUNTOS:
        return obtener_miniconsultas_operacion(consulta_sql_ast)

    condiciones = obtener_condiciones_where(consulta_sql_ast)['condiciones']

    # Caso donde la consulta es una consulta anidada
    for condicion in condiciones:
        if (isinstance(condicion, In) or
            isinstance(condicion, Not) or
            isinstance(condicion, Binary) and isinstance(condicion.this, Subquery) or
                isinstance(condicion, Binary) and isinstance(condicion.args.get('expression'), Subquery)):
            return obtener_miniconsultas_anidadas(consulta_sql_ast)

#
    # Caso donde la consulta es un select sin condicion IN
    if consulta_sql_ast.key == 'select':
        return obtener_miniconsultas_join(consulta_sql_ast)

    return miniconsultas


def obtener_lista_miniconsultas(consulta_sql: str) -> list[miniconsulta_sql]:
    """
        Dado una consulta SQL lo divide en consultas mas simples de forma tal
        que despues se pueda usar la informacion de estas nuevas consultas
        mas pequeñas para hacerle preguntas a algun LLM.

        Tenga en cuenta que el resultado de esta funcion es una lista donde 
        las primeras consultas son consultas que dependen de otras, y las 
        ultimas son consultas que no depende de ninguna otra.

        Parametros
        -----------------

        consulta_sql: Un string con la consulta SQL

        Retorna
        ------------

        Una lista con las distintas consultas mas simples a realizar para devolver
        la información que requiere la consulta original.
    """
    miniconsultas = obtener_miniconsultas(consulta_sql)

    return miniconsultas['dependientes'] + miniconsultas['independientes']


def obtener_ejecutor(consulta_sql: str):
    """
        Divide una consulta SQL en miniconsultas de menor complejidad
        y devuelve el ejecutor necesario para combinar las miniconsultas
        de forma tal que se obtenga un resulado suficiente para responder
        la consulta SQL original

        parametros
        -----------
        consulta_sql: Un string con la consulta SQL en sintaxis de Postgres

        retorna
        --------
            El ejecutor necesario para combinar las miniconsultas
    """
    return obtener_miniconsultas(consulta_sql)['ejecutor']


def obtener_miniconsultas_anidadas(consulta_sql_ast: Expression):
    """
        Dada un ast de una consulta SQL de postgres con una o mas condiciones anidadas lo  
        divide en consultas mas simples de forma tal que despues se pueda usar la informacion  
        de estas nuevas consultas mas pequeñas para hacerle preguntas a algun LLM.

        Parametros
        -----------------

        consulta_sql_ast: Un objeto Expression de sqlglot. Representa un 
                          ast de una consulta SQL

        Retorna
        ------------

    """

    aliases, tablas_alias = obtener_tablas(consulta_sql_ast)
    condiciones, condiciones_or = obtener_condiciones_where(
        consulta_sql_ast).values()
    proyecciones, agregaciones = obtener_proyecciones_agregaciones(
        consulta_sql_ast, aliases, tablas_alias)
    condiciones_clasificada = clasificar_condiciones_where(
        condiciones, aliases, tablas_alias)
    condiciones_joins = obtener_condiciones_joins(
        consulta_sql_ast, aliases, tablas_alias, condiciones_clasificada)

    limite = -1
    lista_order_by = []
    lista_group_by = []
    lista_having = []
    lista_having_or = []
    if consulta_sql_ast.args.get('order') != None:
        lista_order_by = obtener_order_by(consulta_sql_ast)

    if consulta_sql_ast.args.get('group') != None:
        lista_group_by = obtener_group_by(consulta_sql_ast)

    if consulta_sql_ast.args.get('limit') != None:
        limite = obtener_limit(consulta_sql_ast)

    if consulta_sql_ast.args.get('having') != None:
        lista_having, lista_having_or = obtener_condiciones_having(
            consulta_sql_ast).values()

    return {"ejecutor": miniconsulta_sql_anidadas(proyecciones=proyecciones,
                                                  agregaciones=agregaciones,
                                                  aliases=aliases,
                                                  tablas_aliases=tablas_alias,
                                                  condiciones=condiciones,
                                                  condiciones_or=condiciones_or,
                                                  condiciones_join=condiciones_joins,
                                                  limite=limite,
                                                  lista_order_by=lista_order_by,
                                                  lista_group_by=lista_group_by,
                                                  condiciones_having=lista_having,
                                                  condiciones_having_or=lista_having_or)}
