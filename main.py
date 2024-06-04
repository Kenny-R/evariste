import json
import logging
from datetime import datetime
from parser_SQL import *
from sqlglot import Expression
from sqlglot.expressions import In, Binary, Not, Subquery, Paren, Column
from traduccion_sql_ln.funciones import *
configuraciones = json.load(open("./configuraciones.json"))
DEBUG = configuraciones['debug']

def main():
    prueba_singular()
    # hacer_pruebas_en_lote()

def prueba_singular():

    if DEBUG:
        nombre = 'log_parser_' + datetime.today().strftime('%d_%m_%Y') + ".log"

        logging.basicConfig(filename=nombre, 
                            filemode='w', 
                            format='%(levelname)s - %(message)s',
                            level=logging.INFO,
                            force=True)


    # consulta_sql = """
    #                 select distinct t3.name 
    #                 from country as t1 
    #                 join countrylanguage as t2 on t1.code = t2.countrycode 
    #                 join city as t3 on t3.countrycode = t1.verga  
    #                 where t2.isofficial = 't' and (t2.language = 'chinese' or t1.continent = "asia") and t3.nose = 1 and t3.otra = 2 and t3.jejox = 3
    #             """

    # consulta_sql = """
    #                 select t1.name
    #                 from country as t1 
    #                 join countrylanguage as t2 on t1.code = t2.countrycode 
    #                 join tumadre as t3 on t1.age = t3.age
    #                 where t2.isofficial = 't' and t2.language = 'chinese' and t1.nose = panqueca
    #                 """

    # consulta_sql = """
                    # SELECT t1.border 
                    # FROM border_info as t1
                    # WHERE t1.state_name IN ( SELECT t2.border 
                    #       FROM border_info as t2
                    #       WHERE t2.state_name = "colorado" 
                    #     );
    #                 """

    # consulta_sql = """
    #                 select t1.a
    #                 from nose as t1
    #                 join hola as t2 on t1.c = t2.a
    #                 where t1.b = 'x'
    #             """

    # consulta_sql = """
    #                 SELECT SUM(T2.Name)
    #             FROM
    #                 country AS T1
    #                 JOIN city AS T2 ON T2.CountryCode = T1.Code
    #             WHERE
    #                 T1.Continent = 'Europe'
    #                 AND T1.Name NOT IN (
    #                     SELECT
    #                         T3.Name
    #                     FROM
    #                         country AS T3
    #                         JOIN countrylanguage AS T4 ON T3.Code = T4.CountryCode
    #                     WHERE
    #                         T4.IsOfficial = 'T'
    #                         AND T4.Language = 'English'
    #                 )
    #                """s

    consulta_sql = '''SELECT T1.Name FROM country as T1 WHERE 10 > T1.pgb and (T1.Continent = T1.Name OR T1.Continent = "Europe")'''
    
    miniconsulta_sql = obtener_lista_miniconsultas(consulta_sql)[0]
    print(traducir_miniconsulta_sql(miniconsulta_sql))
    #for cond in miniconsulta_sql.condiciones:
    #    columna = cond.args.get('this').args.get('this') if isinstance(cond.args.get('this'), Column) else cond.args.get('expression').args.get('this')
    #    print(columna)

def hacer_pruebas_en_lote():
    import pandas as pd

    nombre = 'log_parser_' + datetime.today().strftime('%d_%m_%Y') + ".log"

    logging.basicConfig(filename=nombre, 
                        filemode='w', 
                        format='%(levelname)s - %(message)s',
                        level=logging.INFO,
                        force=True)

    df = pd.read_csv('ignorar/queries_ejecutar.csv', sep=";")
    for i, fila in df.iterrows():
        if fila.loc['ejecutar'] == "No":
            continue

        logging.warning(f'parseando la consulta nro.{i + 1}')
        logging.warning(f'Esta es:  {fila.loc["query"]}')
        try:
            resultado = obtener_ejecutor(fila.loc['query'])
            logging.warning(f'Se parseo la consulta nro.{i + 1}')
            logging.warning(f'Esta es:  {fila.loc["query"]}')
            logging.warning(f'Se obtuvo el siguiente resultado:\n{str(resultado)}')
        except:
            logging.error(f'No se pudo completar el parse de la consulta {fila.loc["query"]}', exc_info=True)

if __name__ == "__main__":
    main()