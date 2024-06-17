import json
import logging
import asyncio
import warnings
from parser_SQL import *
from datetime import datetime
from sqlglot import Expression
from traduccion_sql_ln.funciones import *
from sqlglot.expressions import In, Binary, Not, Subquery, Paren, Column

configuraciones = json.load(open("./configuraciones.json"))
DEBUG = configuraciones['debug']

warnings.filterwarnings('ignore')

def main():
    # prueba_parseo_singular()
    # hacer_pruebas_en_lote()
    # prueba_LLM()
    pruebas_join()

def prueba_LLM():
    # consulta_sql= '''SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.HeadOfState = "Beatrix" AND T2.IsOfficial = "T"'''
    # consulta_sql = """select distinct t3.name from country as t1 join countrylanguage as t2 on  t2.countrycode = t1.code join city as t3 on  t3.countrycode = t1.code where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    consulta_sql = """select distinct t3.name from country as t1 join countrylanguage as t2 on  t2.country_name = t1.country_name join city as t3 on  t3.country_name = t1.country_name where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    
    lista_miniconsulta = obtener_lista_miniconsultas(consulta_sql)

    # print(lista_miniconsulta)
    for miniconsulta in lista_miniconsulta:

        print("inicio de la ejecucion de la consulta")
        asyncio.run(miniconsulta.ejecutar())
        print("fin de la ejecucion de la consulta")

        traduccion, _, _ = miniconsulta.crear_prompt()
        
        print("################################################")
        print(f"Pregunta: {traduccion}")
        print("Respuesta: ")
        df = miniconsulta.resultado
        print(df)

def prueba_parseo_singular():

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
    #                 SELECT t1.border 
    #                 FROM border_info as t1
    #                 WHERE t1.state_name IN ( SELECT t2.border 
    #                       FROM border_info as t2
    #                       WHERE t2.state_name = "colorado" 
    #                     );
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

    # consulta_sql = '''SELECT T1.name 
    #                   FROM employees as T1 
    #                   JOIN personal_data as T2 ON T1.name = T2.name 
    #                   WHERE T2.xd = 10 and T2.hola = 200 '''
    

    # consulta_sql = """select distinct t3.name from country as t1 join countrylanguage as t2 on  t2.countrycode = t1.code join city as t3 on  t3.countrycode = t1.code where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    consulta_sql = """select count(t1.name), min(t1.population), max(t1.population), avg(t1.population) from country as t1 where t1.continent = 'north america'"""
    # miniconsulta_sql = obtener_ejecutor(consulta_sql)
    print(obtener_ejecutor(consulta_sql).lista_agregaciones)

def pruebas_parseo_en_lote():
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

def pruebas_join():
    # consulta_sql = """select t3.city_name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode join city as t3 on t1.code = t3.countrycode where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    # consulta_sql = """select t3.name from country as t1 join countrylanguage as t2 on  t2.country_name = t1.country_name join city as t3 on  t3.country_name = t1.country_name where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    # consulta_sql = """select t1.name, t1.capital from country as t1 join language as t2 on t1.language = t2.language_name where t1.continent = 'north america' order by t2.language_name ASC, t1.name DESC, t1.capital ASC"""
    # consulta_sql = """select t1.name, t1.population from country as t1 where t1.continent = 'north america'"""
    consulta_sql = """select count(t1.name), min(t1.population), max(t1.population), avg(t1.population) from country as t1 where t1.continent = 'north america'"""
    # consulta_sql = """select t1.city_name from city as t1 where t1.country_name = 'china'"""
    ejecutor: join_miniconsultas_sql = obtener_ejecutor(consulta_sql)
    
    ejecutor.ejecutar()

    print(ejecutor.resultado)

if __name__ == "__main__":
    main()