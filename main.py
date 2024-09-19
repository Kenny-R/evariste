import ast
import json
import logging
import asyncio
import warnings
import pandas as pd
from time import time
from parser_SQL import *
from datetime import datetime
from ejecutar_LLM import hacer_consulta, hacer_pregunta
import ejecutar_LLM
from traduccion_sql_ln.funciones import *
configuraciones = json.load(open("./configuraciones.json"))
DEBUG = configuraciones['debug']

warnings.filterwarnings('ignore')

def main():
    # prueba_parseo_singular()
    # hacer_pruebas_en_lote()
    # prueba_LLM()
    prueba_singular()
    # pruebas_operacion()
    # pruebas_join()
    # pruebas_anidamientos()
    # ejecucion_repetida_en_lote('./ignorar/queries_ejecutar_modificados.xlsx')
    # ejecucion_repetida_nl_en_lote('./ignorar/queries_ejecutar_modificados.xlsx')
    # ejecucion_repetida_nl_mod_en_lote('./ignorar/queries_ejecutar_modificados.xlsx')


def prueba_LLM():
    # consulta_sql= '''SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.HeadOfState = "Beatrix" AND T2.IsOfficial = "T"'''
    consulta_sql = """select distinct t3.name from country as t1 join countrylanguage as t2 on  t2.countrycode = t1.code join city as t3 on  t3.countrycode = t1.code where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    
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

def prueba_singular():
    # if DEBUG:
    #     nombre = 'log_parser_' + datetime.today().strftime('%d_%m_%Y') + ".log"

    #     logging.basicConfig(filename=nombre, 
    #                         filemode='w', 
    #                         format=u'%(levelname)s: %(message)s',
    #                         level=logging.INFO,
    #                         force=True,
    #                         encoding="utf-8")


    # consulta_sql = 'select t3.name from country as t1 join countrylanguage as t2 on  t1.country_name = t2.country_name join city as t3 on  t1.country_name = t3.country_name where t2.isofficial = "t" and t2.language = "chinese" and t1.continent = "asia"'
    # consulta_sql = 'SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Name = T2.Country_name WHERE T1.GovernmentForm = "Republic"'
    # consulta_sql = '''SELECT t1.area FROM country as t1 WHERE t1.countryName = "Spain";'''
    # consulta_sql = '''SELECT T1.Name , T1.District FROM city AS T1 JOIN countrylanguage AS T2 ON T1.CountryName = T2.CountryName WHERE T2.Language = "English" ORDER BY T1.District DESC LIMIT 10'''
    # consulta_sql = '''SELECT T1.state_name FROM state as T1 WHERE T1.state_name NOT IN ( SELECT T2.border FROM border_info as T2 WHERE T2.state_name = "texas" ) and T1.Country_Name = "United States"'''
    # consulta_sql = '''SELECT T1.Capital FROM Country as T1 WHERE T1.Name IN ( SELECT T2.Country_Name FROM Country_Language as T2 WHERE T2.Name = "English" and T2.IsOfficialLanguage = 'T' )'''
    # consulta_sql = '''SELECT MIN(T1.Population) FROM Country as T1 WHERE T1.Continent = "Europe"'''
    # consulta_sql = '''SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Name = T2.CountryName WHERE T1.HeadOfState = "Beatrix" AND T2.IsOfficial = "T"'''
    consulta_sql = '''select t3.Name from Country as t1 join CountryLanguage as t2 on t1.Name = t2.CountryName join City as t3 on t1.Name = t3.CountryName where t2.IsOfficial = "T" and t2.Language = "chinese" and t1.Continent = "asia"'''
    ejecutor = obtener_ejecutor(consulta_sql)
    
    ejecutor.ejecutar()

    print("Resultado final: ")
    print(ejecutor.resultado)
    print(f'Se hicieron {ejecutar_LLM.ejecuciones} peticiones al LLM')
    
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

def pruebas_join():
    consulta_sql = '''SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T2.CountryCode = T1.Code  WHERE T1.HeadOfState = "Beatrix" AND T2.IsOfficial = "T"'''
    # consulta_sql = """select t3.city_name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode join city as t3 on t1.code = t3.countrycode where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    # consulta_sql = """select t3.name from country as t1 join countrylanguage as t2 on  t2.country_name = t1.country_name join city as t3 on  t3.country_name = t1.country_name where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = 'asia'"""
    # consulta_sql = """select t1.name, t1.capital from country as t1 join language as t2 on t1.language = t2.language_name where t1.continent = 'north america' order by t2.language_name ASC, t1.name DESC, t1.capital ASC"""
    # consulta_sql = """select t1.name, t1.population from country as t1 where t1.continent = 'north america'"""
    # consulta_sql = """select count(t1.name), min(t1.population), max(t1.population), avg(t1.population) from country as t1 where t1.continent = 'north america'"""
    # consulta_sql = """select t1.city_name from city as t1 where t1.country_name = 'china'"""
    # consulta_sql = '''SELECT T1.name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.name = T2.country_name WHERE T2.Language = "Spanish" AND T2.IsOfficial = "T"'''
    # consulta_sql = '''SELECT MIN ( T2.LENGTH ) FROM river AS T2 WHERE T2.country = "United States"'''
    # consulta_sql = '''SELECT t2.capital FROM state AS t2 JOIN city AS t1 ON t2.state_name = t1.state_name WHERE t1.city_name = "durham"'''
    
    # consulta_sql = '''SELECT T1.name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.name = T2.country_name WHERE T2.Language = "Spanish" AND T2.IsOfficial = "T" AND T1.continent = "Europe"'''
    
    ejecutor: join_miniconsultas_sql = obtener_ejecutor(consulta_sql)
    
    ejecutor.ejecutar()

    print(ejecutor.resultado)

def pruebas_operacion():
    if DEBUG:
        nombre = 'log_parser_' + datetime.today().strftime('%d_%m_%Y') + ".log"

        logging.basicConfig(filename=nombre, 
                            filemode='w', 
                            format=u'%(levelname)s: %(message)s',
                            level=logging.INFO,
                            force=True,
                            encoding="utf-8")
    
    consulta_sql = '''SELECT T1.name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.name = T2.country_name WHERE T2.Language = "Spanish" AND T2.IsOfficial = "T" EXCEPT SELECT T1.name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.name = T2.country_name WHERE T2.Language = "Spanish" AND T2.IsOfficial = "T" AND T1.continent = "Europe"'''
    ejecutor: operacion_miniconsultas_sql = obtener_ejecutor(consulta_sql)
    
    ejecutor.ejecutar()

    print("Resultado final: ")
    print(ejecutor.resultado)

def pruebas_anidamientos():
    consulta_sql = '''SELECT T1.Capital FROM Country as T1 WHERE T1.Name IN ( SELECT T2.Country_Name FROM Country_Language as T2 WHERE T2.Name = "English" and T2.IsOfficialLanguage = 'T' )'''
    # consulta_sql = '''SELECT T1.Capital FROM Country as T1 WHERE T1.Area IN ( SELECT MIN(T2.Area) FROM country as T2 WHERE T2.CountryLanguage = "English" and T2.IsOfficialLanguage = 'T' )'''
    
    ejecutor: miniconsulta_sql_anidadas = obtener_ejecutor(consulta_sql)
    
    ejecutor.ejecutar()

    print("Resultado final: ")
    print(ejecutor.resultado)

def ejecucion_repetida_en_lote(archivo_xls):
    df = pd.read_excel(archivo_xls)
    
    for i, fila in df.iterrows():
        if fila['ejecutar'] == "No":
            continue
        print(f"Procesando el query {fila['query']}")
        ejecucion_repetida(f'./resultados/ejecucion_queries_salida_md/gemma/ejecuciones_query_nro_{i+1}.log', fila['query'], 0, 20)
        
def ejecucion_repetida(nombre, consulta_sql, inicio= 0, fin = 11):
    if DEBUG:
        logging.basicConfig(filename=nombre, 
                            filemode='a', 
                            format=u'%(levelname)s: %(message)s',
                            level=logging.INFO,
                            force=True,
                            encoding="utf-8")
  
    # logging.warning(f"Resumen de ejecucion del query\n {consulta_sql}")
    
    for i in range(inicio, fin):
        logging.warning(f"@Ejecutando la iteracion {i+1}@\n")
        s = time()
        ejecutor = obtener_ejecutor(consulta_sql)
        ejecutor.ejecutar()

        logging.warning("//////////////////////////////////////////////////////////")
        logging.warning("resumen: {")
        logging.info(f"tiempo: {time() - s},")
        logging.info(f"resultado: '''\n{ejecutor.resultado.to_markdown(index=False)}\n '''")
        logging.warning("}")
        logging.warning("//////////////////////////////////////////////////////////")
                
        print(f"Termino la iteracion {i+1}")

def ejecucion_repetida_nl(nombre, consulta_ln, columnas,inicio = 0, fin = 11, titulo = True):
    
    if DEBUG:
        logging.basicConfig(filename=nombre, 
                            filemode='a', 
                            format=u'%(levelname)s: %(message)s',
                            level=logging.INFO,
                            force=True,
                            encoding="utf-8")
  
    if titulo: logging.warning(f"Resumen de ejecucion del query\n {consulta_ln}")
    
    for i in range(inicio, fin):
        logging.warning(f"@Ejecutando la iteracion {i+1}@\n")
        s = time()
        try:
            resultado = asyncio.run(hacer_consulta(consulta_ln, columnas))
            
            logging.warning("//////////////////////////////////////////////////////////")
            logging.warning("resumen: {")
            logging.info(f"tiempo: {time() - s},")
            logging.info(f"cantidad_peticiones_LLM: {ejecutar_LLM.ejecuciones},")
            logging.info(f"resultado: '''\n{resultado.to_markdown(index=False)}\n '''")
            logging.warning("}")
            logging.warning("//////////////////////////////////////////////////////////")
        except:
            logging.warning("//////////////////////////////////////////////////////////")
            logging.warning("resumen: {")
            logging.info(f"tiempo: {time() - s},")
            logging.info(f"cantidad_peticiones_LLM: {ejecutar_LLM.ejecuciones},")
            logging.info(f"resultado: '''\n\n '''")
            logging.warning("}")
            logging.warning("//////////////////////////////////////////////////////////")
                
        print(f"Termino la iteracion {i+1}")
    
def ejecucion_repetida_nl_en_lote(archivo_xls):
    df = pd.read_excel(archivo_xls)
    df['columnas'] = df['columnas'].apply(ast.literal_eval)
    
    for i, fila in df.iterrows():
        if fila['ejecutar'] == "No":
            continue
        
        print(f"Procesando el query {fila['query']}")
        ejecucion_repetida_nl(f'./resultados/ejecucion_preguntas_ln_salida_md/gemma/ejecuciones_preguntas_ln_nro_{i+1}.log',fila['ln'], fila['columnas'], 0, 20)

def ejecucion_repetida_nl_mod_en_lote(archivo_xls):
    df = pd.read_excel(archivo_xls)
    df['columnas'] = df['columnas'].apply(ast.literal_eval)
    
    for i, fila in df.iterrows():
        if fila['ejecutar'] == "No":
            continue
        
        print(f"Procesando el query {fila['query']}")
        consulta = asyncio.run(hacer_pregunta(f"""Translate this SQL sentence to  a question in natural language. SQL Sentence: '{fila['query']}'""", 
                                    [], 
                                    "your response must be the shortest one\ndon't Explain yourself\ndon't apologize if you can't response\n"
                                 ))
        
        ejecucion_repetida_nl(f'./resultados/ejecucion_pregunta_ln_mod_salida_md/gemma/ejecuciones_preguntas_ln_nro_{i+1}.log', consulta, fila['columnas'], 0, 20)

if __name__ == "__main__":
    main()