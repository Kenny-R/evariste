from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from traduccion_sql_ln import *
from parser_SQL import *
from embeddings import *
import jellyfish
import logging
import mdpd
import json

configuraciones = json.load(open("./configuraciones.json"))

EMBEDDINGS_MODEL = configuraciones['EMBEDDINGS_MODEL']
EMBEDDINGS_FOLDER = configuraciones['EMBEDDINGS_FOLDER']
EMBEDDINGS_INDEX = configuraciones['EMBEDDINGS_INDEX']
DEBUG = configuraciones['debug']

# Cantidad de ejecuciones 
global ejecuciones
ejecuciones = 0

# Configuración del modelo de embeding
modelPath = EMBEDDINGS_MODEL

model_kwargs = {'device':'cuda'}

encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

db = cargar_embeddings(EMBEDDINGS_FOLDER, EMBEDDINGS_INDEX, embeddings)

# Configuración para el LLM
retriever = db.as_retriever()

ollama = Ollama(
    base_url='http://localhost:3030',
    # model="llama2-uncensored",
    # model="llama3",
    model="gemma:7b",
    num_ctx=4096,
    temperature = 0.3,
    timeout = 1000 * 60 * 5
)

# Configuraciones para hacer las preguntas
system_prompt=("You are a highly intelligent question answering bot. "
               "You will answer concisely. "
               "Use only the given context to answer the question. "
               "Context: {context}"
               "\n{format_instructions}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "In the next Markdown table there are the answer of the query: {question}"),
    ],
)

def format_docs(docs):
    texto = "\n\n".join(doc.page_content for doc in docs)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # for doc in docs:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print(doc.metadata['source'])
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print(doc.page_content)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return texto

def crear_instrucciones(columnas: list[str]):
    texto = "Instructions: \n"
    texto = "Format the information as a table with columns for "
    
    if len(columnas) == 1:
        texto += columnas[0]
    elif len(columnas) > 1:
        texto += ", ".join(columnas[:-1]) + f" and {columnas[-1]}"
            
    texto += " Your response should be a table\n"

    texto += "If your answer is a number like millions or thousands, return the always all its digits using the format used in America. \n"
    texto += "If I ask you a question that is rooted in truth, you will give you the answer.\n"
    texto += "If I ask you a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'. "
    
    return (lambda *args: texto)

async def hacer_consulta(traduccion: str, columnas: list[str]):
    global ejecuciones
    
    columnas_traduccion = type(columnas)(columnas)
    print("############################################################")
    print(f"Procesando la pregunta:\n\t{traduccion}")
    rag_chain = (
        {"context": retriever | format_docs, 
        "question": RunnablePassthrough(),
        "format_instructions": crear_instrucciones(columnas_traduccion)
        }
        | prompt
        | ollama
        | StrOutputParser()
    )

    resultado_limpio = rag_chain.invoke(traduccion)
    print("Resultado sin procesar: ")
    print(resultado_limpio)
        
    if DEBUG:  
        logging.warning(f"Procesando la petición: {traduccion}\n")
        logging.info(f"Resultado sin procesar: \n{resultado_limpio}\n")

    try:
        df = mdpd.from_md(resultado_limpio)
    except:
        df = pd.DataFrame()
        
    print(f"Columnas requeridas: {columnas}")
    
    print("Resultado Antes de procesar: ")
    print(df.to_markdown(index=False))
    if len(df) != 0:
        asignaciones = {}
        
        por_asignar = list(df.columns)
                
        por_asignar = [x for x in por_asignar if type(x) != int]
        
        while len(por_asignar) != 0:
            columna_por_asignar = por_asignar.pop(0)
            
            posibles_asignaciones = sorted(columnas, key=lambda x: jellyfish.jaro_similarity(x, columna_por_asignar), reverse= True)
            
            for asignacion in posibles_asignaciones:
                if asignaciones.get(asignacion) != None:
                    columna_asignada = asignaciones.get(asignacion)
                    
                    if jellyfish.jaro_similarity(columna_por_asignar, asignacion) > jellyfish.jaro_similarity(columna_asignada, asignacion):
                        por_asignar.append(columna_asignada)
                        asignaciones[asignacion] = columna_por_asignar
                else:
                    asignaciones[asignacion] = columna_por_asignar
        
        nuevo_df = pd.DataFrame()
        
        df  = df.loc[:,~df.columns.duplicated()]
        
        for columna in asignaciones:
        
            nuevo_df.insert(len(nuevo_df.columns), columna, df[asignaciones[columna]])
        
        df = nuevo_df
        
        print("Resultado despues de procesar: ")
        print(df.to_markdown(index=False))
        print("############################################################")
        
        
        if DEBUG: logging.info(f"Resultado procesado:\n{df.to_string()}\n")
        
    else:
        if DEBUG: logging.warning("La respuesta del LLM no tiene una tabla\n")
        
    ejecuciones +=1
    return df

def crear_ejemplos():
    texto = "Examples: \n"
    fewshot_chatgpt = [
                        ['What is human life expectancy in the United States?', '78.'],
                        ['Who was president of the United States in 1955?', 'Dwight D. Eisenhower.'],
                        ['Which party was founded by Gramsci?', 'Comunista.'],
                        ['What is the capital of France?', 'Paris.'],
                        ['What is a continent starting with letter O?', 'Oceania.'],
                        ['Where were the 1992 Olympics held?', 'Barcelona.'],
                        ['How many squigs are in a bonk?', 'Unknown'],
                        ['What is the population of Venezuela: 28,300,000']]
    
    texto += "\n".join([': '.join(shot) for shot in fewshot_chatgpt])
    return (lambda *args: texto)

async def hacer_pregunta(pregunta: str, contexto: str = "", instrucciones_extra:str = "", con_ejemplos: bool = False):

    global ejecuciones
    
    system_prompt=("You are a highly intelligent question answering bot. "
                    "You will answer concisely. "
                    "Use only the given context to answer the question. "
                    "Context: {context}"
                    "\n{format_instructions}"
                    "\n{examples}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ],
    )
    instrucciones = "Intructions: "
    instrucciones += instrucciones_extra
    datos = {"context": lambda x: format_docs(contexto), 
            "question": RunnablePassthrough(),
            "format_instructions": lambda *args: instrucciones} 
    
    datos['examples'] = lambda *args: ""
    
    if con_ejemplos:
        datos['examples'] = crear_ejemplos()
        
    print(f"Procesando la pregunta:\n\t{pregunta}")
    rag_chain = (
        datos
        | prompt
        | ollama
    )

    resultado_limpio = rag_chain.invoke(pregunta)
    ejecuciones += 1
    print(f"Resultado: {resultado_limpio}\n")
    return resultado_limpio