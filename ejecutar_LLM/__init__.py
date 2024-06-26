from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from traduccion_sql_ln import *
from parser_SQL import *
from embeddings import *
import logging
import mdpd
import json

configuraciones = json.load(open("./configuraciones.json"))

EMBEDDINGS_MODEL = configuraciones['EMBEDDINGS_MODEL']
EMBEDDINGS_FOLDER = configuraciones['EMBEDDINGS_FOLDER']
EMBEDDINGS_INDEX = configuraciones['EMBEDDINGS_INDEX']
DEBUG = configuraciones['debug']


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
    model="llama2-uncensored",
    # model="llama3",
    # model="gemma:7b",
    num_ctx=4096,
    temperature = 0.3
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
    # print("Resultado sin procesar: ")
    # print(resultado_limpio)
    # print("############################################################")
    
    if DEBUG:  
        logging.warning(f"Procesando la petición: {traduccion}\n")
        logging.info(f"Resultado sin procesar: \n{resultado_limpio}\n")

    df = mdpd.from_md(resultado_limpio)
    if len(df) != 0:
        if len(df.columns) > len(columnas):
            # Hacer una busqueda de similitud por los nombres
            if DEBUG: logging.info("La tabla que se obtuvo de la respuesta tiene mas columnas de las que se le pidio\n")
            df.columns = columnas + list(df.columns)[len(columnas):]
        elif len(df.columns) < len(columnas):
            # Hacer una busqueda de similitud por los nombres
            if DEBUG: logging.info("La tabla que se obtuvo de la respuesta tiene menos columnas de las que se le pidio\n")
            df.columns = columnas[:len(df.columns)]
        else:
            df.columns = columnas
        
        if DEBUG: logging.info(f"Resultado procesado:\n{df.to_string()}\n")
        
    else:
        if DEBUG: logging.warning("La respuesta del LLM no tiene una tabla\n")
        

    return df