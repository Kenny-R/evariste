from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from traduccion_sql_ln import *
from parser_SQL import *
from embeddings import *
import mdpd
import json

configuraciones = json.load(open("./configuraciones.json"))

EMBEDDINGS_MODEL = configuraciones['EMBEDDINGS_MODEL']
EMBEDDINGS_FOLDER = configuraciones['EMBEDDINGS_FOLDER']
EMBEDDINGS_INDEX = configuraciones['EMBEDDINGS_INDEX']

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
    num_ctx=4096,
    temperature = 0.2
)

# Configuraciones para hacer las preguntas
system_prompt=("You are a highly intelligent question answering bot. "
               "You will answer concisely. "
               "Use only the given context to answer the question. "
               "Context: {context}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Answer the query.\n{question}\n{examples}\n{format_instructions}\n"),
    ],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
    return (lambda *args: texto)

async def hacer_consulta(traduccion: str, columnas: list[str]):

    columnas_traduccion = type(columnas)(columnas)
    print(f"Procesando la pregunta:\n\t{traduccion}")
    rag_chain = (
        {"context": retriever | format_docs, 
        "question": RunnablePassthrough(),
        "format_instructions": crear_instrucciones(columnas_traduccion),
        "examples": crear_ejemplos()}
        | prompt
        | ollama
        | StrOutputParser()
    )
    df = mdpd.from_md(rag_chain.invoke(traduccion))
    if len(df) != 0:
        if len(df.columns) > len(columnas):
            # Hacer una busqueda de similitud por los nombres
            df.columns = columnas + list(df.columns)[len(columnas):]
        elif len(df.columns) < len(columnas):
            # Hacer una busqueda de similitud por los nombres
            df.columns = columnas[:len(df.columns)]
        else:
            df.columns = columnas

    return df