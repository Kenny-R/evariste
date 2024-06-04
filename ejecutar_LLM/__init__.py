from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from traduccion_sql_ln import *
from parser_SQL import *
from embeddings import *
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
    model="llama2-uncensored"
)

# Configuraciones para hacer las preguntas
system_prompt=("You are a highly intelligent question answering bot. "
               "If I ask you a question that is rooted in truth, you will give you the answer. "
               "If I ask you a question that is nonsense, trickery, or has no clear answer, you will respond with 'Unknown'. "
               "You will answer concisely. "
               "Use the given context as a support to answer the question if you can't answer the question."
               "Context: {context}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Answer the query.\n{format_instructions}\n{question}\n"),
    ],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def crear_instrucciones(columnas: list[str]):
    texto = "Format the information as a table with columns for "
    if len(columnas) > 0:
        texto += columnas.pop(0)
    
    for _ in range(len(columnas) -1):
        texto += f", {columnas.pop(0)}"
    
    texto += f"and {columnas.pop(0)}."

    texto += " Your response should be a table"
    
    return (lambda *args: texto)

def hacer_consulta(traduccion: str, columnas: list[str]):
    rag_chain = (
        {"context": retriever | format_docs, 
        "question": RunnablePassthrough(),
        "format_instructions": crear_instrucciones(columnas)}
        | prompt
        | ollama
        | StrOutputParser()
    )

    return rag_chain.invoke(traduccion)