from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --- CARGAR VARIABLES DEL .env ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING")
MODEL_CHAT = os.getenv("MODEL_CHAT")

# --- INICIALIZAR GEMINI ---
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel(MODEL_CHAT)

# --- CARGAR EMBEDDINGS ---
with open("qa_embeddings_final.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# --- FUNCIONES AUXILIARES ---
def embed_user_query(text: str):
    result = genai.embed_content(
        model=MODEL_EMBEDDING,
        content=text,
        task_type="RETRIEVAL_QUERY"
    )
    return result["embedding"]

def buscar_similitud(consulta, umbral=0.80):
    consulta_emb = embed_user_query(consulta)
    similitudes = []
    for item in qa_data:
        sim = cosine_similarity([consulta_emb], [item["embedding"]])[0][0]
        similitudes.append((sim, item))
    
    if not similitudes:
        return None
    
    similitudes.sort(reverse=True, key=lambda x: x[0])
    mejor_score, mejor_item = similitudes[0]

    if mejor_score >= umbral:
        return {
            "similitud": mejor_score,
            "pregunta": mejor_item["pregunta"],
            "respuesta": mejor_item["respuesta"],
            "contexto": mejor_item.get("contexto", mejor_item["respuesta"])  # opcional
        }
    else:
        return None

def generar_respuesta(prompt):
    response = client.generate_content(prompt)
    return response.text

# --- FLASK APP ---
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    mensaje = data.get("mensaje", "").strip()
    if not mensaje:
        return jsonify({"error": "Mensaje vacío"}), 400

    resultado = buscar_similitud(mensaje)

    if resultado:
        prompt = f"""Eres un consultor de seguros profesional. Un usuario ha hecho la siguiente consulta:
"{mensaje}"

Usa esta información como contexto y responde de forma clara y concisa:

"{resultado['contexto']}"
"""
    else:
        prompt = f"""Eres un consultor de seguros profesional.Responde de manera breve, clara y precisa a la siguiente pregunta del usuario:
"{mensaje}"
"""

    respuesta = generar_respuesta(prompt)
    respuesta = quitar_urls_duplicadas(respuesta)
    return jsonify({
        "respuesta": respuesta,
        "usó_contexto": bool(resultado),
        "similitud": resultado["similitud"] if resultado else None
    })



@app.route("/")
def home():
    return send_from_directory(".", "chat.html")

def quitar_urls_duplicadas(texto):
    urls = re.findall(r'https?://\S+', texto)
    urls_vistos = set()
    resultado = []
    for linea in texto.split('\n'):
        nueva_linea = linea
        for url in urls:
            if url in nueva_linea:
                if url in urls_vistos:
                    nueva_linea = nueva_linea.replace(url, '')
                else:
                    urls_vistos.add(url)
        if nueva_linea.strip():
            resultado.append(nueva_linea.strip())
    return '\n'.join(resultado).strip()




# --- EJECUCIÓN ---
if __name__ == "__main__":
    app.run(debug=True)


