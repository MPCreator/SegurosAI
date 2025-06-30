from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os
from flask_cors import CORS
from collections import defaultdict

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
        for emb_info in item.get("embeddings", []):
            emb = emb_info["embedding"]
            sim = cosine_similarity([consulta_emb], [emb])[0][0]
            similitudes.append((sim, item, emb_info["texto"]))

    if not similitudes:
        return None

    similitudes.sort(reverse=True, key=lambda x: x[0])
    mejor_score, mejor_item, texto_match = similitudes[0]

    if mejor_score >= umbral:
        return {
            "similitud": mejor_score,
            "pregunta": mejor_item["pregunta"],
            "respuesta": mejor_item["respuesta"],
            "texto_coincidente": texto_match,
            "contexto": mejor_item.get("contexto", mejor_item["respuesta"])
        }
    else:
        return None

def clasificar_consulta(texto):
    prompt = f"""
    Clasifica la siguiente pregunta como "general" o "específica":

    - General: si es un saludo o una duda amplia que no requiere detalles como montos, coberturas o procedimientos específicos.
    - Específica: si está pidiendo información concreta que normalmente vendría de una base de datos o documento oficial.

    Pregunta:
    "{texto}"

    Solo responde con una palabra: general o específica.
    """
    try:
        respuesta = client.generate_content(prompt).text.strip().lower()
        return "específica" if "específic" in respuesta else "general"
    except Exception:
        return "específica"  # fallback de seguridad

def generar_respuesta(prompt):
    response = client.generate_content(prompt)
    return response.text

def quitar_urls_duplicadas(texto: str) -> str:
    import re

    texto = re.sub(
        r'\[(https?://[^\s\]]+)\]\(\1\)', 
        r'\1',
        texto
    )

    pattern = re.compile(
        r'\[([^\]]+)\]\((https?://[^\s)]+)\)'  
        r'|(?P<bare>https?://[^\s)]+)'         
    )

    seen = set()
    resultado = []
    last_end = 0

    for m in pattern.finditer(texto):
        url = m.group(2) or m.group('bare')
        raw_match = texto[m.start():m.end()]
        resultado.append(texto[last_end:m.start()])

        if url not in seen:
            resultado.append(raw_match)
            seen.add(url)

        last_end = m.end()

    resultado.append(texto[last_end:])
    return ''.join(resultado)


# --- FLASK APP ---
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    mensaje = data.get("mensaje", "").strip()
    if not mensaje:
        return jsonify({"error": "Mensaje vacío"}), 400

    resultado = buscar_similitud(mensaje)

    if resultado:
        prompt = f"""
        Eres un asistente especializado en atención al afiliado de un plan de salud privado del Grupo San Pablo. Tu función es resolver dudas de manera clara, precisa y empática, utilizando la información proporcionada. Prioriza siempre ayudar al usuario, sin inventar datos que no estén presentes.

        Pregunta del usuario:
        \"{mensaje}\"

        Información relacionada encontrada:
        \"{resultado['contexto']}\"

        Responde con claridad, como si fueras parte del equipo de soporte oficial de San Pablo.
        """
    else:
        tipo_consulta = clasificar_consulta(mensaje)

        if tipo_consulta == "específica":
            prompt = f"""
            Eres un asistente especializado en atención al afiliado de un plan de salud privado del Grupo San Pablo. El usuario hizo una consulta que requiere datos específicos, pero no tienes información suficiente para responder con precisión.

            Indica de forma clara y empática que debe comunicarse con atención al afiliado al número (01) 610 3232 para obtener ayuda directa.

            Pregunta del usuario:
            \"{mensaje}\"

            Sé breve, amable y profesional.
            """
        else:
            prompt = f"""
            Eres un asistente especializado en atención al afiliado de un plan de salud privado del Grupo San Pablo. Tu función es resolver dudas generales de manera clara, cordial y profesional. No inventes datos ni proporciones montos, pero puedes responder saludos y dudas comunes.

            Pregunta del usuario:
            \"{mensaje}\"

            Responde de forma amigable y útil, sin mencionar el número de atención.
            """

    respuesta = generar_respuesta(prompt)
    print(f"🤖 Respuesta generada: {respuesta}")
    respuesta = quitar_urls_duplicadas(respuesta)
    print(f"🔗 URLs eliminadas: {respuesta}")
    return jsonify({
        "respuesta": respuesta,
        "usó_contexto": bool(resultado),
        "similitud": resultado["similitud"] if resultado else None
    })

@app.route("/")
def home():
    return send_from_directory(".", "chat.html")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    app.run(debug=True)
