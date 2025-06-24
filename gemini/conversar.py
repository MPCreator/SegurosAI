import requests

API_URL = "http://localhost:5000/chat"

def main():
    print("🧠 Chat con Gemini + Embeddings (Ctrl+C para salir)\n")
    
    while True:
        try:
            mensaje = input("👤 Tú: ").strip()
            if not mensaje:
                continue
            
            response = requests.post(API_URL, json={"mensaje": mensaje})
            
            if response.status_code == 200:
                data = response.json()
                respuesta = data.get("respuesta", "🤖 (Sin respuesta)")
                contexto = data.get("usó_contexto", False)
                similitud = data.get("similitud")
                
                print(f"🤖 Gemini: {respuesta}")
                if contexto:
                    print(f"   📎 (Usó contexto, similitud: {similitud:.2f})")
                else:
                    print("   ✨ (Respuesta generada sin coincidencia previa)")
            else:
                print("⚠️ Error:", response.status_code, response.text)

        except KeyboardInterrupt:
            print("\n👋 Hasta luego!")
            break
        except Exception as e:
            print("⚠️ Error inesperado:", str(e))

if __name__ == "__main__":
    main()
