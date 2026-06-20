import os
import json
import asyncio
from groq import Groq
from collections import defaultdict

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HISTORICO_FILE = "historico.json"

client = Groq(api_key=GROQ_API_KEY)

MODELOS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

# Estrutura: { user_id: { "geral": [...], "trabalho": [...], "estudos": [...] } }
historico = defaultdict(lambda: defaultdict(list))

# ================= CARREGAR / SALVAR =================
def carregar_historico():
    if os.path.exists(HISTORICO_FILE):
        try:
            with open(HISTORICO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return defaultdict(lambda: defaultdict(list), 
                                 {int(k): defaultdict(list, v) for k, v in data.items()})
        except:
            pass
    return defaultdict(lambda: defaultdict(list))


def salvar_historico():
    try:
        data = {str(k): dict(v) for k, v in historico.items()}
        with open(HISTORICO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        pass


historico = carregar_historico()

# ================= DETECTAR TÓPICO =================
async def detectar_topico(texto: str) -> str:
    try:
        prompt = f"""
        Analise a mensagem abaixo e responda APENAS com o tópico principal em uma palavra ou duas.
        Exemplos: trabalho, estudos, saúde, finanças, projetos, pessoal, entretenimento, etc.
        Se não tiver um tópico claro, responda "geral".

        Mensagem: {texto}
        Tópico:
        """

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=15
        )
        topico = res.choices[0].message.content.strip().lower()
        return topico if topico else "geral"
    except:
        return "geral"


# ================= GERAR RESPOSTA =================
async def gerar_resposta(texto: str, user_id: int) -> str:
    if not texto or not texto.strip():
        return ""

    user_id = int(user_id)
    topico = await detectar_topico(texto)

    # Adiciona no tópico correspondente
    historico[user_id][topico].append({"role": "user", "content": texto})

    # Limita cada tópico em 8 mensagens
    if len(historico[user_id][topico]) > 8:
        historico[user_id][topico] = historico[user_id][topico][-8:]

    # Monta o contexto (geral + tópico atual)
    mensagens = historico[user_id]["geral"][-4:] + historico[user_id][topico]

    for model in MODELOS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Você é eu. Responda de forma natural.
                        Estamos falando sobre o tópico: {topico}.
                        Use as informações salvas nesse tópico quando relevante."""
                    },
                    *mensagens
                ],
                temperature=0.75,
                max_tokens=900
            )

            resposta = completion.choices[0].message.content.strip()

            historico[user_id][topico].append({"role": "assistant", "content": resposta})

            # Salva em segundo plano
            asyncio.create_task(asyncio.to_thread(salvar_historico))

            return resposta

        except Exception as e:
            print(f"Erro com modelo {model}: {e}")
            continue

    return "Não consegui processar isso agora."