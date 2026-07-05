import os
from dotenv import load_dotenv
load_dotenv()
import re
import io
import json
import time
import threading
import platform
import fitz
import chromadb
import streamlit as st
import pytesseract
from PIL import Image
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class ProcesamientoCancelado(Exception):
    pass


# ─────────────────────────────────────────
# CONFIG  (secrets en Streamlit Cloud, .env en local)
# ─────────────────────────────────────────
def _get_secret(key: str, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
CHROMA_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "puce_normativa"
ADMIN_PASSWORD  = _get_secret("ADMIN_PASSWORD")

# Tesseract: en Windows hay que apuntar al .exe; en Linux (Streamlit Cloud) está en el PATH.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Reglamentos PUCE", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    /* ─── HEADER ─── */
    .titulo {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #00d4aa 0%, #00a8d4 50%, #0078d4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0;
    }
    .subtitulo { color: #888; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.5rem; }

    /* ─── BADGES ─── */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-doc { background: rgba(0, 212, 170, 0.15); color: #00d4aa; border: 1px solid rgba(0, 212, 170, 0.3); }
    .badge-art { background: rgba(0, 168, 212, 0.15); color: #00a8d4; border: 1px solid rgba(0, 168, 212, 0.3); }
    .badge-time { background: rgba(255, 193, 7, 0.15); color: #ffc107; border: 1px solid rgba(255, 193, 7, 0.3); }

    /* ─── QUICK PROMPT CARDS ─── */
    .qp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 12px 0; }
    .qp-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 10px; padding: 14px;
        transition: all 0.2s ease;
    }
    .qp-card:hover {
        border-color: #00d4aa;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.15);
    }
    .qp-icon { font-size: 1.5rem; margin-bottom: 6px; }
    .qp-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }
    .qp-sub { color: #888; font-size: 0.8rem; }

    /* ─── STEP BOX ─── */
    .step-box {
        background: var(--secondary-background-color); border-left: 3px solid #00d4aa;
        padding: 10px 15px; border-radius: 4px; margin: 6px 0; font-size: 0.9rem;
    }
    .step-done { border-left-color: #28a745; }
    .step-active { border-left-color: #ffc107; }
    .step-ocr { border-left-color: #ff7b00; }

    /* ─── METRIC CARD ─── */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(0, 168, 212, 0.08) 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 10px; padding: 14px; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00d4aa; line-height: 1; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

    /* ─── HERO WELCOME ─── */
    .hero-welcome {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.05) 0%, rgba(0, 120, 212, 0.05) 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 14px; padding: 20px; margin-bottom: 16px;
    }

    /* ─── ANIMATIONS ─── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .pulsing { animation: pulse 1.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """Eres un asistente legal experto de los Reglamentos de la PUCE. Respondes con precisión, exhaustividad y claridad.

REGLAS DE CONTENIDO:
1. Usa ÚNICAMENTE la información de los fragmentos proporcionados.
2. Si la pregunta pide artículos sobre un tema, LISTA TODOS los artículos relevantes encontrados en los fragmentos, incluyendo de DISTINTOS documentos si los hay. No omitas ninguno.
3. Si la pregunta es sobre un artículo específico, reproduce su contenido relevante de forma estructurada y completa.
3b. Si el número de artículo solicitado aparece en MÁS DE UN documento (lo verás porque hay fragmentos con el mismo "Artículo N" y distinto "Documento"), reproduce el contenido de CADA documento por separado, con un encabezado por documento. NUNCA elijas solo uno.
4. Si hay numerales o listas en el reglamento, preséntalos como lista.
5. Si la información NO está en los fragmentos, responde EXACTAMENTE: "No encontré información sobre eso en los reglamentos cargados."
6. NO inventes, NO supongas, NO uses conocimiento externo.
7. Mantén coherencia con la conversación previa.

REGLAS DE CITACIÓN (OBLIGATORIAS):
8. SIEMPRE cita el origen al final de cada afirmación con el formato: **[Art. N - Título del artículo - nombre_documento]**
9. Si combinas información de múltiples artículos o documentos, cita CADA UNO explícitamente.
10. NUNCA omitas la cita del documento, especialmente cuando hay varios reglamentos cargados.
11. Si dos artículos del mismo número vienen de documentos distintos, distínguelos.

REGLAS DE FORMATO:
12. NO incluyas frases como "Artículo reformado por resolución..." en tu respuesta.
13. Estructura las respuestas largas con encabezados o listas para fácil lectura.
14. Sé directo: empieza con la respuesta, no con introducciones largas.

CONVERSACIÓN PREVIA (para contexto):
{historial}

FRAGMENTOS DEL REGLAMENTO:
{context}

PREGUNTA ACTUAL: {question}

RESPUESTA:"""


def exportar_conversacion_md(historial: list) -> str:
    """Convierte el historial actual a Markdown descargable."""
    lineas = [
        "# Conversación · Reglamentos PUCE",
        f"_Exportado: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "---",
        ""
    ]
    for m in historial:
        rol = "🧑 Usuario" if m["role"] == "user" else "🤖 Asistente"
        lineas.append(f"### {rol}\n\n{m['content']}\n")
    return "\n".join(lineas)


# ─────────────────────────────────────────
# QUICK PROMPTS (preguntas sugeridas para el usuario)
# ─────────────────────────────────────────
QUICK_PROMPTS = [
    {"icon": "💰", "title": "Becas y ayudas",       "sub": "Tipos, requisitos y montos",       "q": "¿Qué tipos de becas existen y qué requisitos tienen?"},
    {"icon": "📉", "title": "Reprobar materias",    "sub": "Tercera matrícula y consecuencias", "q": "¿Qué pasa si repruebo una materia tres veces?"},
    {"icon": "🎓", "title": "Titulación",           "sub": "Modalidades y requisitos",          "q": "¿Cómo me puedo titular? ¿Qué modalidades existen?"},
    {"icon": "📝", "title": "Matrícula",            "sub": "Plazos y procedimiento",            "q": "¿Cuál es el procedimiento de matrícula y los plazos?"},
    {"icon": "⚖️", "title": "Régimen disciplinario","sub": "Faltas y sanciones",                "q": "¿Cuáles son las faltas disciplinarias y sus sanciones?"},
    {"icon": "♿", "title": "Inclusión",             "sub": "Necesidades específicas",           "q": "¿Qué dice el reglamento sobre estudiantes con discapacidad?"},
]


# ─────────────────────────────────────────
# RECURSOS CACHEADOS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Iniciando base de datos...")
def get_chroma_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"batch_size": 32}
        )
    except Exception:
        # Si no está en caché local, descarga normalmente
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32}
        )


# ─────────────────────────────────────────
# BM25 — SIN @st.cache_resource  (FIX CRÍTICO 1)
# ─────────────────────────────────────────
def construir_bm25(chunks):
    if not chunks:
        return None
    tokenized = [re.findall(r'\b\w+\b', t.lower()) for t in chunks]
    return BM25Okapi(tokenized)


# ─────────────────────────────────────────
# HELPERS DE BASE DE DATOS
# ─────────────────────────────────────────
def db_tiene_datos():
    try:
        col = get_chroma_client().get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False

def cargar_vectorstore():
    return Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings()
    )


# ─────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────
_PATRON_LINEA_BASURA = re.compile(
    r"^\s*("
    r"pontificia universidad.*"
    r"|ser[eé]is mis testigos.*"
    r"|reglamento general de\s*$"
    r"|reglamento espec[ií]fico.*"
    r"|becas y ayudas econ[oó]micas.*"
    r"|para carreras y programas.*"
    r"|estudiantes\s*$"
    r"|nivel de confidencialidad.*"
    r"|c[oó]digo:\s*[A-Z]{2}-.*"
    r"|vigencia:.*"
    r"|versi[oó]n:\s*\d.*"
    r"|p[aá]g\.?\s*:?\s*\d+\s*de\s*\d+.*"
    r"|derechos reservados.*"
    r"|jesuitas ecuador.*"
    r"|©\s*\d{4}.*"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE
)
_PATRON_RUIDO_LEGAL = re.compile(
    r"Art[ií]culo\s+(reformado|incorporado|sustituido|derogado|modificado)\s+(por|con|mediante)\s+resoluci[oó]n[^.]*\.\s*",
    re.IGNORECASE
)
_PATRON_DISPOSICION_RUIDO = re.compile(
    r"Disposici[oó]n\s+(reformada|incorporada|transitoria\s+incorporada)\s+(por|con|mediante)\s+resoluci[oó]n[^.]*\.\s*",
    re.IGNORECASE
)

_PATRON_CORTE = re.compile(
    r'(?=(?:^|\n)[ \t]*(?:'
    r'Art[ií]culo\s+\d+\s*[\.\-–]'
    r'|DISPOSICI[OÓ]N(?:ES)?\s+(?:GENERAL(?:ES)?|TRANSITORIA(?:S)?|DEROGATORIA(?:S)?|FINAL)'
    r'))'
)
_ES_ARTICULO   = re.compile(r'^Art[ií]culo\s+\d+\s*[\.\-–]')
_ES_DISPOSICION = re.compile(
    r'^DISPOSICI[OÓ]N(?:ES)?\s+(GENERAL(?:ES)?|TRANSITORIA(?:S)?|DEROGATORIA(?:S)?|FINAL)',
    re.IGNORECASE)


def limpiar_lineas(texto: str) -> str:
    texto = _PATRON_LINEA_BASURA.sub("", texto)
    texto = _PATRON_RUIDO_LEGAL.sub("", texto)
    texto = _PATRON_DISPOSICION_RUIDO.sub("", texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


# ─────────────────────────────────────────
# EXTRACCIÓN HÍBRIDA: TEXTO NATIVO + OCR FALLBACK
# ─────────────────────────────────────────
def _rect_dentro(bbox_bloque, rect_tabla, umbral=0.6):
    """True si el bloque cae mayormente (≥umbral) dentro de la tabla → es texto duplicado de celda."""
    rb = fitz.Rect(bbox_bloque)
    rb.normalize()
    inter = rb & rect_tabla
    area_b = rb.width * rb.height
    if inter.is_empty or area_b <= 0:
        return False
    area_i = inter.width * inter.height
    return (area_i / area_b) >= umbral


def extraer_texto_pagina_nativo(pagina) -> str:
    # 1) Detectar tablas: sus regiones y su versión en Markdown
    try:
        tablas = pagina.find_tables().tables
    except Exception:
        tablas = []
    rects_tabla = [fitz.Rect(t.bbox) for t in tablas]
    md_tablas   = [t.to_markdown() for t in tablas]
    emitidas    = [False] * len(tablas)

    partes = []
    for bloque in pagina.get_text("blocks"):
        if bloque[6] != 0:          # solo bloques de texto
            continue
        linea = bloque[4].strip()
        if not linea:
            continue

        # 2) ¿El bloque pertenece a alguna tabla?
        idx = next(
            (i for i, rt in enumerate(rects_tabla) if _rect_dentro(bloque[:4], rt)),
            None
        )
        if idx is not None:
            # Insertar la tabla UNA vez (en su posición de lectura) y omitir el texto suelto
            if not emitidas[idx]:
                md = md_tablas[idx].strip()
                if md:
                    partes.append(md)
                emitidas[idx] = True
            continue                # se descarta el texto plano duplicado de la celda

        partes.append(linea)

    # 3) Defensa: si alguna tabla no cruzó con ningún bloque, añadirla al final
    for i, ok in enumerate(emitidas):
        if not ok and md_tablas[i].strip():
            partes.append(md_tablas[i].strip())

    return "\n".join(partes)


def extraer_texto_pagina_ocr(pagina) -> str:
    matriz = fitz.Matrix(300/72, 300/72)
    pix = pagina.get_pixmap(matrix=matriz, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    texto = pytesseract.image_to_string(img, lang="spa", config="--psm 6")
    return texto.strip()


def extraer_texto_completo_pdf(doc_pdf, log_callback=None, cancel_check=None) -> tuple[str, int]:
    """Extrae texto página a página (texto nativo si existe, OCR si no)."""
    paginas = []
    paginas_ocr = 0

    for i, pagina in enumerate(doc_pdf):
        if cancel_check and cancel_check():
            raise ProcesamientoCancelado()

        texto_nativo = extraer_texto_pagina_nativo(pagina)
        chars_utiles = len(re.sub(r'\s+', '', texto_nativo))

        if chars_utiles < 50:
            if log_callback:
                log_callback(f"🔍 OCR en página {i+1} (escaneada)...")
            try:
                texto = extraer_texto_pagina_ocr(pagina)
                paginas_ocr += 1
            except Exception as e:
                if log_callback:
                    log_callback(f"⚠️ Error OCR pág. {i+1}: {e}")
                texto = texto_nativo
        else:
            texto = texto_nativo

        texto_limpio = limpiar_lineas(texto)
        if texto_limpio:
            paginas.append(texto_limpio)

    return "\n\n".join(paginas), paginas_ocr


# ─────────────────────────────────────────
# DIVISIÓN POR ARTÍCULOS
# ─────────────────────────────────────────
_PATRON_ARTICULO = re.compile(
    r'(?=(?:^|\n)[ \t]*Art[ií]culo\s+\d+\s*[\.\-–])'
)


def recortar_a_cuerpo_normativo(texto: str) -> str:
    """
    Defensa adicional: descarta el preámbulo/CONSIDERANDO del reglamento.
    """
    m = re.search(
        r'(?:^|\n)[ \t]*Art[ií]culo\s+1\s*[\.\-–]\s*[–-]?\s*[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
        texto
    )
    if m:
        return texto[m.start():]
    return texto


def dividir_en_articulos(texto: str, nombre_doc: str) -> list[Document]:
    # 1) Descartar preámbulo/CONSIDERANDO
    texto = recortar_a_cuerpo_normativo(texto)

    # 2) Dividir por artículos Y disposiciones
    partes = _PATRON_CORTE.split(texto)
    documentos = []

    for parte in partes:
        parte = parte.strip()
        if not parte or len(parte) < 50:
            continue

        if _ES_ARTICULO.match(parte):
            m = re.match(
                r'Art[ií]culo\s+(\d+)\s*[\.\-–]\s*[–-]?\s*([^\n.]{2,120})',
                parte
            )
            if m:
                num_art = m.group(1)
                titulo_art = m.group(2).strip().rstrip('.').strip()
                titulo_art = re.sub(r'^[\-–\s]+', '', titulo_art)
            else:
                m2 = re.match(r'Art[ií]culo\s+(\d+)', parte)
                num_art = m2.group(1) if m2 else "?"
                titulo_art = ""

            meta_base = {
                "source": nombre_doc,
                "articulo": num_art,
                "titulo": titulo_art[:150]
            }

            if len(parte) > 2500:
                encabezado = parte[:200]
                sub_chunks = [parte[i:i+2000] for i in range(0, len(parte), 1800)]
                for j, sub in enumerate(sub_chunks):
                    contenido = sub if j == 0 else f"[{encabezado[:80]}...]\n{sub}"
                    meta = {**meta_base, "sub_chunk": j}
                    documentos.append(Document(page_content=contenido, metadata=meta))
            else:
                documentos.append(Document(page_content=parte, metadata=meta_base))

        elif _ES_DISPOSICION.match(parte):
            tipo_disp = _ES_DISPOSICION.match(parte).group(1).capitalize()
            meta = {
                "source": nombre_doc,
                "articulo": "—",
                "titulo": f"Disposiciones {tipo_disp}"
            }
            documentos.append(Document(page_content=parte, metadata=meta))

        else:
            continue

    return documentos


# ─────────────────────────────────────────
# CONTEXTUALIZACIÓN
# ─────────────────────────────────────────
def contextualizar_pregunta(pregunta: str, historial: list, llm) -> str:
    if not historial:
        return pregunta

    historial_reciente = historial[-6:]
    historial_texto = "\n".join([
        f"{'Usuario' if m['role']=='user' else 'Asistente'}: {m['content'][:400]}"
        for m in historial_reciente
    ])

    template = """Dada la conversación y una pregunta de seguimiento, reformula la pregunta para que sea ENTENDIBLE POR SÍ SOLA.

REGLAS:
- Si la pregunta YA es independiente, devuélvela TAL CUAL.
- Si es referencial (ej: "y el 3?", "más sobre eso", "explica más", "el siguiente", "y en el otro reglamento?"), reescríbela con contexto.
- NO respondas la pregunta, solo reformúlala.
- Devuelve SOLO la pregunta reformulada, sin comillas, sin prefijos.

HISTORIAL:
{historial}

PREGUNTA DE SEGUIMIENTO: {pregunta}

PREGUNTA REFORMULADA:"""

    chain = PromptTemplate(template=template, input_variables=["historial", "pregunta"]) | llm | StrOutputParser()
    try:
        reformulada = chain.invoke({"historial": historial_texto, "pregunta": pregunta}).strip()
        reformulada = reformulada.strip('"\'').strip()
        if len(reformulada) < 3:
            return pregunta
        return reformulada
    except Exception:
        return pregunta


# ─────────────────────────────────────────
# DETECTAR SALUDOS / CHARLA
# ─────────────────────────────────────────
_PATRON_SALUDO = re.compile(
    r"^\s*(hola|holi|buenas|buen[oa]s? d[ií]as|buenas tardes|buenas noches|hey|qué tal|que tal|gracias|ok|listo|perfecto|chao|adi[oó]s|bye)\s*[!.?]*\s*$",
    re.IGNORECASE
)

def es_charla_no_consulta(pregunta: str) -> bool:
    return bool(_PATRON_SALUDO.match(pregunta.strip()))


# ─────────────────────────────────────────
# DETECCIÓN DE META-PREGUNTAS
# ─────────────────────────────────────────
_PATRON_CONTAR_ARTS = re.compile(
    r'(cu[aá]ntos?\s+art[ií]culos?'
    r'|n[uú]mero\s+(total\s+)?de\s+art[ií]culos?'
    r'|total\s+de\s+art[ií]culos?'
    r'|qu[eé]\s+art[ií]culos?\s+(tiene|hay|contiene))',
    re.IGNORECASE
)
_PATRON_CONTAR_DOCS = re.compile(
    r'(cu[aá]ntos?\s+(reglamentos?|documentos?|pdfs?|archivos?)'
    r'|qu[eé]\s+(reglamentos?|documentos?)\s+(hay|tienes|tengo|est[aá]n cargados?))',
    re.IGNORECASE
)
_PATRON_LISTAR_ARTS = re.compile(
    r'(lista|list[aá]me|enum[eé]rame|enumera|listado|d[aá]me\s+(la\s+)?lista|todos\s+los?)'
    r'\s+(los?\s+)?art[ií]culos?',
    re.IGNORECASE
)
_PATRON_LISTAR_DOCS = re.compile(
    r'(lista|list[aá]me|enum[eé]rame|enumera|qu[eé]|cu[aá]les)'
    r'\s+(son\s+)?(los?\s+|las?\s+)?(reglamentos?|documentos?|pdfs?|archivos?)',
    re.IGNORECASE
)
_PATRON_INDICE = re.compile(
    r'(tabla\s+de\s+contenidos?|[ií]ndice|estructura\s+(del|de\s+los)|cap[ií]tulos?\s+(tiene|hay)|t[ií]tulos?\s+(tiene|hay))',
    re.IGNORECASE
)


def detectar_meta_pregunta(pregunta: str):
    p = pregunta.strip()
    if _PATRON_LISTAR_ARTS.search(p):
        return "listar_articulos"
    if _PATRON_INDICE.search(p):
        return "indice"
    if _PATRON_CONTAR_ARTS.search(p):
        return "contar_articulos"
    if _PATRON_LISTAR_DOCS.search(p):
        return "listar_documentos"
    if _PATRON_CONTAR_DOCS.search(p):
        return "contar_documentos"
    return None


def detectar_doc_filtro(pregunta: str, documentos_disponibles: list) -> str:
    p = pregunta.lower()

    for d in documentos_disponibles:
        if d.lower() in p:
            return d

    es_becas   = any(k in p for k in ["beca", "ayuda econ", "ayudas econ", "fopedeupo"])
    es_general = ("general" in p) or ("estudiante" in p and not es_becas)

    if es_becas:
        for d in documentos_disponibles:
            if "beca" in d.lower():
                return d
    if es_general:
        for d in documentos_disponibles:
            if "general" in d.lower() or "estudiante" in d.lower():
                if "beca" not in d.lower():
                    return d
    return ""


def _agrupar_arts_por_doc(chunks_fuentes, chunks_arts, chunks_titulos, indices=None):
    if indices is None:
        indices = list(range(len(chunks_fuentes)))

    por_doc: dict = {}
    for i in indices:
        doc = chunks_fuentes[i]
        art = chunks_arts[i]
        tit = chunks_titulos[i] or ""
        if art == "?" or not art or art == "—":
            continue
        por_doc.setdefault(doc, {})
        if art not in por_doc[doc] or (not por_doc[doc][art] and tit):
            por_doc[doc][art] = tit
    return por_doc


def _ordenar_arts(arts_dict):
    try:
        return sorted(arts_dict.items(), key=lambda x: int(x[0]))
    except (ValueError, TypeError):
        return sorted(arts_dict.items())


def responder_meta_pregunta(tipo, chunks_fuentes, chunks_arts, chunks_titulos, doc_filtro=""):
    if doc_filtro:
        idx = [i for i, f in enumerate(chunks_fuentes) if f == doc_filtro]
        if not idx:
            return f"No encontré el documento **{doc_filtro}** en los reglamentos cargados."
    else:
        idx = list(range(len(chunks_fuentes)))

    por_doc = _agrupar_arts_por_doc(chunks_fuentes, chunks_arts, chunks_titulos, idx)

    if not por_doc:
        return "No hay artículos identificables en los reglamentos cargados."

    if tipo == "contar_articulos":
        if len(por_doc) == 1:
            doc, arts = list(por_doc.items())[0]
            try:
                nums = sorted(int(a) for a in arts.keys())
                rango = f" (Art. {nums[0]} al Art. {nums[-1]})"
                faltan = [n for n in range(nums[0], nums[-1] + 1) if n not in nums]
                aviso = f"\n\n⚠️ Posibles artículos no detectados: {faltan}" if faltan else ""
            except ValueError:
                rango, aviso = "", ""
            return f"El reglamento **{doc}** tiene **{len(arts)} artículos**{rango}.{aviso}"

        partes = ["**Número de artículos por reglamento:**\n"]
        for doc in sorted(por_doc.keys()):
            arts = por_doc[doc]
            try:
                nums = sorted(int(a) for a in arts.keys())
                rango = f" (Art. {nums[0]} al Art. {nums[-1]})"
            except ValueError:
                rango = ""
            partes.append(f"- **{doc}** — {len(arts)} artículos{rango}")
        return "\n".join(partes)

    if tipo == "contar_documentos":
        docs = sorted(set(chunks_fuentes))
        partes = [f"Hay **{len(docs)} reglamento(s)** cargado(s):\n"]
        for d in docs:
            arts_unicos = len(por_doc.get(d, {}))
            partes.append(f"- **{d}** — {arts_unicos} artículos")
        return "\n".join(partes)

    if tipo == "listar_documentos":
        docs = sorted(set(chunks_fuentes))
        partes = [f"**Reglamentos cargados ({len(docs)}):**\n"]
        for d in docs:
            arts_unicos = len(por_doc.get(d, {}))
            partes.append(f"- 📚 **{d}** — {arts_unicos} artículos")
        return "\n".join(partes)

    if tipo == "listar_articulos":
        partes = []
        for doc in sorted(por_doc.keys()):
            arts = por_doc[doc]
            items_ord = _ordenar_arts(arts)
            partes.append(f"\n### 📚 {doc}  —  {len(arts)} artículos\n")
            for num, tit in items_ord:
                partes.append(f"- **Art. {num}** — {tit}" if tit else f"- **Art. {num}**")
        return "\n".join(partes).strip()

    if tipo == "indice":
        partes = ["## 📑 Índice de los reglamentos\n"]
        for doc in sorted(por_doc.keys()):
            arts = por_doc[doc]
            items_ord = _ordenar_arts(arts)
            partes.append(f"\n### {doc}  —  {len(arts)} artículos")
            for num, tit in items_ord:
                partes.append(f"- Art. {num} — {tit}" if tit else f"- Art. {num}")
        return "\n".join(partes)

    return "No pude procesar esa meta-pregunta."


# ─────────────────────────────────────────
# ROUTER DE INTENCIÓN (LLM)
# ─────────────────────────────────────────
def clasificar_intencion(pregunta: str, llm, documentos_disponibles: list = None) -> dict:
    docs_str = ", ".join(documentos_disponibles) if documentos_disponibles else "(ninguno)"

    template = """Analiza esta pregunta sobre los reglamentos de la PUCE y devuelve SOLO JSON válido (sin markdown).

DOCUMENTOS DISPONIBLES: {docs}

Pregunta: "{pregunta}"

Devuelve EXACTAMENTE:
{{"tipo":"articulo|tema|comparativa|saludo|ambigua",
  "articulos":[lista_numeros],
  "temas":["concepto","sinonimo1","sinonimo2"],
  "documento_filtro":"nombre_doc_si_lo_menciona_o_vacio",
  "reformulada":"pregunta clara"}}

Reglas:
- "articulos": números si menciona artículos específicos, vacío si no
- "temas": expandir con sinónimos en español. Para becas incluir: beca, ayuda económica, subvención, FOPEDEUPO. Para discapacidad: discapacidad, inclusión, necesidades específicas. Etc.
- "documento_filtro": SOLO si el usuario menciona un documento específico ("en el reglamento de becas...", "del reglamento general..."), si no, vacío
- "tipo": "comparativa" si pide comparar/diferenciar dos cosas; "saludo" si es solo saludo

Ejemplos:
- "art 29" → {{"tipo":"articulo","articulos":[29],"temas":[],"documento_filtro":"","reformulada":"contenido del artículo 29"}}
- "Art. 7 y 11" → {{"tipo":"articulo","articulos":[7,11],"temas":[],"documento_filtro":"","reformulada":"artículos 7 y 11"}}
- "diferencia entre art 28 y 29" → {{"tipo":"comparativa","articulos":[28,29],"temas":[],"documento_filtro":"","reformulada":"diferencias entre artículo 28 y artículo 29"}}
- "becas" → {{"tipo":"tema","articulos":[],"temas":["beca","ayuda económica","subvención","FOPEDEUPO","categoría"],"documento_filtro":"","reformulada":"categorías y tipos de becas"}}
- "puedo perder mi beca?" → {{"tipo":"tema","articulos":[],"temas":["pérdida de beca","mantener","causales","obligaciones","rendimiento"],"documento_filtro":"","reformulada":"causales de pérdida de beca"}}
- "qué dice el reglamento de becas sobre pérdida" → {{"tipo":"tema","articulos":[],"temas":["pérdida","beca","causales"],"documento_filtro":"becas","reformulada":"causales de pérdida de beca según reglamento de becas"}}
- "qué pasa si repruebo 3 veces" → {{"tipo":"tema","articulos":[],"temas":["tercera matrícula","reprobar","repetir"],"documento_filtro":"","reformulada":"consecuencias de reprobar tres veces"}}
- "como me titulo" → {{"tipo":"tema","articulos":[],"temas":["titulación","graduación","integración curricular","examen complexivo"],"documento_filtro":"","reformulada":"requisitos de titulación"}}

JSON:"""

    chain = PromptTemplate(template=template, input_variables=["pregunta", "docs"]) | llm | StrOutputParser()
    try:
        respuesta = chain.invoke({"pregunta": pregunta, "docs": docs_str})
        respuesta = re.sub(r'```json|```', '', respuesta).strip()
        m = re.search(r'\{.*\}', respuesta, re.DOTALL)
        if m:
            respuesta = m.group(0)
        data = json.loads(respuesta)
        return {
            "tipo": data.get("tipo", "tema"),
            "articulos": [str(a) for a in data.get("articulos", [])],
            "temas": data.get("temas", []),
            "documento_filtro": data.get("documento_filtro", "").lower().strip(),
            "reformulada": data.get("reformulada", pregunta),
        }
    except Exception:
        return {
            "tipo": "tema", "articulos": [], "temas": [pregunta],
            "documento_filtro": "", "reformulada": pregunta
        }


def _norm_doc(s):
    return re.sub(r'[\s\-_]+', ' ', s.lower()).strip()


def _doc_coincide(filtro, fuente):
    if not filtro:
        return False
    f, src = _norm_doc(filtro), _norm_doc(fuente)
    if f in src:
        return True
    if "beca" in f:
        return "beca" in src
    if "general" in f or "estudiante" in f:
        return ("general" in src or "estudiante" in src) and "beca" not in src
    return False


# ─────────────────────────────────────────
# RECUPERACIÓN HÍBRIDA
# ─────────────────────────────────────────
def recuperar_contexto(pregunta: str, vectorstore, chunks_datos: list, bm25, llm,
                       historial: list = None) -> tuple[str, dict]:
    pregunta_ctx = contextualizar_pregunta(pregunta, historial or [], llm)
    documentos_disponibles = sorted(set(c[1] for c in chunks_datos))

    intencion = clasificar_intencion(pregunta_ctx, llm, documentos_disponibles)
    intencion["pregunta_original"] = pregunta
    intencion["pregunta_contextualizada"] = pregunta_ctx

    if intencion["tipo"] != "comparativa":
        if not any(_doc_coincide(intencion["documento_filtro"], d) for d in documentos_disponibles):
            intencion["documento_filtro"] = detectar_doc_filtro(pregunta_ctx, documentos_disponibles)
    else:
        intencion["documento_filtro"] = ""   # forzar búsqueda en todo el corpus

    if intencion["tipo"] == "saludo":
        return "", intencion

    if intencion["documento_filtro"]:
        filtro = intencion["documento_filtro"]
        chunks_filtrados = [c for c in chunks_datos if _doc_coincide(filtro, c[1])]
        chunks_busqueda = chunks_filtrados or chunks_datos
    else:
        chunks_busqueda = chunks_datos

    candidatos = []
    vistos = set()

    def agregar(texto, fuente, art, titulo, score):
        clave = (fuente, art, texto[:80])
        if clave not in vistos:
            vistos.add(clave)
            candidatos.append((score, texto, fuente, art, titulo))

    # 5. Artículos específicos: prioridad MÁXIMA
    for num in intencion["articulos"]:
        for texto, fuente, art, titulo in chunks_busqueda:
            if str(art) == str(num):
                agregar(texto, fuente, art, titulo, score=10000.0)

    # 6. BM25
    query_lexica = " ".join(intencion["temas"]) + " " + intencion["reformulada"]
    tokens = re.findall(r'\b\w+\b', query_lexica.lower())
    if tokens and bm25 is not None:
        try:
            scores = bm25.get_scores(tokens)
        except Exception:
            scores = []

        if len(scores) == len(chunks_datos):
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:12]
            for i in top_idx:
                if i >= len(chunks_datos):
                    continue
                if scores[i] <= 0:
                    continue
                texto, fuente, art, titulo = chunks_datos[i]
                if intencion["documento_filtro"] and not _doc_coincide(intencion["documento_filtro"], fuente):
                    continue
                agregar(texto, fuente, art, titulo, score=float(scores[i]))
        else:
            print(f"⚠️ BM25 desincronizado: {len(scores)} scores vs {len(chunks_datos)} chunks. Saltando BM25 esta vez.")

    # 7. Vector search
    try:
        filter_meta = None
        if intencion["documento_filtro"]:
            for f in documentos_disponibles:
                if _doc_coincide(intencion["documento_filtro"], f):
                    filter_meta = {"source": f}
                    break

        if filter_meta:
            docs_vec = vectorstore.similarity_search_with_score(
                intencion["reformulada"], k=12, filter=filter_meta
            )
        else:
            docs_vec = vectorstore.similarity_search_with_score(intencion["reformulada"], k=12)

        for d, dist in docs_vec:
            agregar(
                d.page_content,
                d.metadata.get("source", "Reglamento"),
                d.metadata.get("articulo", "?"),
                d.metadata.get("titulo", ""),
                score=1.0 / (1.0 + dist)
            )
    except Exception:
        pass

    candidatos.sort(key=lambda x: -x[0])
    finales = candidatos[:18]

    partes = []
    for _, texto, fuente, art, titulo in finales:
        titulo_str = f" - {titulo}" if titulo else ""
        header = f"[Documento: {fuente} | Artículo {art}{titulo_str}]"
        partes.append(f"{header}\n{texto}")

    return "\n\n---\n\n".join(partes), intencion


# ─────────────────────────────────────────
# PROCESAMIENTO DE PDFs
# ─────────────────────────────────────────
def procesar_pdfs(archivos):
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    st.markdown("### Procesando documentos...")
    progress = st.progress(0, text="Iniciando...")
    log = st.empty()

    docs_langchain = []
    total = len(archivos)
    total_ocr_pages = 0

    for i, archivo in enumerate(archivos):
        progress.progress(
            max(1, int(i / (total + 2) * 100)),
            text=f"Procesando {i+1}/{total}..."
        )
        log.markdown(
            f'<div class="step-box step-active">📄 Extrayendo: <b>{archivo.name}</b> ({i+1}/{total})</div>',
            unsafe_allow_html=True
        )

        doc_pdf = fitz.open(stream=archivo.read(), filetype="pdf")
        nombre_doc = os.path.splitext(archivo.name)[0]

        def log_ocr(msg, _name=archivo.name):
            log.markdown(
                f'<div class="step-box step-ocr">{msg} (en {_name})</div>',
                unsafe_allow_html=True
            )

        texto_completo, num_ocr = extraer_texto_completo_pdf(doc_pdf, log_callback=log_ocr)
        total_ocr_pages += num_ocr

        if num_ocr > 0:
            log.markdown(
                f'<div class="step-box step-ocr">🔍 {num_ocr} página(s) procesadas con OCR en {archivo.name}</div>',
                unsafe_allow_html=True
            )

        articulos = dividir_en_articulos(texto_completo, nombre_doc)
        docs_langchain.extend(articulos)

        arts_unicos = sorted(
            {d.metadata.get("articulo", "?") for d in articulos
             if d.metadata.get("articulo") != "?"},
            key=lambda x: int(x) if str(x).isdigit() else 9999
        )
        rango_str = f"{arts_unicos[0]} → {arts_unicos[-1]}" if arts_unicos else "—"
        log.markdown(
            f'<div class="step-box step-done">✅ {archivo.name}: '
            f'{len(arts_unicos)} artículos únicos detectados (rango: {rango_str})</div>',
            unsafe_allow_html=True
        )

        progress.progress(int((i + 1) / (total + 2) * 100), text=f"Extraído {i+1}/{total}...")

    if not docs_langchain:
        st.error("❌ No se pudo extraer ningún artículo. Verifica el formato de los PDFs.")
        return None

    st.session_state.chunks_texto   = [d.page_content                  for d in docs_langchain]
    st.session_state.chunks_fuentes = [d.metadata["source"]            for d in docs_langchain]
    st.session_state.chunks_arts    = [d.metadata.get("articulo","?")  for d in docs_langchain]
    st.session_state.chunks_titulos = [d.metadata.get("titulo","")     for d in docs_langchain]

    log.markdown(
        f'<div class="step-box step-done">✅ {len(docs_langchain)} fragmentos generados '
        f'({total_ocr_pages} páginas con OCR)</div>',
        unsafe_allow_html=True
    )
    progress.progress(80, text="Indexando en Chroma...")

    vectorstore = Chroma.from_documents(
        documents=docs_langchain,
        embedding=get_embeddings(),
        client=client,
        collection_name=COLLECTION_NAME
    )

    st.session_state.bm25 = construir_bm25(st.session_state.chunks_texto)

    progress.progress(100, text="¡Completado!")

    por_doc_summary = _agrupar_arts_por_doc(
        st.session_state.chunks_fuentes,
        st.session_state.chunks_arts,
        st.session_state.chunks_titulos,
    )
    resumen_lineas = []
    for doc, arts in sorted(por_doc_summary.items()):
        resumen_lineas.append(f"• **{doc}** → {len(arts)} artículos únicos")
    resumen_md = "\n".join(resumen_lineas)
    msg_extra = f" · {total_ocr_pages} pág. con OCR" if total_ocr_pages > 0 else ""
    st.success(f"✅ ¡Listo! {len(docs_langchain)} fragmentos de {total} archivo(s){msg_extra}.\n\n{resumen_md}")
    return vectorstore


# ─────────────────────────────────────────
# WORKER THREADED
# ─────────────────────────────────────────
def procesar_pdfs_worker(archivos_data: list, state: dict):
    try:
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        docs_langchain = []
        total = len(archivos_data)
        total_ocr_pages = 0

        for i, (nombre, file_bytes) in enumerate(archivos_data):
            if state.get('cancelar'):
                state['cancelado'] = True
                return

            state.update({
                'pct':   max(1, int(i / (total + 2) * 100)),
                'texto': f"Procesando {i+1}/{total}...",
                'log':   f"📄 Extrayendo: <b>{nombre}</b> ({i+1}/{total})",
            })

            doc_pdf = fitz.open(stream=file_bytes, filetype="pdf")
            nombre_doc = os.path.splitext(nombre)[0]

            def log_ocr(msg, _n=nombre):
                state['log'] = f"🔍 {msg} (en {_n})"

            try:
                texto_completo, num_ocr = extraer_texto_completo_pdf(
                    doc_pdf,
                    log_callback=log_ocr,
                    cancel_check=lambda: state.get('cancelar', False),
                )
            except ProcesamientoCancelado:
                state['cancelado'] = True
                return

            total_ocr_pages += num_ocr
            articulos = dividir_en_articulos(texto_completo, nombre_doc)
            docs_langchain.extend(articulos)

            arts_unicos = sorted(
                {d.metadata.get("articulo", "?") for d in articulos
                 if d.metadata.get("articulo") != "?"},
                key=lambda x: int(x) if str(x).isdigit() else 9999,
            )
            rango_str = f"{arts_unicos[0]} → {arts_unicos[-1]}" if arts_unicos else "—"
            state.update({
                'pct':   int((i + 1) / (total + 2) * 100),
                'texto': f"Extraído {i+1}/{total}...",
                'log':   f"✅ {nombre}: {len(arts_unicos)} artículos (rango: {rango_str})",
            })

        if not docs_langchain:
            state.update({'error': "No se pudo extraer ningún artículo.", 'done': True})
            return

        state.update({'pct': 80, 'texto': "Indexando en Chroma...",
                      'log': "🔄 Generando embeddings y vectorizando..."})

        vectorstore = Chroma.from_documents(
            documents=docs_langchain,
            embedding=get_embeddings(),
            client=client,
            collection_name=COLLECTION_NAME,
        )

        state.update({
            'pct':             100,
            'texto':           "¡Completado!",
            'log':             f"✅ {len(docs_langchain)} fragmentos ({total_ocr_pages} pág. con OCR)",
            'chunks_texto':    [d.page_content                 for d in docs_langchain],
            'chunks_fuentes':  [d.metadata["source"]           for d in docs_langchain],
            'chunks_arts':     [d.metadata.get("articulo","?") for d in docs_langchain],
            'chunks_titulos':  [d.metadata.get("titulo","")    for d in docs_langchain],
            'total_ocr_pages': total_ocr_pages,
            'vectorstore':     vectorstore,
            'done':            True,
        })

    except Exception as e:
        state.update({'error': str(e), 'done': True})


# ─────────────────────────────────────────
# INICIALIZACIÓN
# ─────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = cargar_vectorstore() if db_tiene_datos() else None
if "historial" not in st.session_state:
    st.session_state.historial = []
if "es_admin" not in st.session_state:
    st.session_state.es_admin = False
if "procesando" not in st.session_state:
    st.session_state.procesando = False
if "progress_state" not in st.session_state:
    st.session_state.progress_state = {}
if "prompt_inyectado" not in st.session_state:
    st.session_state.prompt_inyectado = None
if "ultima_pregunta_ts" not in st.session_state:
    st.session_state.ultima_pregunta_ts = {}

if "chunks_texto" not in st.session_state or not st.session_state.get("chunks_texto"):
    if st.session_state.get("vectorstore") and db_tiene_datos():
        try:
            col = get_chroma_client().get_collection(COLLECTION_NAME)
            datos = col.get(include=["documents", "metadatas"])
            st.session_state.chunks_texto   = datos["documents"] or []
            st.session_state.chunks_fuentes = [m.get("source",   "Reglamento") for m in (datos["metadatas"] or [])]
            st.session_state.chunks_arts    = [m.get("articulo", "?")          for m in (datos["metadatas"] or [])]
            st.session_state.chunks_titulos = [m.get("titulo",   "")           for m in (datos["metadatas"] or [])]
        except Exception:
            st.session_state.chunks_texto   = []
            st.session_state.chunks_fuentes = []
            st.session_state.chunks_arts    = []
            st.session_state.chunks_titulos = []
    else:
        st.session_state.chunks_texto   = []
        st.session_state.chunks_fuentes = []
        st.session_state.chunks_arts    = []
        st.session_state.chunks_titulos = []

# BM25: siempre se reconstruye al iniciar la sesión si hay chunks
if "bm25" not in st.session_state or st.session_state.get("bm25") is None:
    if st.session_state.get("chunks_texto"):
        st.session_state.bm25 = construir_bm25(st.session_state.chunks_texto)
    else:
        st.session_state.bm25 = None

# Defensa adicional: si BM25 y chunks no coinciden → reconstruir
_n_chunks = len(st.session_state.get("chunks_texto", []))
_n_bm25 = 0
if st.session_state.get("bm25") is not None:
    try:
        _n_bm25 = len(st.session_state.bm25.doc_freqs)
    except Exception:
        _n_bm25 = -1
if _n_chunks > 0 and _n_bm25 != _n_chunks:
    st.session_state.bm25 = construir_bm25(st.session_state.chunks_texto)
    try:
        _n_bm25 = len(st.session_state.bm25.doc_freqs)
    except Exception:
        _n_bm25 = _n_chunks


# ─────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────
st.markdown('<p class="titulo">🎓 Reglamentos PUCE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Consulta reglamentos universitarios con IA · Powered by Groq · OCR automático</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Reglamentos PUCE")

    if st.session_state.vectorstore:
        n_chunks = len(st.session_state.chunks_texto)
        docs_unicos = sorted(set(st.session_state.chunks_fuentes))

        por_doc_side = _agrupar_arts_por_doc(
            st.session_state.chunks_fuentes,
            st.session_state.chunks_arts,
            st.session_state.chunks_titulos,
        )
        total_arts = sum(len(v) for v in por_doc_side.values())

        # ── Estadísticas visibles para todos ──
        st.success("Base de datos activa ✅")
        st.caption(f"📚 {len(docs_unicos)} reglamento(s) disponibles")
        st.caption(f"📄 {total_arts} artículos indexados")

        with st.expander("📚 Documentos cargados"):
            for d in docs_unicos:
                arts = por_doc_side.get(d, {})
                st.caption(f"• **{d}** — {len(arts)} artículos")

        st.divider()

        # ── 🔍 BUSCADOR RÁPIDO DE ARTÍCULOS ──
        with st.expander("🔍 Buscar artículo directo"):
            pares = []
            vistos_arts = set()
            for f, a, t in zip(
                st.session_state.chunks_fuentes,
                st.session_state.chunks_arts,
                st.session_state.chunks_titulos,
            ):
                clave = (f, a)
                if a == "?" or clave in vistos_arts:
                    continue
                vistos_arts.add(clave)
                pares.append((f, a, t))
            try:
                pares.sort(key=lambda x: (x[0], int(x[1])))
            except (ValueError, TypeError):
                pares.sort(key=lambda x: (x[0], str(x[1])))

            opciones = ["— elige —"] + [
                f"📚 {f[:30]}… · Art. {a}" + (f" — {t[:30]}" if t else "")
                for f, a, t in pares
            ]
            sel = st.selectbox("Salta a un artículo:", opciones, key="quick_art_select")
            if sel != "— elige —":
                idx_sel = opciones.index(sel) - 1
                doc_sel, art_sel, _ = pares[idx_sel]
                if st.button("📖 Ver contenido", use_container_width=True, key="btn_ver_art"):
                    st.session_state.prompt_inyectado = f"Muéstrame el artículo {art_sel} del reglamento {doc_sel}"
                    st.rerun()

        # ── 💾 EXPORTAR CONVERSACIÓN ──
        if st.session_state.historial:
            st.download_button(
                "💾 Exportar conversación",
                data=exportar_conversacion_md(st.session_state.historial),
                file_name=f"conversacion_puce_{time.strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        if st.session_state.es_admin:
            # ── Panel de administración ──
            st.caption("🔧 **Panel de administración**")

            if st.button("🔄 Cargar nuevos PDFs", use_container_width=True):
                st.session_state.show_uploader = True
                st.rerun()

            if st.session_state.get("show_uploader"):
                if st.session_state.procesando:
                    _s = st.session_state.progress_state
                    if _s.get('done') or _s.get('cancelado'):
                        if not _s.get('cancelado') and not _s.get('error') and _s.get('vectorstore'):
                            st.session_state.chunks_texto   = _s['chunks_texto']
                            st.session_state.chunks_fuentes = _s['chunks_fuentes']
                            st.session_state.chunks_arts    = _s['chunks_arts']
                            st.session_state.chunks_titulos = _s['chunks_titulos']
                            st.session_state.vectorstore    = _s['vectorstore']
                            st.session_state.bm25 = construir_bm25(_s['chunks_texto'])
                            st.session_state.show_uploader  = False
                        st.session_state.procesando     = False
                        st.session_state.progress_state = {}
                        st.rerun()
                    else:
                        st.markdown("**Procesando documentos...**")
                        st.progress(_s.get('pct', 0), text=_s.get('texto', 'Iniciando...'))
                        if _s.get('log'):
                            st.markdown(f'<div class="step-box step-active">{_s["log"]}</div>', unsafe_allow_html=True)
                        if st.button("⛔ Detener", use_container_width=True, type="secondary", key="stop_replace"):
                            _s['cancelar'] = True
                        time.sleep(0.4)
                        st.rerun()
                else:
                    archivos = st.file_uploader("Sube nuevos PDFs", type="pdf", accept_multiple_files=True)
                    if archivos and st.button("⚡ Procesar y reemplazar", use_container_width=True, type="primary"):
                        archivos_data = [(a.name, a.read()) for a in archivos]
                        st.session_state.vectorstore    = None
                        st.session_state.historial      = []
                        st.session_state.bm25           = None
                        st.session_state.chunks_texto   = []
                        st.session_state.chunks_fuentes = []
                        st.session_state.chunks_arts    = []
                        st.session_state.chunks_titulos = []
                        _s = {'pct': 0, 'texto': 'Iniciando...', 'log': '', 'done': False, 'cancelar': False, 'cancelado': False}
                        st.session_state.progress_state = _s
                        threading.Thread(target=procesar_pdfs_worker, args=(archivos_data, _s), daemon=True).start()
                        st.session_state.procesando = True
                        st.rerun()

            if st.button("🗑️ Limpiar conversación", use_container_width=True):
                st.session_state.historial = []
                st.rerun()

            with st.expander("🔍 Diagnóstico"):
                st.caption("**Artículos detectados por documento:**")
                for d in docs_unicos:
                    arts = por_doc_side.get(d, {})
                    try:
                        nums = sorted(int(a) for a in arts.keys())
                        if nums:
                            rango = f"Art. {nums[0]} → Art. {nums[-1]}"
                            faltan = [n for n in range(nums[0], nums[-1] + 1) if n not in nums]
                            if faltan:
                                st.caption(f"⚠️ **{d}** — {len(arts)} arts. ({rango}) · Faltan: {faltan}")
                            else:
                                st.caption(f"✅ **{d}** — {len(arts)} arts. ({rango})")
                        else:
                            st.caption(f"⚠️ **{d}** — sin artículos numerados")
                    except ValueError:
                        st.caption(f"✅ **{d}** — {len(arts)} arts.")

                st.divider()
                estado_bm25 = "✅ OK" if _n_bm25 == _n_chunks else "⚠️ desincronizado"
                st.caption(f"**Fragmentos indexados:** {_n_chunks}")
                st.caption(f"**BM25:** {_n_bm25} entradas — {estado_bm25}")
        else:
            # ── Usuario normal ──
            if st.button("🗑️ Limpiar conversación", use_container_width=True):
                st.session_state.historial = []
                st.rerun()

    else:
        if st.session_state.es_admin:
            if st.session_state.procesando:
                _s = st.session_state.progress_state
                if _s.get('done') or _s.get('cancelado'):
                    if not _s.get('cancelado') and not _s.get('error') and _s.get('vectorstore'):
                        st.session_state.chunks_texto   = _s['chunks_texto']
                        st.session_state.chunks_fuentes = _s['chunks_fuentes']
                        st.session_state.chunks_arts    = _s['chunks_arts']
                        st.session_state.chunks_titulos = _s['chunks_titulos']
                        st.session_state.vectorstore    = _s['vectorstore']
                        st.session_state.bm25 = construir_bm25(_s['chunks_texto'])
                    st.session_state.procesando     = False
                    st.session_state.progress_state = {}
                    st.rerun()
                else:
                    st.markdown("**Procesando documentos...**")
                    st.progress(_s.get('pct', 0), text=_s.get('texto', 'Iniciando...'))
                    if _s.get('log'):
                        st.markdown(f'<div class="step-box step-active">{_s["log"]}</div>', unsafe_allow_html=True)
                    if st.button("⛔ Detener", use_container_width=True, type="secondary", key="stop_inicial"):
                        _s['cancelar'] = True
                    time.sleep(0.4)
                    st.rerun()
            else:
                archivos = st.file_uploader("Sube uno o más PDFs", type="pdf", accept_multiple_files=True)
                if archivos:
                    st.caption(f"{len(archivos)} archivo(s) seleccionado(s)")
                    st.info("ℹ️ Si algún PDF está escaneado, se aplicará OCR automáticamente.")
                if st.button("⚡ Procesar PDFs", disabled=not archivos, use_container_width=True, type="primary"):
                    archivos_data = [(a.name, a.read()) for a in archivos]
                    _s = {'pct': 0, 'texto': 'Iniciando...', 'log': '', 'done': False, 'cancelar': False, 'cancelado': False}
                    st.session_state.progress_state = _s
                    threading.Thread(target=procesar_pdfs_worker, args=(archivos_data, _s), daemon=True).start()
                    st.session_state.procesando = True
                    st.rerun()
        else:
            st.info("📚 Aún no hay reglamentos cargados. Contacta a un administrador.", icon="🎓")

    st.divider()
    st.markdown("**LLM:** llama-3.3-70b via Groq")
    st.markdown("**Embeddings:** multilingual-MiniLM")
    st.markdown("**Búsqueda:** BM25 + Vector + Router LLM")
    st.markdown("**OCR:** Tesseract (auto)")

    # ── Control de rol (al final del sidebar) ──
    st.divider()
    if not st.session_state.es_admin:
        with st.expander("🔐 Acceso administrador"):
            pwd = st.text_input("Contraseña:", type="password", key="pwd_input")
            if st.button("Ingresar", use_container_width=True):
                if pwd == ADMIN_PASSWORD:
                    st.session_state.es_admin = True
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta")
    else:
        st.success("Modo administrador activo ✅")
        if st.button("Cerrar sesión admin", use_container_width=True):
            st.session_state.es_admin = False
            st.rerun()


# ─────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────
if not st.session_state.vectorstore:
    if st.session_state.es_admin:
        st.info("👈 Sube uno o más PDFs en el panel lateral y pulsa **Procesar PDFs** para comenzar.", icon="📋")
    else:
        st.info("📚 Los reglamentos aún no están disponibles. Por favor, contacta a un administrador.", icon="🎓")
else:
    # ── HERO + QUICK PROMPTS si no hay historial ──
    if not st.session_state.historial and not st.session_state.prompt_inyectado:
        st.markdown(
            '<div class="hero-welcome">'
            '<h3 style="margin:0">👋 Hola, ¿qué quieres consultar hoy?</h3>'
            '<p style="color:#888;margin:6px 0 0">Pregunta libremente o elige una sugerencia para empezar.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("##### ✨ Sugerencias rápidas")
        cols = st.columns(3)
        for i, qp in enumerate(QUICK_PROMPTS):
            with cols[i % 3]:
                if st.button(
                    f"{qp['icon']}  **{qp['title']}**\n\n{qp['sub']}",
                    use_container_width=True,
                    key=f"qp_{i}",
                ):
                    st.session_state.prompt_inyectado = qp["q"]
                    st.rerun()

    # ── HISTORIAL ──
    for idx, msg in enumerate(st.session_state.historial):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and idx > 0:
                tiempo = st.session_state.ultima_pregunta_ts.get(idx)
                if tiempo:
                    st.caption(f"⏱️ {tiempo:.1f}s")

    # ── INPUT: chat normal O prompt inyectado por quick-prompt ──
    pregunta_chat = st.chat_input("¿Qué quieres saber del reglamento?")
    pregunta = st.session_state.prompt_inyectado or pregunta_chat
    if st.session_state.prompt_inyectado:
        st.session_state.prompt_inyectado = None

    if pregunta:
        st.session_state.historial.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            # ── 1) SALUDOS ───────────────────────
            if es_charla_no_consulta(pregunta):
                respuesta = (
                    "¡Hola! Soy tu asistente para los reglamentos de la PUCE. "
                    "Puedes preguntarme:\n\n"
                    "- 📄 Por **artículos específicos** (ej: «art 29», «art. 7 y 11»)\n"
                    "- 🎯 Por **temas** (ej: «becas», «titulación», «matrícula»)\n"
                    "- 📊 Por **estructura** (ej: «cuántos artículos tiene el reglamento de becas», "
                    "«lista los artículos del reglamento general»)\n"
                    "- ⚖️ Para **comparar** entre reglamentos\n\n"
                    "¿En qué te ayudo?"
                )
                st.markdown(respuesta)
                st.session_state.historial.append({"role": "assistant", "content": respuesta})
                st.stop()

            # ── 2) META-PREGUNTAS (estructura del corpus) ───
            meta_tipo = detectar_meta_pregunta(pregunta)
            if meta_tipo:
                docs_disp = sorted(set(st.session_state.chunks_fuentes))
                doc_filtro = detectar_doc_filtro(pregunta, docs_disp)

                with st.spinner("Analizando estructura del corpus..."):
                    respuesta = responder_meta_pregunta(
                        meta_tipo,
                        st.session_state.chunks_fuentes,
                        st.session_state.chunks_arts,
                        st.session_state.chunks_titulos,
                        doc_filtro=doc_filtro,
                    )

                if st.session_state.es_admin:
                    with st.expander("🔬 Cómo entendí tu pregunta", expanded=False):
                        st.json({
                            "pregunta_original": pregunta,
                            "tipo_meta": meta_tipo,
                            "documento_filtro": doc_filtro or "(todos)",
                            "fuente_respuesta": "metadata indexada (no RAG)",
                        })
                st.markdown(respuesta)
                st.session_state.historial.append({"role": "assistant", "content": respuesta})
                st.stop()

            # ── 3) CONSULTAS DE CONTENIDO (RAG normal) ───
            t0 = time.time()
            with st.spinner("Analizando los reglamentos..."):
                chunks_datos = list(zip(
                    st.session_state.chunks_texto,
                    st.session_state.chunks_fuentes,
                    st.session_state.chunks_arts,
                    st.session_state.chunks_titulos,
                ))

                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0,
                )

                historial_previo = st.session_state.historial[:-1]

                contexto, intencion = recuperar_contexto(
                    pregunta,
                    st.session_state.vectorstore,
                    chunks_datos,
                    st.session_state.bm25,
                    llm,
                    historial=historial_previo
                )

                if st.session_state.es_admin:
                    with st.expander("🔬 Cómo entendí tu pregunta", expanded=False):
                        st.json({
                            "pregunta_original": intencion.get("pregunta_original"),
                            "pregunta_reformulada": intencion.get("pregunta_contextualizada"),
                            "tipo": intencion.get("tipo"),
                            "articulos": intencion.get("articulos"),
                            "temas": intencion.get("temas"),
                            "documento_filtro": intencion.get("documento_filtro") or "(todos)",
                            "fragmentos_recuperados": contexto.count("---") + 1 if contexto else 0,
                        })

                if intencion["tipo"] == "saludo" or not contexto:
                    respuesta = (
                        "No encontré información sobre eso en los reglamentos cargados. "
                        "¿Puedes reformular tu pregunta o indicar qué tema buscas?"
                    )
                    st.markdown(respuesta)
                    st.session_state.historial.append({"role": "assistant", "content": respuesta})
                    st.stop()

                historial_texto = ""
                if historial_previo:
                    ultimas = historial_previo[-4:]
                    historial_texto = "\n\n".join([
                        f"{'Usuario' if m['role']=='user' else 'Asistente'}: {m['content']}"
                        for m in ultimas
                    ])

                template = DEFAULT_SYSTEM_PROMPT

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["historial", "context", "question"]
                )
                chain = prompt | llm | StrOutputParser()

                payload = {
                    "historial": historial_texto if historial_texto else "(Primera pregunta)",
                    "context": contexto,
                    "question": pregunta,
                }

                # ── STREAMING: respuesta palabra-por-palabra ──
                try:
                    respuesta = st.write_stream(chain.stream(payload))
                except Exception:
                    respuesta = chain.invoke(payload)
                    st.markdown(respuesta)

                tiempo_s = time.time() - t0
                st.session_state.historial.append({"role": "assistant", "content": respuesta})
                st.session_state.ultima_pregunta_ts[len(st.session_state.historial) - 1] = tiempo_s
                st.rerun()
