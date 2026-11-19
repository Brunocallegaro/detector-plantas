"""API REST do Detector de Doenças em Plantas (FastAPI).

Executar:
    python api.py
Depois abra:
    http://localhost:8000        -> interface web (React, já compilada em frontend/dist)
    http://localhost:8000/docs   -> documentação interativa da API (Swagger)

Endpoints:
    GET    /api/saude                 funcionamento da API, do modelo e do banco
    GET    /api/modelo                informações do modelo (arquitetura, classes, métricas)
    POST   /api/analisar              envia uma foto (form-data, campo "imagem") e recebe o resultado
    GET    /api/historico             lista as análises anteriores (paginação e filtro)
    GET    /api/historico/{id}        detalhes de uma análise
    DELETE /api/historico/{id}        remove uma análise do histórico
    GET    /api/estatisticas          totais para o dashboard
    GET    /api/imagens/{arquivo}     imagem analisada

Compatibilidade com a versão anterior do projeto:
    GET  /status    (era o GET /)   e   POST /analisar   continuam funcionando.
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from banco import Banco
from inferencia import Analisador, ErroImagem, validar_e_abrir
from modelo import ErroModelo

VERSAO = "2.0"
log = logging.getLogger("detector.api")
EXTENSAO_POR_FORMATO = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
PADRAO_ARQUIVO_SALVO = re.compile(r"^[0-9a-f]{32}\.(jpg|png|webp|bmp)$")


def configurar_logs(dir_logs: Path) -> None:
    """Logs no terminal e em instancia/logs/api.log (até 5 arquivos de 1 MB)."""
    raiz = logging.getLogger("detector")
    if raiz.handlers:
        return
    raiz.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    raiz.addHandler(console)
    try:
        dir_logs.mkdir(parents=True, exist_ok=True)
        arquivo = RotatingFileHandler(dir_logs / "api.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        arquivo.setFormatter(fmt)
        raiz.addHandler(arquivo)
    except OSError as e:
        raiz.warning("Não foi possível gravar logs em arquivo: %s", e)


def criar_app(
    caminho_banco: Path = config.CAMINHO_BANCO,
    dir_uploads: Path = config.DIR_UPLOADS,
    dir_frontend: Path | None = config.DIR_FRONTEND,
    analisador: Analisador | None = None,
) -> FastAPI:
    """Cria a aplicação. Os parâmetros permitem que os testes usem um banco separado."""
    configurar_logs(config.DIR_LOGS)
    estado: dict[str, Any] = {"analisador": analisador, "erro_modelo": None}

    @asynccontextmanager
    async def ciclo_de_vida(_app: FastAPI):
        if estado["analisador"] is None:
            try:
                estado["analisador"] = Analisador()
            except ErroModelo as e:
                estado["erro_modelo"] = str(e)
                log.error("Modelo NÃO carregado: %s", e)
        log.info("API pronta. Interface: http://localhost:8000 | Docs: http://localhost:8000/docs")
        yield

    app = FastAPI(
        title="Detector de Doenças em Plantas",
        description="Envie a foto de uma folha e receba a classificação (saudável/doente) feita por uma rede "
                    "neural (MobileNetV3). Classificação automatizada: não substitui a avaliação de um "
                    "profissional da área agronômica.",
        version=VERSAO,
        lifespan=ciclo_de_vida,
    )
    # Libera o frontend em modo de desenvolvimento (npm run dev, porta 5173).
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    banco = Banco(caminho_banco)
    dir_uploads = Path(dir_uploads)
    dir_uploads.mkdir(parents=True, exist_ok=True)

    # ---------------- tratamento de erros: sempre JSON {"erro": "..."} ----------------

    @app.exception_handler(ErroImagem)
    async def _erro_imagem(_req: Request, e: ErroImagem):
        log.info("Imagem rejeitada: %s", e.mensagem)
        return JSONResponse({"erro": e.mensagem}, status_code=e.status_http)

    @app.exception_handler(StarletteHTTPException)
    async def _erro_http(_req: Request, e: StarletteHTTPException):
        return JSONResponse({"erro": e.detail}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _erro_validacao(_req: Request, e: RequestValidationError):
        campos = {str(err.get("loc", ["", ""])[-1]) for err in e.errors()}
        if "imagem" in campos:
            msg = "Nenhuma imagem recebida. Envie o arquivo no campo 'imagem' (multipart/form-data)."
        else:
            msg = "Parâmetros inválidos: " + "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return JSONResponse({"erro": msg}, status_code=422)

    @app.exception_handler(Exception)
    async def _erro_inesperado(_req: Request, e: Exception):
        log.exception("Erro inesperado")
        return JSONResponse({"erro": "Erro interno ao processar a solicitação. Consulte o log da API."}, status_code=500)

    def obter_analisador() -> Analisador:
        if estado["analisador"] is None:
            raise HTTPException(503, f"O modelo de IA não está disponível: {estado['erro_modelo'] or 'carregando'}")
        return estado["analisador"]

    def com_url(registro: dict[str, Any]) -> dict[str, Any]:
        registro["url_imagem"] = f"/api/imagens/{registro['arquivo_salvo']}"
        return registro

    # ---------------- endpoints ----------------

    @app.get("/api/saude", tags=["Sistema"], summary="Verifica o funcionamento da API")
    def saude() -> dict[str, Any]:
        try:
            banco.contar()
            banco_ok = True
        except Exception:                                       # noqa: BLE001
            banco_ok = False
        a = estado["analisador"]
        return {"status": "online" if a and banco_ok else "degradado", "versao": VERSAO,
                "modelo_carregado": a is not None, "erro_modelo": estado["erro_modelo"],
                "banco_ok": banco_ok, "dispositivo": str(a.device) if a else None}

    @app.get("/api/modelo", tags=["Sistema"], summary="Informações do modelo de IA")
    def modelo_info() -> dict[str, Any]:
        return obter_analisador().info_modelo()

    async def _analisar(imagem: UploadFile) -> dict[str, Any]:
        analisador = obter_analisador()
        limite = config.TAMANHO_MAX_MB * 1024 * 1024
        conteudo = await imagem.read(limite + 1)               # lê no máximo limite+1 bytes
        img = validar_e_abrir(conteudo)                          # ErroImagem -> 4xx com mensagem clara
        try:
            resultado = await run_in_threadpool(analisador.analisar, img)
        except RuntimeError as e:
            log.exception("Falha na previsão de %s", imagem.filename)
            raise HTTPException(500, str(e)) from e

        formato = Image.open(io.BytesIO(conteudo)).format       # formato real, detectado pelo conteúdo
        nome_salvo = f"{uuid.uuid4().hex}{EXTENSAO_POR_FORMATO.get(formato, '.jpg')}"
        (dir_uploads / nome_salvo).write_bytes(conteudo)
        nome_original = Path(imagem.filename or "imagem").name[:200]
        registro = banco.inserir(nome_original, nome_salvo, resultado)
        log.info("Análise #%d | %s | %s | P(doente)=%.3f | %.0f ms", registro["id"], nome_original,
                 resultado["classificacao"], resultado["probabilidades"]["doente"], resultado["tempo_ms"])
        return com_url(registro)

    @app.post("/api/analisar", tags=["Análise"], summary="Analisa a foto de uma folha")
    async def analisar(imagem: UploadFile = File(..., description="Foto da folha (JPG, PNG, WEBP ou BMP, até 10 MB)")):
        """Resposta: registro salvo no histórico, com o campo `resultado` contendo a classificação,
        probabilidades, planta/doença (quando identificadas com segurança) e a mensagem explicativa."""
        return await _analisar(imagem)

    @app.get("/api/historico", tags=["Histórico"], summary="Lista as análises anteriores")
    def historico(limite: int = Query(20, ge=1, le=200), deslocamento: int = Query(0, ge=0),
                  classificacao: Literal["saudavel", "doente", "inconclusivo"] | None = None) -> dict[str, Any]:
        return {"total": banco.contar(classificacao), "limite": limite, "deslocamento": deslocamento,
                "itens": [com_url(r) for r in banco.listar(limite, deslocamento, classificacao)]}

    @app.get("/api/historico/{id_analise}", tags=["Histórico"], summary="Detalhes de uma análise")
    def historico_item(id_analise: int) -> dict[str, Any]:
        r = banco.obter(id_analise)
        if not r:
            raise HTTPException(404, f"Análise #{id_analise} não encontrada.")
        return com_url(r)

    @app.delete("/api/historico/{id_analise}", tags=["Histórico"], summary="Remove uma análise")
    def historico_excluir(id_analise: int) -> dict[str, Any]:
        arquivo = banco.excluir(id_analise)
        if arquivo is None:
            raise HTTPException(404, f"Análise #{id_analise} não encontrada.")
        (dir_uploads / arquivo).unlink(missing_ok=True)
        log.info("Análise #%d removida do histórico", id_analise)
        return {"removido": id_analise}

    @app.get("/api/estatisticas", tags=["Histórico"], summary="Totais para o dashboard")
    def estatisticas() -> dict[str, Any]:
        return banco.estatisticas()

    @app.get("/api/imagens/{arquivo}", tags=["Histórico"], summary="Imagem analisada")
    def imagem(arquivo: str):
        if not PADRAO_ARQUIVO_SALVO.match(arquivo) or not (dir_uploads / arquivo).is_file():
            raise HTTPException(404, "Imagem não encontrada.")
        return FileResponse(dir_uploads / arquivo)

    # ---------------- compatibilidade com a versão 1.0 ----------------

    @app.get("/status", tags=["Compatibilidade"], summary="(v1) status da API")
    def status_v1() -> dict[str, Any]:
        a = obter_analisador()
        return {"status": "online", "modelo": a.caminho_modelo.name, "classes": a.classes,
                "dispositivo": str(a.device), "uso": "POST /api/analisar com form-data, campo 'imagem'"}

    @app.post("/analisar", tags=["Compatibilidade"], summary="(v1) mesmo que POST /api/analisar")
    async def analisar_v1(imagem: UploadFile = File(...)):
        return await _analisar(imagem)

    # ---------------- interface web (React compilado) ----------------

    if dir_frontend is not None and (Path(dir_frontend) / "index.html").exists():
        app.mount("/", StaticFiles(directory=dir_frontend, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def raiz() -> dict[str, str]:
            return {"status": "online", "aviso": "Interface web não compilada. Rode 'npm run build' em frontend/ "
                    "ou use http://localhost:8000/docs", "docs": "/docs"}

    return app


app = criar_app()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
