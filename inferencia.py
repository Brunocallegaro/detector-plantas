"""Serviço de inferência: transforma uma foto em um resultado de análise.

É o ÚNICO lugar do projeto que faz a previsão "de verdade" para o usuário.
A API (api.py), o predict.py e os testes usam esta mesma classe, garantindo que
todos apliquem exatamente o mesmo pré-processamento usado na avaliação do modelo
(dataset.transform_avaliacao: Resize 255 -> CenterCrop 224 -> Normalize ImageNet).

Etapas de uma análise:
    1. validar_e_abrir()  -> confere tamanho, formato real, resolução, conteúdo;
                             corrige a rotação de fotos de celular (EXIF).
    2. modelo principal   -> probabilidade de "doente" (decisão com limiar calibrado).
    3. verificação de padrão -> a imagem se parece com as do treino?
                             (distância no espaço de características; ver calibrar.py)
    4. modelo detalhado   -> planta e doença, SOMENTE se houver confiança suficiente.
    5. mensagem explicativa, sempre lembrando que não substitui um profissional.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

import config
from dataset import transform_avaliacao
from modelo import ErroModelo, camada_de_caracteristicas, carregar_checkpoint
from rotulos import NOMES_CLASSE_BINARIA, info

log = logging.getLogger("detector.inferencia")

AVISO_PROFISSIONAL = (
    "Esta é uma classificação automatizada feita por um modelo de aprendizado de máquina. "
    "O nível de confiança não é garantia de que o resultado está correto, e a análise "
    "não substitui a avaliação de um profissional da área agronômica."
)

# Impede que imagens gigantes (ex.: 30000 x 30000) esgotem a memória.
Image.MAX_IMAGE_PIXELS = 40_000_000


class ErroImagem(ValueError):
    """Imagem rejeitada na validação. status_http indica o código para a API."""

    def __init__(self, mensagem: str, status_http: int = 422) -> None:
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.status_http = status_http


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# ------------------------------------------------------------
# 1. VALIDAÇÃO DA IMAGEM
# ------------------------------------------------------------

def validar_e_abrir(conteudo: bytes) -> Image.Image:
    """Valida os bytes recebidos e devolve uma imagem RGB pronta para o modelo.

    Lança ErroImagem com uma mensagem compreensível se algo estiver errado.
    """
    if not conteudo:
        raise ErroImagem("O arquivo enviado está vazio.", 400)
    if len(conteudo) > config.TAMANHO_MAX_MB * 1024 * 1024:
        raise ErroImagem(f"A imagem tem {len(conteudo) / 1024 / 1024:.1f} MB; o limite é {config.TAMANHO_MAX_MB} MB.", 413)

    try:
        img = Image.open(io.BytesIO(conteudo))
        formato = img.format
        img.load()                                   # decodifica de fato (detecta arquivos truncados)
    except UnidentifiedImageError:
        raise ErroImagem("O arquivo enviado não é uma imagem válida. Use JPG, PNG, WEBP ou BMP.", 415) from None
    except Image.DecompressionBombError:
        raise ErroImagem("A imagem tem resolução grande demais para ser processada.", 413) from None
    except (OSError, SyntaxError, ValueError) as e:
        raise ErroImagem(f"A imagem está corrompida ou incompleta ({e}).", 422) from None

    if formato not in config.FORMATOS_ACEITOS:
        raise ErroImagem(f"Formato {formato} não suportado. Use JPG, PNG, WEBP ou BMP.", 415)

    largura, altura = img.size
    if min(largura, altura) < config.LADO_MINIMO_PX:
        raise ErroImagem(
            f"A imagem é muito pequena ({largura}x{altura} px). O mínimo é "
            f"{config.LADO_MINIMO_PX}x{config.LADO_MINIMO_PX} px.", 422)

    img = ImageOps.exif_transpose(img)               # fotos de celular: aplica a rotação gravada no EXIF

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGB", img.size, (255, 255, 255))
        fundo.paste(img, mask=img.getchannel("A"))   # transparência vira fundo branco (e não preto)
        img = fundo
    else:
        img = img.convert("RGB")

    # Imagem "sem conteúdo": praticamente uma cor só (ex.: foto toda preta ou branca).
    desvio = ImageStat.Stat(img.convert("L").resize((128, 128))).stddev[0]
    if desvio < 6.0:
        raise ErroImagem("A imagem não tem conteúdo analisável (é praticamente de uma única cor).", 422)

    return img


# ------------------------------------------------------------
# 2. CALIBRAÇÃO (gerada por calibrar.py)
# ------------------------------------------------------------

@dataclass
class Calibracao:
    limiar_doente: float = 0.5               # P(doente) >= limiar  ->  "doente"
    referencia: torch.Tensor | None = None   # características (normalizadas) das imagens de TREINO
    k_vizinhos: int = 5
    limiar_padrao: float | None = None       # acima disso a imagem é "fora do padrão"
    limiar_planta: float = 0.80              # confiança mínima para exibir a planta
    limiar_doenca: float = 0.60              # confiança mínima para exibir a doença
    fonte_padrao: str = "principal"          # de qual modelo vêm as características ("principal" ou "detalhado")
    metricas: dict[str, Any] | None = None
    carregada: bool = False


def carregar_calibracao(caminho: Path, sha_modelo: str, sha_detalhado: str | None) -> Calibracao:
    if not caminho.exists():
        log.warning("Calibração não encontrada (%s). Usando limiar 0,5 e sem verificação de padrão. "
                    "Rode: python calibrar.py", caminho.name)
        return Calibracao()
    dados = torch.load(caminho, map_location="cpu", weights_only=True)
    if dados.get("sha_modelo") != sha_modelo or dados.get("sha_detalhado") != sha_detalhado:
        log.warning("A calibração foi feita para OUTRO modelo (melhor_modelo.pth ou modelo_detalhado.pth mudou). "
                    "Ignorando-a. Rode novamente: python calibrar.py")
        return Calibracao()
    return Calibracao(
        limiar_doente=float(dados["limiar_doente"]),
        referencia=dados["ood"]["referencia"].float(),
        k_vizinhos=int(dados["ood"]["k"]),
        limiar_padrao=float(dados["ood"]["limiar"]),
        limiar_planta=float(dados["detalhado"]["limiar_planta"]),
        limiar_doenca=float(dados["detalhado"]["limiar_doenca"]),
        fonte_padrao=str(dados["ood"]["fonte"]),
        metricas=dados.get("metricas"),
        carregada=True,
    )


def pontuacao_padrao(caracteristicas: torch.Tensor, referencia: torch.Tensor, k: int) -> torch.Tensor:
    """Distância (cosseno) média até os k vizinhos mais próximos do treino.

    0 = idêntica a imagens do treino; valores altos = muito diferente de tudo
    que o modelo viu. caracteristicas: (N, D) já normalizadas.
    """
    similaridade = caracteristicas @ referencia.T          # (N, R)
    topk = similaridade.topk(k, dim=1).values               # k mais parecidas
    return 1.0 - topk.mean(dim=1)


# ------------------------------------------------------------
# 3. O ANALISADOR
# ------------------------------------------------------------

class Analisador:
    """Carrega os modelos uma única vez e analisa imagens."""

    def __init__(
        self,
        caminho_modelo: Path = config.CAMINHO_MODELO,
        caminho_detalhado: Path | None = config.CAMINHO_MODELO_DETALHADO,
        caminho_calibracao: Path = config.CAMINHO_CALIBRACAO,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or config.dispositivo()
        self.caminho_modelo = Path(caminho_modelo)
        self.modelo, self.classes = carregar_checkpoint(self.caminho_modelo, self.device)
        if sorted(self.classes) != ["doente", "saudavel"]:
            raise ErroModelo(f"O modelo principal deveria ter as classes doente/saudavel, mas tem {self.classes}.")
        self.idx_doente = self.classes.index("doente")
        self.arquitetura = torch.load(self.caminho_modelo, map_location="cpu", weights_only=True)["arquitetura"]
        self.sha_modelo = sha256_arquivo(self.caminho_modelo)
        self.transform = transform_avaliacao()
        self._trava = threading.Lock()                   # a API atende requisições em paralelo
        self._caracteristicas: dict[str, torch.Tensor] = {}
        camada_de_caracteristicas(self.modelo).register_forward_hook(self._gancho("principal"))

        self.modelo_detalhado = None
        self.classes_detalhadas: list[str] = []
        self.sha_detalhado: str | None = None
        if caminho_detalhado is not None and Path(caminho_detalhado).exists():
            try:
                self.modelo_detalhado, self.classes_detalhadas = carregar_checkpoint(caminho_detalhado, self.device)
                self.sha_detalhado = sha256_arquivo(Path(caminho_detalhado))
                camada_de_caracteristicas(self.modelo_detalhado).register_forward_hook(self._gancho("detalhado"))
            except ErroModelo as e:
                log.warning("Modelo detalhado ignorado: %s", e)
        elif caminho_detalhado is not None:
            log.warning("Modelo detalhado não encontrado; planta e doença não serão identificadas. "
                        "Rode: python train_detalhado.py")

        self.calibracao = carregar_calibracao(Path(caminho_calibracao), self.sha_modelo, self.sha_detalhado)
        if self.calibracao.referencia is not None:
            self.calibracao.referencia = self.calibracao.referencia.to(self.device)
        log.info("Modelo %s (%s) carregado em %s | limiar doente=%.3f | detalhado=%s | verificação de padrão=%s",
                 self.caminho_modelo.name, self.arquitetura, self.device, self.calibracao.limiar_doente,
                 self.modelo_detalhado is not None, self.calibracao.limiar_padrao is not None)

    def _gancho(self, nome: str):
        """Cria um "forward hook": guarda a saída da camada de características do modelo."""
        def guardar(_modulo, _entrada, saida) -> None:
            self._caracteristicas[nome] = torch.flatten(saida, 1)
        return guardar

    # -------- previsão em lote (usada também pela calibração e pelos testes) --------

    @torch.no_grad()
    def prever_tensores(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (N, 3, 224, 224) já normalizado. Devolve probabilidades e características."""
        with self._trava:
            x = x.to(self.device)
            logits = self.modelo(x)
            saida = {"p_doente": torch.softmax(logits, 1)[:, self.idx_doente].cpu(),
                     "caracteristicas_principal": F.normalize(self._caracteristicas["principal"], dim=1)}
            if self.modelo_detalhado is not None:
                saida["p_detalhado"] = torch.softmax(self.modelo_detalhado(x), 1).cpu()
                saida["caracteristicas_detalhado"] = F.normalize(self._caracteristicas["detalhado"], dim=1)
        return saida

    # -------- análise de UMA imagem, com resultado explicado --------

    def analisar(self, img: Image.Image) -> dict[str, Any]:
        t0 = time.perf_counter()
        cal = self.calibracao
        try:
            saida = self.prever_tensores(self.transform(img).unsqueeze(0))
        except Exception as e:                                   # erro inesperado do PyTorch
            log.exception("Falha na previsão")
            raise RuntimeError("Erro interno ao executar o modelo de IA.") from e

        p_doente = float(saida["p_doente"][0])
        decisao = "doente" if p_doente >= cal.limiar_doente else "saudavel"
        confianca = p_doente if decisao == "doente" else 1.0 - p_doente

        fora_padrao, pontuacao = False, None
        if cal.referencia is not None and cal.limiar_padrao is not None:
            feats = saida[f"caracteristicas_{cal.fonte_padrao}"]
            pontuacao = float(pontuacao_padrao(feats, cal.referencia, cal.k_vizinhos)[0])
            fora_padrao = pontuacao > cal.limiar_padrao

        planta = doenca = None
        top3: list[dict[str, Any]] = []
        if "p_detalhado" in saida and not fora_padrao:
            planta, doenca, top3 = self._interpretar_detalhado(saida["p_detalhado"][0], decisao)

        classificacao = "inconclusivo" if fora_padrao else decisao
        resultado = {
            "status": "inconclusiva" if fora_padrao else "concluida",
            "classificacao": classificacao,
            "classificacao_nome": "Inconclusivo" if fora_padrao else NOMES_CLASSE_BINARIA[decisao],
            "classificacao_modelo": decisao,
            "confianca": round(confianca, 4),
            "nivel_confianca": self._nivel(confianca),
            "probabilidades": {"doente": round(p_doente, 4), "saudavel": round(1 - p_doente, 4)},
            "limiar_doente": round(cal.limiar_doente, 4),
            "planta": planta,
            "doenca": doenca,
            "top3_detalhado": top3,
            "fora_do_padrao": fora_padrao,
            "pontuacao_padrao": None if pontuacao is None else round(pontuacao, 4),
            "limiar_padrao": None if cal.limiar_padrao is None else round(cal.limiar_padrao, 4),
            "aviso": AVISO_PROFISSIONAL,
            "tempo_ms": 0.0,
        }
        resultado["mensagem"] = self._mensagem(resultado)
        resultado["tempo_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return resultado

    @staticmethod
    def _nivel(confianca: float) -> str:
        if confianca >= 0.90:
            return "alta"
        if confianca >= 0.70:
            return "média"
        return "baixa"

    def _interpretar_detalhado(self, probs: torch.Tensor, decisao: str):
        """Planta e doença só são exibidas se o modelo detalhado estiver confiante
        e for coerente com a decisão do modelo principal."""
        cal = self.calibracao
        por_planta: dict[str, float] = {}
        for chave, p in zip(self.classes_detalhadas, probs.tolist()):
            nome = info(chave)["planta"]
            por_planta[nome] = por_planta.get(nome, 0.0) + p
        nome_planta, p_planta = max(por_planta.items(), key=lambda kv: kv[1])
        planta = {"nome": nome_planta, "confianca": round(p_planta, 4)} if p_planta >= cal.limiar_planta else None

        valores, indices = probs.topk(min(3, len(probs)))
        top3 = []
        for p, i in zip(valores.tolist(), indices.tolist()):
            r = info(self.classes_detalhadas[i])
            top3.append({"classe": self.classes_detalhadas[i], "planta": r["planta"], "condicao": r["condicao"],
                         "probabilidade": round(p, 4)})

        doenca = None
        melhor = info(self.classes_detalhadas[int(indices[0])])
        p_melhor = float(valores[0])
        if (decisao == "doente" and not melhor["saudavel"] and p_melhor >= cal.limiar_doenca
                and (planta is None or melhor["planta"] == planta["nome"])):
            doenca = {"nome": melhor["condicao"], "nome_cientifico": melhor["nome_cientifico"],
                      "planta": melhor["planta"], "confianca": round(p_melhor, 4)}
        return planta, doenca, top3

    @staticmethod
    def _mensagem(r: dict[str, Any]) -> str:
        def pct(v: float) -> str:                    # nunca "0%" nem "100%" (evita falsa certeza)
            if v >= 0.995:
                return "mais de 99%"
            if v < 0.005:
                return "menos de 1%"
            return f"{v * 100:.0f}%"
        if r["fora_do_padrao"]:
            return ("Resultado inconclusivo: a imagem é muito diferente das fotografias usadas no treinamento "
                    "(uma folha isolada, centralizada e sobre fundo neutro). Por isso o sistema não apresenta "
                    "um diagnóstico. Tente fotografar uma única folha, bem iluminada, ocupando a maior parte da imagem.")
        partes = []
        if r["classificacao"] == "saudavel":
            partes.append(f"O modelo classificou a folha como saudável: a probabilidade estimada de doença foi "
                          f"{pct(r['probabilidades']['doente'])}, abaixo do limiar de decisão de {pct(r['limiar_doente'])}.")
        else:
            partes.append(f"O modelo identificou padrões visuais associados a doença: probabilidade estimada de "
                          f"{pct(r['probabilidades']['doente'])} (limiar de decisão: {pct(r['limiar_doente'])}).")
        if r["planta"]:
            partes.append(f"A planta parece ser {r['planta']['nome'].lower()} (confiança {pct(r['planta']['confianca'])}).")
        else:
            partes.append("A espécie da planta não pôde ser identificada com segurança.")
        if r["classificacao"] == "doente":
            if r["doenca"]:
                partes.append(f"Os sintomas são mais compatíveis com {r['doenca']['nome'].lower()} "
                              f"(confiança {pct(r['doenca']['confianca'])}).")
            else:
                partes.append("A doença específica não pôde ser identificada com segurança pelo modelo.")
            partes.append("Recomenda-se inspecionar a planta e consultar um engenheiro agrônomo.")
        if r["nivel_confianca"] == "baixa":
            partes.append("Atenção: o nível de confiança desta classificação é baixo.")
        return " ".join(partes)

    def info_modelo(self) -> dict[str, Any]:
        cal = self.calibracao
        return {
            "arquivo": self.caminho_modelo.name,
            "arquitetura": self.arquitetura,
            "classes": self.classes,
            "dispositivo": str(self.device),
            "tamanho_entrada": [3, config.TAMANHO_IMAGEM, config.TAMANHO_IMAGEM],
            "normalizacao": {"media": config.MEDIA, "desvio": config.DESVIO},
            "sha256": self.sha_modelo,
            "limiar_doente": cal.limiar_doente,
            "calibracao_carregada": cal.carregada,
            "verificacao_padrao": cal.limiar_padrao is not None,
            "verificacao_padrao_fonte": cal.fonte_padrao,
            "modelo_detalhado": {
                "disponivel": self.modelo_detalhado is not None,
                "num_classes": len(self.classes_detalhadas),
                "plantas": sorted({info(c)["planta"] for c in self.classes_detalhadas}),
                "condicoes": [{"classe": c, **info(c)} for c in self.classes_detalhadas],
                "limiar_planta": cal.limiar_planta,
                "limiar_doenca": cal.limiar_doenca,
            },
            "metricas": cal.metricas or {},
            "aviso": AVISO_PROFISSIONAL,
        }
