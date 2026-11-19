"""Nomes em português das plantas e condições do dataset.

O dataset do projeto é um subconjunto do PlantVillage: o nome de cada arquivo
indica a espécie e a condição da folha, por exemplo:

    data/train/doente/Tomato_Late_blight_003.jpg  ->  chave "Tomato_Late_blight"

Este arquivo apenas TRADUZ essas chaves para exibição. Ele não faz nenhuma
classificação: quem decide é sempre o modelo treinado.
"""

from __future__ import annotations

import re
from typing import TypedDict


class InfoRotulo(TypedDict):
    planta: str            # nome popular da planta em português
    condicao: str          # "Saudável" ou o nome da doença/praga
    nome_cientifico: str   # agente causador (quando se aplica)
    saudavel: bool


ROTULOS: dict[str, InfoRotulo] = {
    # Macieira
    "Apple_Apple_scab": {"planta": "Macieira", "condicao": "Sarna-da-macieira", "nome_cientifico": "Venturia inaequalis", "saudavel": False},
    "Apple_Black_rot": {"planta": "Macieira", "condicao": "Podridão-negra", "nome_cientifico": "Botryosphaeria obtusa", "saudavel": False},
    "Apple_Cedar_apple_rust": {"planta": "Macieira", "condicao": "Ferrugem-do-cedro", "nome_cientifico": "Gymnosporangium juniperi-virginianae", "saudavel": False},
    "Apple_healthy": {"planta": "Macieira", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Cerejeira
    "Cherry_including_sour_Powdery_mildew": {"planta": "Cerejeira", "condicao": "Oídio", "nome_cientifico": "Podosphaera clandestina", "saudavel": False},
    "Cherry_including_sour_healthy": {"planta": "Cerejeira", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Milho
    "Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot": {"planta": "Milho", "condicao": "Cercosporiose (mancha-cinzenta)", "nome_cientifico": "Cercospora zeae-maydis", "saudavel": False},
    "Corn_maize_Common_rust": {"planta": "Milho", "condicao": "Ferrugem-comum", "nome_cientifico": "Puccinia sorghi", "saudavel": False},
    "Corn_maize_Northern_Leaf_Blight": {"planta": "Milho", "condicao": "Helmintosporiose (queima-das-folhas)", "nome_cientifico": "Exserohilum turcicum", "saudavel": False},
    "Corn_maize_healthy": {"planta": "Milho", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Videira
    "Grape_Black_rot": {"planta": "Videira", "condicao": "Podridão-negra", "nome_cientifico": "Guignardia bidwellii", "saudavel": False},
    "Grape_Esca_Black_Measles": {"planta": "Videira", "condicao": "Esca (sarampo-negro)", "nome_cientifico": "complexo de fungos do lenho", "saudavel": False},
    "Grape_Leaf_blight_Isariopsis_Leaf_Spot": {"planta": "Videira", "condicao": "Mancha-das-folhas (Isariopsis)", "nome_cientifico": "Pseudocercospora vitis", "saudavel": False},
    "Grape_healthy": {"planta": "Videira", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Pessegueiro
    "Peach_Bacterial_spot": {"planta": "Pessegueiro", "condicao": "Mancha-bacteriana", "nome_cientifico": "Xanthomonas arboricola pv. pruni", "saudavel": False},
    "Peach_healthy": {"planta": "Pessegueiro", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Pimentão
    "Pepper_bell_Bacterial_spot": {"planta": "Pimentão", "condicao": "Mancha-bacteriana", "nome_cientifico": "Xanthomonas spp.", "saudavel": False},
    "Pepper_bell_healthy": {"planta": "Pimentão", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Batata
    "Potato_Early_blight": {"planta": "Batata", "condicao": "Pinta-preta", "nome_cientifico": "Alternaria solani", "saudavel": False},
    "Potato_Late_blight": {"planta": "Batata", "condicao": "Requeima", "nome_cientifico": "Phytophthora infestans", "saudavel": False},
    "Potato_healthy": {"planta": "Batata", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Morangueiro
    "Strawberry_Leaf_scorch": {"planta": "Morangueiro", "condicao": "Queima-das-folhas", "nome_cientifico": "Diplocarpon earlianum", "saudavel": False},
    "Strawberry_healthy": {"planta": "Morangueiro", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
    # Tomateiro
    "Tomato_Bacterial_spot": {"planta": "Tomateiro", "condicao": "Mancha-bacteriana", "nome_cientifico": "Xanthomonas spp.", "saudavel": False},
    "Tomato_Early_blight": {"planta": "Tomateiro", "condicao": "Pinta-preta", "nome_cientifico": "Alternaria solani", "saudavel": False},
    "Tomato_Late_blight": {"planta": "Tomateiro", "condicao": "Requeima", "nome_cientifico": "Phytophthora infestans", "saudavel": False},
    "Tomato_Leaf_Mold": {"planta": "Tomateiro", "condicao": "Mofo-das-folhas", "nome_cientifico": "Passalora fulva", "saudavel": False},
    "Tomato_Septoria_leaf_spot": {"planta": "Tomateiro", "condicao": "Septoriose", "nome_cientifico": "Septoria lycopersici", "saudavel": False},
    "Tomato_Spider_mites_Two-spotted_spider_mite": {"planta": "Tomateiro", "condicao": "Ácaro-rajado (praga)", "nome_cientifico": "Tetranychus urticae", "saudavel": False},
    "Tomato_Target_Spot": {"planta": "Tomateiro", "condicao": "Mancha-alvo", "nome_cientifico": "Corynespora cassiicola", "saudavel": False},
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {"planta": "Tomateiro", "condicao": "Vírus do enrolamento amarelo (TYLCV)", "nome_cientifico": "Tomato yellow leaf curl virus", "saudavel": False},
    "Tomato_Tomato_mosaic_virus": {"planta": "Tomateiro", "condicao": "Vírus do mosaico (ToMV)", "nome_cientifico": "Tomato mosaic virus", "saudavel": False},
    "Tomato_healthy": {"planta": "Tomateiro", "condicao": "Saudável", "nome_cientifico": "", "saudavel": True},
}

NOMES_CLASSE_BINARIA: dict[str, str] = {"doente": "Doente", "saudavel": "Saudável"}

_PADRAO_ARQUIVO = re.compile(r"^(?P<chave>.+)_\d+$")


def chave_do_arquivo(nome_arquivo: str) -> str:
    """'Tomato_Late_blight_003.jpg' -> 'Tomato_Late_blight'."""
    base = nome_arquivo.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    m = _PADRAO_ARQUIVO.match(base)
    if not m:
        raise ValueError(f"Nome de arquivo fora do padrão <classe>_<numero>: {nome_arquivo}")
    return m.group("chave")


def info(chave: str) -> InfoRotulo:
    """Informações em português de uma classe; usa a própria chave se ela não estiver no dicionário."""
    return ROTULOS.get(chave, {"planta": chave.split("_")[0], "condicao": chave, "nome_cientifico": "", "saudavel": "healthy" in chave})
