"""Imagens que não são folhas devem dar resultado INCONCLUSIVO (e não um diagnóstico inventado)."""
import numpy as np
import pytest
from PIL import Image

from conftest import TESTE, png_bytes
from inferencia import validar_e_abrir

skimage_data = pytest.importorskip("skimage.data")
NAO_FOLHAS = ["astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry",
              "retina", "brick", "grass", "gravel", "colorwheel", "camera", "coins", "horse", "page"]


def _imagem(nome):
    arr = np.asarray(getattr(skimage_data, nome)())
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return Image.fromarray(arr).convert("RGB")


def test_fotos_que_nao_sao_folhas_ficam_inconclusivas(analisador):
    resultados = {n: analisador.analisar(validar_e_abrir(png_bytes(_imagem(n))))["classificacao"] for n in NAO_FOLHAS}
    inconclusivas = [n for n, c in resultados.items() if c == "inconclusivo"]
    print(f"\nInconclusivas: {len(inconclusivas)}/{len(NAO_FOLHAS)} | não detectadas: "
          f"{[n for n in NAO_FOLHAS if n not in inconclusivas]}")
    assert len(inconclusivas) / len(NAO_FOLHAS) >= 0.85


def test_inconclusivo_nao_mostra_planta_nem_doenca(analisador):
    r = analisador.analisar(validar_e_abrir(png_bytes(_imagem("chelsea"))))
    assert r["classificacao"] == "inconclusivo" and r["planta"] is None and r["doenca"] is None
    assert "inconclusivo" in r["mensagem"].lower()


def test_poucas_folhas_reais_marcadas_como_fora_do_padrao(analisador):
    arquivos = sorted(TESTE.rglob("*.jpg"))
    marcadas = sum(analisador.analisar(validar_e_abrir(f.read_bytes()))["fora_do_padrao"] for f in arquivos)
    print(f"\nFolhas de teste marcadas como fora do padrão: {marcadas}/{len(arquivos)}")
    assert marcadas / len(arquivos) <= 0.05
