"""Configuração comum dos testes. Execute a partir da raiz do projeto:  python -m pytest -v"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import config  # noqa: E402
from inferencia import Analisador  # noqa: E402

TESTE = config.DIR_DADOS / "test"


@pytest.fixture(scope="session")
def analisador() -> Analisador:
    return Analisador()


@pytest.fixture()
def app_teste(tmp_path, analisador):
    """App com banco e uploads em pasta temporária (não mexe no histórico real)."""
    from api import criar_app
    return criar_app(caminho_banco=tmp_path / "historico.db", dir_uploads=tmp_path / "uploads",
                     dir_frontend=None, analisador=analisador)


@pytest.fixture()
def cliente(app_teste):
    from fastapi.testclient import TestClient
    with TestClient(app_teste) as c:
        yield c


def amostras(classe: str, n: int) -> list[Path]:
    """n imagens de teste espalhadas por todas as espécies (sem escolher a dedo)."""
    arquivos = sorted((TESTE / classe).glob("*.jpg"))
    passo = max(1, len(arquivos) // n)
    return arquivos[::passo][:n]


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()
