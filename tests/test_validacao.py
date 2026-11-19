"""Validação de entradas: formatos, tamanho, arquivos inválidos e imagens sem conteúdo."""
import io

import pytest
from PIL import Image

import config
from conftest import TESTE, png_bytes
from inferencia import ErroImagem, validar_e_abrir

JPG = (TESTE / "doente" / "Apple_Black_rot_018.jpg").read_bytes()


def erro(conteudo):
    with pytest.raises(ErroImagem) as e:
        validar_e_abrir(conteudo)
    return e.value


def test_arquivo_vazio():
    assert erro(b"").status_http == 400


def test_arquivo_texto():
    e = erro(b"apenas um texto qualquer")
    assert e.status_http == 415 and "não é uma imagem" in e.mensagem


def test_arquivo_grande_demais():
    assert erro(b"\xff" * (config.TAMANHO_MAX_MB * 1024 * 1024 + 1)).status_http == 413


def test_jpeg_truncado():
    assert erro(JPG[: len(JPG) // 3]).status_http == 422


def test_imagem_pequena_demais():
    assert "muito pequena" in erro(png_bytes(Image.new("RGB", (40, 40), "green"))).mensagem


def test_imagem_de_uma_cor_so():
    assert "única cor" in erro(png_bytes(Image.new("RGB", (300, 300), (30, 120, 40)))).mensagem


def test_formato_nao_suportado_gif():
    b = io.BytesIO()
    Image.open(io.BytesIO(JPG)).convert("P").save(b, "GIF")
    assert erro(b.getvalue()).status_http == 415


@pytest.mark.parametrize("formato", ["JPEG", "PNG", "WEBP", "BMP"])
def test_formatos_aceitos(formato):
    b = io.BytesIO()
    Image.open(io.BytesIO(JPG)).save(b, formato)
    img = validar_e_abrir(b.getvalue())
    assert img.mode == "RGB" and img.size == (256, 256)


def test_png_transparente_vira_fundo_branco():
    pixels = Image.open(io.BytesIO(JPG)).convert("RGBA")
    pixels.paste((0, 0, 0, 0), (0, 0, 60, 60))
    saida = validar_e_abrir(png_bytes(pixels))
    assert saida.getpixel((5, 5)) == (255, 255, 255)


def test_rotacao_exif_de_celular_e_aplicada():
    original = Image.open(io.BytesIO(JPG)).crop((0, 0, 256, 128))   # 256 x 128 (paisagem)
    exif = Image.Exif()
    exif[0x0112] = 6                                                 # "girar 90°" (padrão de celulares)
    b = io.BytesIO()
    original.save(b, "JPEG", exif=exif)
    assert validar_e_abrir(b.getvalue()).size == (128, 256)
