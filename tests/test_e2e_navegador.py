"""Teste ponta a ponta: navegador real -> interface React -> API -> modelo -> SQLite.

Requer: pip install playwright  e  python -m playwright install chromium
(é ignorado automaticamente se o Playwright não estiver instalado).
"""
import os
import shutil
import socket
import threading
import time

import pytest
import uvicorn

import config
from conftest import TESTE

playwright = pytest.importorskip("playwright.sync_api")


def _porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def servidor(tmp_path, analisador):
    if not (config.DIR_FRONTEND / "index.html").exists():
        pytest.skip("frontend não compilado (npm run build)")
    from api import criar_app
    porta = _porta_livre()
    app = criar_app(tmp_path / "e2e.db", tmp_path / "up", config.DIR_FRONTEND, analisador)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=porta, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{porta}"
    srv.should_exit = True
    t.join(5)


def _navegador(p):
    chrome = os.environ.get("CHROME") or shutil.which("google-chrome") or shutil.which("chrome")
    try:
        return p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else p.chromium.launch()
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"navegador indisponível: {e}")


def test_fluxo_completo_no_navegador(servidor):
    with playwright.sync_playwright() as p:
        nav = _navegador(p)
        pg = nav.new_page()
        pg.goto(servidor + "/#/analisar")
        pg.set_input_files("input[type=file]", str(TESTE / "doente" / "Grape_Black_rot_019.jpg"))
        pg.wait_for_selector(".previa img")
        pg.click('button:has-text("Analisar folha")')
        pg.wait_for_selector(".ficha", timeout=30_000)
        assert pg.inner_text(".veredito") == "Doente"
        assert "Videira" in pg.inner_text(".campos")

        pg.click('button:has-text("Analisar outra folha")')
        pg.set_input_files("input[type=file]", files=[{"name": "x.png", "mimeType": "image/png", "buffer": b"nada"}])
        pg.click('button:has-text("Analisar folha")')
        assert "não é uma imagem" in pg.wait_for_selector(".aviso.erro").inner_text()

        pg.click("nav >> text=Histórico")
        pg.wait_for_selector(".lista-analises li")
        assert pg.locator(".lista-analises li").count() == 1

        pg.click("nav >> text=Painel")
        pg.wait_for_selector(".legenda")
        assert "Doentes\n1" in pg.inner_text(".legenda")
        nav.close()
