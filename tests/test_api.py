"""API: endpoints, erros em JSON, histórico e persistência no SQLite."""
from fastapi.testclient import TestClient

from api import criar_app
from conftest import TESTE

SAUDAVEL = TESTE / "saudavel" / "Tomato_healthy_051.jpg"
DOENTE = TESTE / "doente" / "Potato_Late_blight_020.jpg"


def enviar(cliente, caminho, nome=None, rota="/api/analisar"):
    with open(caminho, "rb") as f:
        return cliente.post(rota, files={"imagem": (nome or caminho.name, f, "image/jpeg")})


def test_saude(cliente):
    r = cliente.get("/api/saude").json()
    assert r["status"] == "online" and r["modelo_carregado"] and r["banco_ok"]


def test_info_modelo(cliente):
    r = cliente.get("/api/modelo").json()
    assert r["arquitetura"] == "mobilenet" and r["classes"] == ["doente", "saudavel"]
    assert r["tamanho_entrada"] == [3, 224, 224] and r["modelo_detalhado"]["num_classes"] == 33
    assert "não substitui" in r["aviso"]


def test_analisar_folha_saudavel_e_doente(cliente):
    s = enviar(cliente, SAUDAVEL)
    d = enviar(cliente, DOENTE)
    assert s.status_code == d.status_code == 200
    s, d = s.json(), d.json()
    assert s["classificacao"] == "saudavel" and d["classificacao"] == "doente"
    assert d["resultado"]["doenca"]["nome"] == "Requeima"
    for r in (s, d):
        assert 0 <= r["confianca"] <= 1 and r["data_hora"] and r["resultado"]["mensagem"]
        assert "não substitui" in r["resultado"]["aviso"]
        assert cliente.get(r["url_imagem"]).status_code == 200


def test_erros_retornam_json_compreensivel(cliente):
    casos = [
        (cliente.post("/api/analisar", files={"imagem": ("a.txt", b"texto", "text/plain")}), 415, "não é uma imagem"),
        (cliente.post("/api/analisar", files={"imagem": ("v.jpg", b"", "image/jpeg")}), 400, "vazio"),
        (cliente.post("/api/analisar"), 422, "Nenhuma imagem"),
        (cliente.get("/api/historico/99999"), 404, "não encontrada"),
        (cliente.get("/api/historico?limite=0"), 422, "Parâmetros"),
    ]
    for resposta, codigo, trecho in casos:
        assert resposta.status_code == codigo, resposta.text
        assert trecho in resposta.json()["erro"]


def test_erro_na_previsao_vira_500_com_mensagem(cliente, analisador, monkeypatch):
    def falhar(_img):
        raise RuntimeError("Erro interno ao executar o modelo de IA.")
    monkeypatch.setattr(analisador, "analisar", falhar)
    r = enviar(cliente, SAUDAVEL)
    assert r.status_code == 500 and "modelo de IA" in r.json()["erro"]
    assert cliente.get("/api/historico").json()["total"] == 0          # nada é gravado se falhar


def test_historico_filtro_paginacao_e_estatisticas(cliente):
    for _ in range(3):
        enviar(cliente, SAUDAVEL)
    enviar(cliente, DOENTE)
    h = cliente.get("/api/historico?limite=2").json()
    assert h["total"] == 4 and len(h["itens"]) == 2
    assert h["itens"][0]["classificacao"] == "doente"                    # mais recente primeiro
    assert cliente.get("/api/historico?classificacao=doente").json()["total"] == 1
    e = cliente.get("/api/estatisticas").json()
    assert (e["total"], e["saudaveis"], e["doentes"], e["inconclusivas"]) == (4, 3, 1, 0)
    assert e["doencas_mais_frequentes"][0]["doenca"] == "Requeima"


def test_excluir_remove_registro_e_imagem(cliente, tmp_path):
    r = enviar(cliente, DOENTE).json()
    arquivo = tmp_path / "uploads" / r["arquivo_salvo"]
    assert arquivo.exists()
    assert cliente.delete(f"/api/historico/{r['id']}").status_code == 200
    assert not arquivo.exists() and cliente.get(f"/api/historico/{r['id']}").status_code == 404
    assert cliente.delete(f"/api/historico/{r['id']}").status_code == 404


def test_imagens_nao_permitem_acessar_outros_arquivos(cliente):
    for nome in ["../api.py", "..%2Fapi.py", "historico.db", "abc.jpg"]:
        assert cliente.get(f"/api/imagens/{nome}").status_code == 404


def test_historico_persiste_apos_reiniciar(tmp_path, analisador):
    banco, uploads = tmp_path / "persist.db", tmp_path / "up"
    with TestClient(criar_app(banco, uploads, None, analisador)) as c1:
        id1 = enviar(c1, DOENTE, nome="minha_folha.jpg").json()["id"]
    # "reinicia" a aplicação: nova instância, mesmo arquivo de banco
    with TestClient(criar_app(banco, uploads, None, analisador)) as c2:
        r = c2.get(f"/api/historico/{id1}").json()
        assert r["nome_arquivo"] == "minha_folha.jpg" and r["classificacao"] == "doente"
        assert c2.get(r["url_imagem"]).status_code == 200


def test_endpoints_da_versao_anterior(cliente):
    assert cliente.get("/status").json()["classes"] == ["doente", "saudavel"]
    assert enviar(cliente, DOENTE, rota="/analisar").json()["classificacao"] == "doente"


def test_raiz_serve_interface_quando_compilada(tmp_path, analisador):
    import config
    if not (config.DIR_FRONTEND / "index.html").exists():
        import pytest
        pytest.skip("frontend não compilado (npm run build)")
    with TestClient(criar_app(tmp_path / "f.db", tmp_path / "u", config.DIR_FRONTEND, analisador)) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Detector de Doenças em Plantas" in r.text
