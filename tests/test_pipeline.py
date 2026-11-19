"""run_pipeline.py: detecção de mudanças e preservação do modelo (backup)."""
import json

import config
import run_pipeline as rp


def test_dados_identicos_ao_treino_original():
    """O hash das fotos de treino/validação é o mesmo gravado quando o modelo foi treinado."""
    salvo = json.loads((config.DIR_MODELOS / "treino_info.json").read_text(encoding="utf-8"))
    assert rp.hash_dados() == salvo["dados"]


def test_nada_mudou_entao_nao_retreina():
    salvo = rp.ler_info_treino()
    opcoes = rp.ler_opcoes([])
    atual = rp.montar_info_atual(opcoes, salvo)
    assert rp.motivos_para_treinar(opcoes, salvo, atual) == []


def test_backup_preserva_o_modelo(tmp_path, monkeypatch):
    modelos = tmp_path / "modelos"
    modelos.mkdir()
    (modelos / "melhor_modelo.pth").write_bytes(b"pesos")
    (modelos / "treino_info.json").write_text("{}")
    (tmp_path / "resultados").mkdir()
    monkeypatch.setattr(rp, "RAIZ", tmp_path)
    monkeypatch.setattr(rp, "MODELO", modelos / "melhor_modelo.pth")
    monkeypatch.setattr(rp, "INFO_TREINO", modelos / "treino_info.json")
    monkeypatch.setattr(rp, "RESULTADOS", tmp_path / "resultados")
    rp.fazer_backup_modelo_antigo()
    copias = list((modelos / "backup").glob("*/melhor_modelo.pth"))
    assert len(copias) == 1 and copias[0].read_bytes() == b"pesos"
