"""Modelo treinado: carregamento, compatibilidade treino x inferência e desempenho real."""
import json

import pytest
import torch
from torchvision import datasets

import config
from conftest import TESTE, amostras
from dataset import transform_avaliacao, transform_treino
from inferencia import validar_e_abrir
from modelo import ErroModelo, carregar_checkpoint, criar_modelo


def test_checkpoint_carrega_com_arquitetura_identica():
    ckpt = torch.load(config.CAMINHO_MODELO, map_location="cpu", weights_only=True)
    assert ckpt["arquitetura"] == "mobilenet"
    assert ckpt["classes"] == ["doente", "saudavel"]
    referencia = criar_modelo("mobilenet", 2, pretreinado=False).state_dict()
    assert set(ckpt["state_dict"]) == set(referencia)                       # mesmas camadas
    for nome, tensor in referencia.items():
        assert ckpt["state_dict"][nome].shape == tensor.shape, nome        # mesmas dimensões
    modelo, classes = carregar_checkpoint(config.CAMINHO_MODELO)
    assert not modelo.training                                             # modo de avaliação


def test_classes_batem_com_as_pastas_do_dataset():
    _, classes = carregar_checkpoint(config.CAMINHO_MODELO)
    for split in ("train", "val", "test"):
        assert datasets.ImageFolder(config.DIR_DADOS / split).classes == classes


def test_preprocessamento_igual_ao_treino():
    nomes = lambda tf: [type(t).__name__ for t in tf.transforms]  # noqa: E731
    assert nomes(transform_avaliacao()) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    norm_av, norm_tr = transform_avaliacao().transforms[-1], transform_treino().transforms[-1]
    assert list(norm_av.mean) == list(norm_tr.mean) == [0.485, 0.456, 0.406]
    assert list(norm_av.std) == list(norm_tr.std) == [0.229, 0.224, 0.225]
    x = transform_avaliacao()(validar_e_abrir((TESTE / "doente" / "Apple_Black_rot_018.jpg").read_bytes()))
    assert tuple(x.shape) == (3, 224, 224)


def test_modelo_inexistente_gera_erro_claro(tmp_path):
    with pytest.raises(ErroModelo, match="não encontrado"):
        carregar_checkpoint(tmp_path / "nada.pth")
    corrompido = tmp_path / "ruim.pth"
    corrompido.write_bytes(b"isto nao e um modelo")
    with pytest.raises(ErroModelo, match="corrompido"):
        carregar_checkpoint(corrompido)


def test_previsao_deterministica(analisador):
    img = validar_e_abrir((TESTE / "saudavel" / "Grape_healthy_051.jpg").read_bytes())
    a, b = analisador.analisar(img), analisador.analisar(img)
    assert a["probabilidades"] == b["probabilidades"]


@pytest.mark.parametrize("classe", ["saudavel", "doente"])
def test_classifica_folhas_do_teste(analisador, classe):
    """Amostra de 30 imagens de teste por classe, espalhadas por todas as espécies."""
    arquivos = amostras(classe, 30)
    acertos = 0
    for f in arquivos:
        r = analisador.analisar(validar_e_abrir(f.read_bytes()))
        acertos += r["classificacao_modelo"] == classe
    assert acertos / len(arquivos) >= 0.90, f"{acertos}/{len(arquivos)}"


def test_metricas_do_conjunto_de_teste_completo(analisador):
    """Acurácia no teste inteiro (177 imagens) com o limiar calibrado na validação."""
    corretas = total = 0
    for classe in ("saudavel", "doente"):
        for f in sorted((TESTE / classe).glob("*.jpg")):
            r = analisador.analisar(validar_e_abrir(f.read_bytes()))
            corretas += r["classificacao_modelo"] == classe
            total += 1
    acuracia = corretas / total
    print(f"\nAcurácia no teste: {corretas}/{total} = {acuracia:.1%}")
    assert total == 177 and acuracia >= 0.95


def test_modelo_detalhado_identifica_planta(analisador):
    assert analisador.modelo_detalhado is not None and len(analisador.classes_detalhadas) == 33
    rel = json.loads((config.DIR_RESULTADOS / "calibracao.json").read_text(encoding="utf-8"))
    assert rel["detalhado"]["planta_teste"]["precisao_quando_exibido"] >= 0.90
    r = analisador.analisar(validar_e_abrir((TESTE / "doente" / "Potato_Late_blight_020.jpg").read_bytes()))
    assert r["planta"]["nome"] == "Batata" and r["doenca"]["nome"] == "Requeima"


def test_doenca_nao_aparece_em_folha_saudavel(analisador):
    for f in amostras("saudavel", 15):
        r = analisador.analisar(validar_e_abrir(f.read_bytes()))
        if r["classificacao"] == "saudavel":
            assert r["doenca"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem GPU CUDA nesta máquina")
def test_funciona_na_gpu():
    from inferencia import Analisador
    a = Analisador(device=torch.device("cuda"))
    r = a.analisar(validar_e_abrir((TESTE / "doente" / "Apple_Black_rot_018.jpg").read_bytes()))
    assert r["classificacao_modelo"] in ("doente", "saudavel")


def test_funciona_na_cpu():
    from inferencia import Analisador
    a = Analisador(device=torch.device("cpu"))
    assert next(a.modelo.parameters()).device.type == "cpu"
    r = a.analisar(validar_e_abrir((TESTE / "doente" / "Apple_Black_rot_018.jpg").read_bytes()))
    assert r["classificacao_modelo"] == "doente"


def test_calibracao_corresponde_ao_modelo(analisador):
    assert analisador.calibracao.carregada, "rode python calibrar.py"
    assert 0 < analisador.calibracao.limiar_doente < 1
