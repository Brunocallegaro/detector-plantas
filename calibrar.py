"""Calibra o sistema usando SOMENTE treino e validação, e mede o resultado no teste.

O que é calculado (e salvo em modelos/calibracao.pt):

1. LIMIAR DE DECISÃO "doente"
   O modelo devolve P(doente). O padrão seria classificar como doente se P >= 0,5.
   Na avaliação original, TODOS os erros do teste foram folhas doentes classificadas
   como saudáveis (falsos negativos). Aqui o limiar é escolhido como o que maximiza
   a acurácia na VALIDAÇÃO (se houver empate, usa-se o valor central do intervalo).

2. VERIFICAÇÃO DE "IMAGEM FORA DO PADRÃO"
   Um classificador binário sempre responde "doente" ou "saudável" — até para a foto
   de um gato. Para evitar diagnósticos inventados, comparamos o vetor de
   características da imagem (saída do avgpool da MobileNet do MODELO DETALHADO,
   576 valores) com os das 738 imagens de TREINO. Se a distância (cosseno) até a
   imagem de treino mais parecida for maior que o percentil 99 das distâncias da
   VALIDAÇÃO, o resultado é "inconclusivo".
   Por que o modelo detalhado? As características do modelo binário separam bem
   "doente x saudável", mas misturam fotos que não são folhas (ex.: um gato ficou
   parecido com folhas doentes). As do modelo de 33 classes são mais ricas.
   Comparação feita durante o desenvolvimento (18 fotos que não são folhas,
   do scikit-image e do scikit-learn): modelo binário detectou 11/18,
   modelo detalhado 17/18, ambos com ~1-2% de falso alarme no teste.

3. LIMIARES DO MODELO DETALHADO (planta e doença)
   A planta/doença só aparece se a confiança passar de um limiar. O limiar é o menor
   valor que, na VALIDAÇÃO, mantém a precisão >= 95% (planta) e >= 90% (doença).

Uso:  python calibrar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from inferencia import Analisador, pontuacao_padrao, sha256_arquivo
from rotulos import chave_do_arquivo, info
from train_detalhado import DatasetDetalhado
from dataset import transform_avaliacao

K_VIZINHOS = 1
PERCENTIL_PADRAO = 99
PRECISAO_MIN_PLANTA = 0.95
PRECISAO_MIN_DOENCA = 0.90


def coletar(analisador: Analisador, split: str) -> dict:
    """Roda os modelos sobre um conjunto inteiro (sem data augmentation)."""
    classes_det = analisador.classes_detalhadas or None
    if classes_det:
        ds = DatasetDetalhado(config.DIR_DADOS / split, transform_avaliacao(), classes_det)
    else:
        from torchvision import datasets
        ds = datasets.ImageFolder(config.DIR_DADOS / split, transform=transform_avaliacao())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    fonte = "detalhado" if analisador.modelo_detalhado is not None else "principal"
    p, feats, pdet = [], [], []
    for x, _ in loader:
        s = analisador.prever_tensores(x)
        p.append(s["p_doente"]); feats.append(s[f"caracteristicas_{fonte}"].cpu())
        if "p_detalhado" in s:
            pdet.append(s["p_detalhado"])
    caminhos = [c for c, _ in ds.samples]
    return {
        "doente": np.array([Path(c).parent.name == "doente" for c in caminhos]),
        "chaves": [chave_do_arquivo(c) for c in caminhos],
        "p_doente": torch.cat(p).numpy(),
        "caracteristicas": torch.cat(feats),
        "p_detalhado": torch.cat(pdet).numpy() if pdet else None,
        "caminhos": caminhos,
        "fonte": fonte,
    }


def metricas_binarias(doente: np.ndarray, p: np.ndarray, limiar: float) -> dict:
    prev = p >= limiar
    vp = int((prev & doente).sum()); vn = int((~prev & ~doente).sum())
    fp = int((prev & ~doente).sum()); fn = int((~prev & doente).sum())
    return {"limiar": round(float(limiar), 4), "acuracia": (vp + vn) / len(p),
            "sensibilidade_doente": vp / max(vp + fn, 1), "especificidade_saudavel": vn / max(vn + fp, 1),
            "precisao_doente": vp / max(vp + fp, 1),
            "matriz_confusao": {"doente_como_doente": vp, "doente_como_saudavel": fn,
                                "saudavel_como_doente": fp, "saudavel_como_saudavel": vn}}


def escolher_limiar_doente(val: dict) -> float:
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    accs = np.array([((val["p_doente"] >= t) == val["doente"]).mean() for t in candidatos])
    melhores = candidatos[accs == accs.max()]
    return float(np.median(melhores))


def planta_e_doenca(dados: dict, classes: list[str]):
    """Para cada imagem: (planta prevista, p_planta, planta real, classe prevista, p_classe, classe real)."""
    plantas = sorted({info(c)["planta"] for c in classes})
    mapa = np.array([[info(c)["planta"] == pl for c in classes] for pl in plantas], dtype=float)  # (P, C)
    p_pl = dados["p_detalhado"] @ mapa.T
    return {
        "planta_prev": np.array(plantas)[p_pl.argmax(1)], "p_planta": p_pl.max(1),
        "planta_real": np.array([info(c)["planta"] for c in dados["chaves"]]),
        "classe_prev": np.array(classes)[dados["p_detalhado"].argmax(1)], "p_classe": dados["p_detalhado"].max(1),
        "classe_real": np.array(dados["chaves"]),
    }


def menor_limiar_com_precisao(acertos: np.ndarray, conf: np.ndarray, precisao_min: float) -> float:
    for t in np.round(np.arange(0.30, 0.991, 0.01), 2):
        sel = conf >= t
        if sel.sum() >= 5 and acertos[sel].mean() >= precisao_min:
            return float(t)
    return 0.99


def seletiva(acertos: np.ndarray, conf: np.ndarray, limiar: float) -> dict:
    sel = conf >= limiar
    return {"cobertura": float(sel.mean()), "precisao_quando_exibido": float(acertos[sel].mean()) if sel.any() else None,
            "acuracia_sem_limiar": float(acertos.mean())}


def main() -> None:
    torch.manual_seed(config.SEED)
    analisador = Analisador(caminho_calibracao=Path("__sem_calibracao__"))
    print("Coletando previsões (treino, validação, teste)...")
    tr, va, te = (coletar(analisador, s) for s in ("train", "val", "test"))

    # 1) limiar de decisão
    limiar = escolher_limiar_doente(va)
    print(f"\nLimiar 'doente' escolhido na validação: {limiar:.2f}")

    # 2) referência de padrão (características do TREINO)
    ref = tr["caracteristicas"]
    s_va = pontuacao_padrao(va["caracteristicas"], ref, K_VIZINHOS).numpy()
    s_te = pontuacao_padrao(te["caracteristicas"], ref, K_VIZINHOS).numpy()
    limiar_padrao = float(np.percentile(s_va, PERCENTIL_PADRAO))
    print(f"Limiar de 'fora do padrão' (percentil {PERCENTIL_PADRAO} da validação): {limiar_padrao:.4f}")

    # 3) limiares do modelo detalhado
    lim_planta, lim_doenca = 0.80, 0.60
    met_det: dict = {}
    if va["p_detalhado"] is not None:
        cls = analisador.classes_detalhadas
        v, t = planta_e_doenca(va, cls), planta_e_doenca(te, cls)
        lim_planta = menor_limiar_com_precisao(v["planta_prev"] == v["planta_real"], v["p_planta"], PRECISAO_MIN_PLANTA)
        dv = va["doente"] & (va["p_doente"] >= limiar)       # doença só é mostrada quando o principal diz "doente"
        lim_doenca = menor_limiar_com_precisao((v["classe_prev"] == v["classe_real"])[dv], v["p_classe"][dv], PRECISAO_MIN_DOENCA)
        dt = te["doente"] & (te["p_doente"] >= limiar)
        met_det = {
            "limiar_planta": lim_planta, "limiar_doenca": lim_doenca,
            "planta_validacao": seletiva(v["planta_prev"] == v["planta_real"], v["p_planta"], lim_planta),
            "planta_teste": seletiva(t["planta_prev"] == t["planta_real"], t["p_planta"], lim_planta),
            "doenca_validacao": seletiva((v["classe_prev"] == v["classe_real"])[dv], v["p_classe"][dv], lim_doenca),
            "doenca_teste": seletiva((t["classe_prev"] == t["classe_real"])[dt], t["p_classe"][dt], lim_doenca),
        }
        print(f"Limiar planta: {lim_planta:.2f} | limiar doença: {lim_doenca:.2f}")

    metricas = {
        "binario_validacao_limiar_0_5": metricas_binarias(va["doente"], va["p_doente"], 0.5),
        "binario_validacao_calibrado": metricas_binarias(va["doente"], va["p_doente"], limiar),
        "binario_teste_limiar_0_5": metricas_binarias(te["doente"], te["p_doente"], 0.5),
        "binario_teste_calibrado": metricas_binarias(te["doente"], te["p_doente"], limiar),
        "padrao": {"fonte_caracteristicas": tr["fonte"], "k_vizinhos": K_VIZINHOS, "percentil": PERCENTIL_PADRAO, "limiar": limiar_padrao,
                   "teste_marcadas_fora_do_padrao": int((s_te > limiar_padrao).sum()),
                   "teste_total": len(s_te),
                   "pontuacao_teste_max": float(s_te.max()), "pontuacao_validacao_max": float(s_va.max())},
        "detalhado": met_det,
        "erros_teste_calibrado": [
            {"arquivo": Path(c).name, "real": "doente" if d else "saudavel", "p_doente": round(float(p), 4)}
            for c, d, p in zip(te["caminhos"], te["doente"], te["p_doente"]) if (p >= limiar) != d],
    }

    torch.save({
        "sha_modelo": sha256_arquivo(config.CAMINHO_MODELO),
        "sha_detalhado": analisador.sha_detalhado,
        "limiar_doente": limiar,
        "ood": {"referencia": ref.half(), "k": K_VIZINHOS, "limiar": limiar_padrao, "fonte": tr["fonte"]},
        "detalhado": {"limiar_planta": lim_planta, "limiar_doenca": lim_doenca},
        "metricas": json.loads(json.dumps(metricas)),
    }, config.CAMINHO_CALIBRACAO)
    config.DIR_RESULTADOS.mkdir(exist_ok=True)
    (config.DIR_RESULTADOS / "calibracao.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    a, b = metricas["binario_teste_limiar_0_5"], metricas["binario_teste_calibrado"]
    print("\n=== TESTE (177 imagens, nunca usadas na calibração) ===")
    print(f"Limiar 0,50: acurácia {a['acuracia']:.1%} | sensibilidade {a['sensibilidade_doente']:.1%} | especificidade {a['especificidade_saudavel']:.1%}")
    print(f"Limiar {limiar:.2f}: acurácia {b['acuracia']:.1%} | sensibilidade {b['sensibilidade_doente']:.1%} | especificidade {b['especificidade_saudavel']:.1%}")
    print(f"Imagens de teste marcadas como fora do padrão: {metricas['padrao']['teste_marcadas_fora_do_padrao']}/{len(s_te)}")
    if met_det:
        print(f"Planta (teste): exibida em {met_det['planta_teste']['cobertura']:.0%} das imagens, "
              f"precisão {met_det['planta_teste']['precisao_quando_exibido']:.1%}")
        d = met_det["doenca_teste"]
        print(f"Doença (teste): exibida em {d['cobertura']:.0%} das folhas doentes detectadas, "
              f"precisão {d['precisao_quando_exibido'] if d['precisao_quando_exibido'] is not None else float('nan'):.1%}")
    print(f"\nCalibração salva em {config.CAMINHO_CALIBRACAO}")


if __name__ == "__main__":
    main()
