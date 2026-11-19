"""Treina o MODELO AUXILIAR que identifica a planta e a doença (33 classes).

Por que um segundo modelo?
    O modelo principal (melhor_modelo.pth) responde apenas "doente" ou "saudavel".
    O enunciado pede também, QUANDO POSSÍVEL, a planta e o nome da doença.
    Os nomes dos arquivos do dataset já trazem essa informação
    (ex.: Tomato_Late_blight_003.jpg = tomateiro com requeima), então dá para
    treinar um classificador mais detalhado com as MESMAS imagens e as MESMAS
    divisões treino/validação/teste — sem misturar conjuntos.

Como ele é treinado (transfer learning em duas etapas):
    1. Parte do modelo principal JÁ treinado (que por sua vez partiu do ImageNet).
       As camadas convolucionais já sabem reconhecer texturas e manchas de folhas.
    2. Troca só a última camada (2 saídas -> 33 saídas) e ajusta a rede inteira
       com taxa de aprendizado pequena.

O modelo principal NÃO é alterado: este script só lê melhor_modelo.pth.

Uso:
    python train_detalhado.py               (15 épocas, padrão)
    python train_detalhado.py --epocas 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from dataset import transform_avaliacao, transform_treino
from modelo import CNNSimples, carregar_checkpoint
from rotulos import chave_do_arquivo, info


class DatasetDetalhado(datasets.ImageFolder):
    """Igual ao ImageFolder, mas o rótulo vem do NOME DO ARQUIVO (33 classes)
    e não da pasta (2 classes)."""

    def __init__(self, raiz: Path, transform, classes_detalhadas: list[str] | None = None) -> None:
        super().__init__(raiz, transform=transform)
        chaves = [chave_do_arquivo(caminho) for caminho, _ in self.samples]
        self.classes_detalhadas: list[str] = classes_detalhadas or sorted(set(chaves))
        idx = {c: i for i, c in enumerate(self.classes_detalhadas)}
        desconhecidas = set(chaves) - set(idx)
        if desconhecidas:
            raise ValueError(f"Classes em {raiz} que não existem no treino: {sorted(desconhecidas)}")
        self.samples = [(caminho, idx[c]) for (caminho, _), c in zip(self.samples, chaves)]
        self.targets = [t for _, t in self.samples]


def trocar_cabeca(modelo: nn.Module, num_classes: int) -> nn.Module:
    """Troca a última camada linear para ter num_classes saídas."""
    if isinstance(modelo, CNNSimples):
        modelo.classificador[2] = nn.Linear(modelo.classificador[2].in_features, num_classes)
    elif hasattr(modelo, "fc"):                       # ResNet
        modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    else:                                             # MobileNetV3
        modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, num_classes)
    return modelo


def rodar_epoca(modelo, loader, criterio, device, otimizador=None) -> tuple[float, float]:
    treinando = otimizador is not None
    modelo.train(treinando)
    perda_total, acertos, total = 0.0, 0, 0
    with torch.set_grad_enabled(treinando):
        for imgs, rotulos in loader:
            imgs, rotulos = imgs.to(device), rotulos.to(device)
            saidas = modelo(imgs)
            perda = criterio(saidas, rotulos)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * imgs.size(0)
            acertos += int((saidas.argmax(1) == rotulos).sum())
            total += imgs.size(0)
    return perda_total / total, acertos / total


@torch.no_grad()
def avaliar_teste(modelo, ds: DatasetDetalhado, device) -> dict:
    """Mede, no conjunto de TESTE: acerto da classe exata, da planta e da doença."""
    modelo.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds: list[int] = []
    for imgs, _ in loader:
        preds += modelo(imgs.to(device)).argmax(1).cpu().tolist()
    classes = ds.classes_detalhadas
    linhas = []
    acerto_classe = acerto_planta = 0
    doentes = acerto_doenca = 0
    por_planta: Counter = Counter()
    total_planta: Counter = Counter()
    for (caminho, real), prev in zip(ds.samples, preds):
        ir, ip = info(classes[real]), info(classes[prev])
        acerto_classe += real == prev
        acerto_planta += ir["planta"] == ip["planta"]
        total_planta[ir["planta"]] += 1
        por_planta[ir["planta"]] += ir["planta"] == ip["planta"]
        if not ir["saudavel"]:
            doentes += 1
            acerto_doenca += real == prev
        if real != prev:
            linhas.append(f"  {Path(caminho).name}: real={classes[real]}  previsto={classes[prev]}")
    n = len(ds)
    return {
        "imagens_teste": n,
        "acuracia_classe_exata": acerto_classe / n,
        "acuracia_planta": acerto_planta / n,
        "acuracia_doenca_entre_doentes": acerto_doenca / max(doentes, 1),
        "acuracia_planta_por_especie": {p: por_planta[p] / total_planta[p] for p in sorted(total_planta)},
        "erros": linhas,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--workers", type=int, default=0, help="processos de leitura (0 = mais compatível com Windows)")
    args = p.parse_args()

    torch.manual_seed(config.SEED)
    device = config.dispositivo()
    print(f"Dispositivo: {device}")

    ds_tr = DatasetDetalhado(config.DIR_DADOS / "train", transform_treino())
    classes = ds_tr.classes_detalhadas
    ds_va = DatasetDetalhado(config.DIR_DADOS / "val", transform_avaliacao(), classes)
    ds_te = DatasetDetalhado(config.DIR_DADOS / "test", transform_avaliacao(), classes)
    print(f"{len(classes)} classes | treino {len(ds_tr)} | validação {len(ds_va)} | teste {len(ds_te)}")

    ld_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    ld_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # 1) Parte do modelo principal já treinado (somente leitura).
    base, _ = carregar_checkpoint(config.CAMINHO_MODELO, device)
    arquitetura = torch.load(config.CAMINHO_MODELO, map_location="cpu", weights_only=True)["arquitetura"]
    modelo = trocar_cabeca(base, len(classes)).to(device)

    # 2) Balanceamento: classes "saudáveis" têm 42 fotos e cada doença 15.
    #    Pesos inversamente proporcionais à frequência evitam que o modelo
    #    favoreça as classes com mais exemplos.
    contagem = Counter(ds_tr.targets)
    pesos = torch.tensor([len(ds_tr) / (len(classes) * contagem[i]) for i in range(len(classes))],
                         dtype=torch.float32, device=device)
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epocas)

    destino = config.CAMINHO_MODELO_DETALHADO
    if destino.exists():
        config.DIR_BACKUP_MODELOS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destino, config.DIR_BACKUP_MODELOS / f"modelo_detalhado_{datetime.now():%Y%m%d_%H%M%S}.pth")

    melhor_acc, hist = -1.0, []
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        pt, at = rodar_epoca(modelo, ld_tr, criterio, device, otimizador)
        pv, av = rodar_epoca(modelo, ld_va, criterio, device)
        agendador.step()
        hist.append({"epoca": ep, "perda_treino": pt, "acc_treino": at, "perda_val": pv, "acc_val": av})
        marca = ""
        if av > melhor_acc:
            melhor_acc = av
            torch.save({"arquitetura": arquitetura, "classes": classes,
                        "state_dict": modelo.state_dict(), "origem": config.CAMINHO_MODELO.name}, destino)
            marca = "  <- melhor, salvo"
        print(f"Época {ep:2d}/{args.epocas} | treino perda {pt:.3f} acc {at:.1%} | "
              f"val perda {pv:.3f} acc {av:.1%} | {time.time() - t0:.0f}s{marca}", flush=True)

    # 3) Avaliação final no TESTE (com o melhor modelo salvo, nunca usado no treino).
    final, _ = carregar_checkpoint(destino, device)
    rel = avaliar_teste(final, ds_te, device)
    rel["melhor_acuracia_validacao"] = melhor_acc
    rel["historico"] = hist
    config.DIR_RESULTADOS.mkdir(exist_ok=True)
    (config.DIR_RESULTADOS / "relatorio_detalhado.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTESTE ({rel['imagens_teste']} imagens)")
    print(f"  Classe exata (planta + condição): {rel['acuracia_classe_exata']:.1%}")
    print(f"  Planta:                            {rel['acuracia_planta']:.1%}")
    print(f"  Doença (só folhas doentes):        {rel['acuracia_doenca_entre_doentes']:.1%}")
    print(f"Modelo salvo em {destino}")
    print("Próximo passo: python calibrar.py")


if __name__ == "__main__":
    main()
