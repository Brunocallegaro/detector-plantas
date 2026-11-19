"""Classifica uma imagem ou uma pasta de imagens pelo terminal.

Usa o MESMO serviço de inferência da API (inferencia.Analisador), então o
resultado no terminal é idêntico ao da interface web.

Exemplos:
    python predict.py data/test/doente/Tomato_Late_blight_018.jpg
    python predict.py data/test/saudavel
    python predict.py minha_foto.jpg --json
"""

import argparse
import json
import logging
from pathlib import Path

import config
from inferencia import Analisador, ErroImagem, validar_e_abrir

EXTENSOES: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Extensões de imagem procuradas quando o caminho é uma pasta.

ICONES = {"saudavel": "🌿", "doente": "🍂", "inconclusivo": "❔"}


def main() -> None:
    p = argparse.ArgumentParser(description="Classifica folhas como saudáveis ou doentes.")
    p.add_argument("caminho", help="imagem ou pasta de imagens")
    p.add_argument("--modelo", default=str(config.CAMINHO_MODELO), help="checkpoint do modelo principal")
    p.add_argument("--json", action="store_true", help="mostra o resultado completo em JSON")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    analisador = Analisador(caminho_modelo=Path(args.modelo))

    alvo = Path(args.caminho)
    if not alvo.exists():
        raise SystemExit(f"Caminho não encontrado: {alvo}")
    arquivos = sorted(f for f in alvo.rglob("*") if f.suffix.lower() in EXTENSOES) if alvo.is_dir() else [alvo]
    if not arquivos:
        raise SystemExit(f"Nenhuma imagem ({', '.join(sorted(EXTENSOES))}) encontrada em {alvo}")

    for f in arquivos:
        try:
            r = analisador.analisar(validar_e_abrir(f.read_bytes()))
        except ErroImagem as e:
            print(f"⚠️  {f.name}: {e.mensagem}")
            continue
        if args.json:
            print(json.dumps({"arquivo": f.name, **r}, ensure_ascii=False, indent=2))
            continue
        extra = []
        if r["planta"]:
            extra.append(r["planta"]["nome"])
        if r["doenca"]:
            extra.append(r["doenca"]["nome"])
        detalhe = f" — {' / '.join(extra)}" if extra else ""
        p = r["probabilidades"]["doente"]
        p_txt = ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{p:.1%}"   # sem falsa certeza
        print(f"{ICONES[r['classificacao']]} {f.name}: {r['classificacao_nome'].upper()} "
              f"(P(doente)={p_txt}, confiança {r['nivel_confianca']}){detalhe}")


if __name__ == "__main__":
    main()
