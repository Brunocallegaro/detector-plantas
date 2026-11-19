"""Histórico de análises em SQLite (arquivo instancia/historico.db).

SQLite é um banco de dados em um único arquivo, já incluso no Python:
não precisa instalar nem configurar servidor. O histórico continua
disponível depois de reiniciar a aplicação.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ESQUEMA = """
CREATE TABLE IF NOT EXISTS analises (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    nome_arquivo     TEXT    NOT NULL,          -- nome original enviado pelo usuário
    arquivo_salvo    TEXT    NOT NULL,          -- nome da cópia em instancia/uploads/
    data_hora        TEXT    NOT NULL,          -- ISO 8601, horário local
    classificacao    TEXT    NOT NULL CHECK (classificacao IN ('saudavel', 'doente', 'inconclusivo')),
    confianca        REAL    NOT NULL,          -- probabilidade da classe escolhida (0 a 1)
    prob_doente      REAL    NOT NULL,
    planta           TEXT,
    doenca           TEXT,
    resultado_json   TEXT    NOT NULL           -- resposta completa do modelo
);
CREATE INDEX IF NOT EXISTS idx_analises_data ON analises (data_hora DESC);
"""


class Banco:
    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._trava = threading.Lock()
        with self._conexao() as c:
            c.executescript(ESQUEMA)

    @contextmanager
    def _conexao(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.caminho, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with self._trava, con:          # "with con" = commit automático (ou rollback se der erro)
                yield con
        finally:
            con.close()

    def inserir(self, nome_arquivo: str, arquivo_salvo: str, resultado: dict[str, Any],
                data_hora: datetime | None = None) -> dict[str, Any]:
        data_hora = data_hora or datetime.now()
        with self._conexao() as c:
            cur = c.execute(
                """INSERT INTO analises (nome_arquivo, arquivo_salvo, data_hora, classificacao, confianca,
                                         prob_doente, planta, doenca, resultado_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (nome_arquivo, arquivo_salvo, data_hora.isoformat(timespec="seconds"), resultado["classificacao"],
                 resultado["confianca"], resultado["probabilidades"]["doente"],
                 (resultado.get("planta") or {}).get("nome"), (resultado.get("doenca") or {}).get("nome"),
                 json.dumps(resultado, ensure_ascii=False)),
            )
            novo_id = cur.lastrowid
        registro = self.obter(novo_id)
        assert registro is not None
        return registro

    def obter(self, id_analise: int) -> dict[str, Any] | None:
        with self._conexao() as c:
            linha = c.execute("SELECT * FROM analises WHERE id = ?", (id_analise,)).fetchone()
        return self._converter(linha) if linha else None

    def listar(self, limite: int = 50, deslocamento: int = 0, classificacao: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM analises", []
        if classificacao:
            sql += " WHERE classificacao = ?"
            params.append(classificacao)
        sql += " ORDER BY data_hora DESC, id DESC LIMIT ? OFFSET ?"
        params += [limite, deslocamento]
        with self._conexao() as c:
            return [self._converter(l) for l in c.execute(sql, params).fetchall()]

    def contar(self, classificacao: str | None = None) -> int:
        with self._conexao() as c:
            if classificacao:
                return c.execute("SELECT COUNT(*) FROM analises WHERE classificacao = ?", (classificacao,)).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM analises").fetchone()[0]

    def estatisticas(self) -> dict[str, Any]:
        with self._conexao() as c:
            por_classe = dict(c.execute("SELECT classificacao, COUNT(*) FROM analises GROUP BY classificacao").fetchall())
            doencas = c.execute(
                "SELECT doenca, planta, COUNT(*) AS n FROM analises WHERE doenca IS NOT NULL "
                "GROUP BY doenca, planta ORDER BY n DESC LIMIT 5").fetchall()
            ultima = c.execute("SELECT MAX(data_hora) FROM analises").fetchone()[0]
        return {
            "total": sum(por_classe.values()),
            "saudaveis": por_classe.get("saudavel", 0),
            "doentes": por_classe.get("doente", 0),
            "inconclusivas": por_classe.get("inconclusivo", 0),
            "doencas_mais_frequentes": [{"doenca": d, "planta": p, "quantidade": n} for d, p, n in doencas],
            "ultima_analise": ultima,
        }

    def excluir(self, id_analise: int) -> str | None:
        """Remove o registro e devolve o nome do arquivo salvo (para apagar a imagem)."""
        with self._conexao() as c:
            linha = c.execute("SELECT arquivo_salvo FROM analises WHERE id = ?", (id_analise,)).fetchone()
            if not linha:
                return None
            c.execute("DELETE FROM analises WHERE id = ?", (id_analise,))
        return linha[0]

    @staticmethod
    def _converter(linha: sqlite3.Row) -> dict[str, Any]:
        d = dict(linha)
        d["resultado"] = json.loads(d.pop("resultado_json"))
        return d
