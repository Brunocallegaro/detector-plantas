// Comunicação com a API Python (FastAPI). Mesma origem em produção; proxy do Vite em desenvolvimento.

async function pedir(caminho, opcoes = {}) {
  let resposta;
  try {
    resposta = await fetch(caminho, opcoes);
  } catch {
    throw new Error("Não foi possível conectar à API. Verifique se o backend está em execução (python api.py).");
  }
  const corpo = await resposta.json().catch(() => ({}));
  if (!resposta.ok) throw new Error(corpo.erro || `A API respondeu com erro ${resposta.status}.`);
  return corpo;
}

export const api = {
  saude: () => pedir("/api/saude"),
  modelo: () => pedir("/api/modelo"),
  estatisticas: () => pedir("/api/estatisticas"),
  historico: ({ limite = 12, deslocamento = 0, classificacao = "" } = {}) => {
    const p = new URLSearchParams({ limite, deslocamento });
    if (classificacao) p.set("classificacao", classificacao);
    return pedir(`/api/historico?${p}`);
  },
  analise: (id) => pedir(`/api/historico/${id}`),
  excluir: (id) => pedir(`/api/historico/${id}`, { method: "DELETE" }),
  analisar: (arquivo) => {
    const form = new FormData();
    form.append("imagem", arquivo);
    return pedir("/api/analisar", { method: "POST", body: form });
  },
};

export const LIMITE_MB = 10;
export const TIPOS_ACEITOS = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export const NOMES = { saudavel: "Saudável", doente: "Doente", inconclusivo: "Inconclusivo" };

export function formatarData(iso) {
  return new Date(iso).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
}

// Probabilidades exibidas sem falsa certeza: nunca "0%" nem "100%".
export const pct = (v, casas = 0) => {
  if (v >= 0.995) return "> 99%";
  if (v < 0.005) return "< 1%";
  return `${(v * 100).toFixed(casas).replace(".", ",")}%`;
};
