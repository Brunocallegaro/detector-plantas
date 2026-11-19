import { useCallback, useEffect, useRef, useState } from "react";
import { api, formatarData, LIMITE_MB, NOMES, pct, TIPOS_ACEITOS } from "./api.js";

/* ---------------- rotas simples por hash (#/analisar, #/historico, #/analise/12) ---------------- */

function useRota() {
  const ler = () => window.location.hash.replace(/^#/, "") || "/";
  const [rota, setRota] = useState(ler);
  useEffect(() => {
    const mudar = () => { setRota(ler()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", mudar);
    return () => window.removeEventListener("hashchange", mudar);
  }, []);
  return rota;
}

const ir = (rota) => { window.location.hash = rota; };

export default function App() {
  const rota = useRota();
  const [saude, setSaude] = useState(null);
  const [versao, setVersao] = useState(0);   // clicar na aba já aberta reinicia a página (ex.: nova análise)

  useEffect(() => {
    api.saude().then(setSaude).catch((e) => setSaude({ status: "offline", erro: e.message }));
  }, []);

  let pagina;
  const detalhe = rota.match(/^\/analise\/(\d+)$/);
  if (rota === "/analisar") pagina = <Analisar />;
  else if (rota === "/historico") pagina = <Historico />;
  else if (detalhe) pagina = <AnaliseSalva id={Number(detalhe[1])} />;
  else pagina = <Painel />;

  const aba = detalhe ? "/historico" : rota;

  return (
    <>
      <a className="pular" href="#conteudo">Ir para o conteúdo</a>
      <header className="topo">
        <a className="marca" href="#/">
          <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M27 5C14 5 5 11 5 22c0 2 .4 3.6 1 5 2-6 7-11 14-14-6 4-10 9-12 15 1.5.6 3 1 5 1 11 0 14-11 14-24z" /></svg>
          <span>Detector de Doenças em Plantas</span>
        </a>
        <nav aria-label="Principal">
          {[["/", "Painel"], ["/analisar", "Analisar"], ["/historico", "Histórico"]].map(([r, nome]) => (
            <a key={r} href={`#${r}`} aria-current={aba === r ? "page" : undefined}
              onClick={() => rota === r && setVersao((v) => v + 1)}>{nome}</a>
          ))}
        </nav>
        <StatusApi saude={saude} />
      </header>
      <main id="conteudo" tabIndex={-1} key={`${rota}-${versao}`}>{pagina}</main>
      <footer className="rodape">
        <p>
          Resultados gerados automaticamente por um modelo de aprendizado de máquina. A confiança informada não garante
          que o diagnóstico esteja correto, e a análise não substitui a avaliação de um profissional da área agronômica.
        </p>
      </footer>
    </>
  );
}

function StatusApi({ saude }) {
  if (!saude) return <span className="status">Conectando…</span>;
  const ok = saude.status === "online";
  const texto = ok ? `Modelo ativo (${saude.dispositivo?.toUpperCase()})` : saude.erro_modelo || saude.erro || "API indisponível";
  return <span className={`status ${ok ? "ok" : "falha"}`} title={texto}><i aria-hidden="true" />{ok ? texto : "API indisponível"}</span>;
}

/* ---------------- Painel ---------------- */

function Painel() {
  const [est, setEst] = useState(null);
  const [recentes, setRecentes] = useState([]);
  const [modelo, setModelo] = useState(null);
  const [erro, setErro] = useState("");

  useEffect(() => {
    Promise.all([api.estatisticas(), api.historico({ limite: 4 })])
      .then(([e, h]) => { setEst(e); setRecentes(h.itens); })
      .catch((e) => setErro(e.message));
    api.modelo().then(setModelo).catch(() => {});
  }, []);

  return (
    <section className="pagina">
      <div className="painel-abertura">
        <h1>Fotografe uma folha. O modelo diz se ela parece saudável ou doente.</h1>
        <p className="lead">
          Uma rede neural MobileNetV3 treinada com 738 fotos de folhas de nove culturas
          {modelo?.modelo_detalhado?.disponivel ? " identifica, quando consegue, a planta e a doença." : "."}
        </p>
        <a className="botao primario" href="#/analisar">Analisar uma folha</a>
      </div>

      {erro && <Aviso tipo="erro">{erro}</Aviso>}

      <div className="painel-grade">
        <section className="bloco" aria-labelledby="t-contagem">
          <h2 id="t-contagem">Análises realizadas</h2>
          {est ? <Contagem est={est} /> : !erro && <p className="suave">Carregando…</p>}
        </section>

        <section className="bloco" aria-labelledby="t-modelo">
          <h2 id="t-modelo">Desempenho medido no conjunto de teste</h2>
          {modelo ? <Desempenho modelo={modelo} /> : <p className="suave">Carregando…</p>}
        </section>
      </div>

      <section className="bloco" aria-labelledby="t-recentes">
        <div className="bloco-topo">
          <h2 id="t-recentes">Últimas análises</h2>
          {recentes.length > 0 && <a href="#/historico">Ver histórico completo</a>}
        </div>
        {recentes.length === 0 && est && (
          <p className="vazio">Nenhuma análise ainda. <a href="#/analisar">Envie a primeira foto</a> para começar o histórico.</p>
        )}
        <ListaAnalises itens={recentes} />
      </section>
    </section>
  );
}

function Contagem({ est }) {
  const partes = [
    ["saudavel", "Saudáveis", est.saudaveis],
    ["doente", "Doentes", est.doentes],
    ["inconclusivo", "Inconclusivas", est.inconclusivas],
  ];
  return (
    <div className="contagem">
      <p className="total"><strong>{est.total}</strong> {est.total === 1 ? "folha analisada" : "folhas analisadas"}</p>
      <div className="barra" role="img"
        aria-label={partes.map(([, n, v]) => `${n}: ${v}`).join(", ")}>
        {est.total > 0
          ? partes.map(([c, , v]) => v > 0 && <span key={c} className={`seg ${c}`} style={{ flexGrow: v }} />)
          : <span className="seg vazio" style={{ flexGrow: 1 }} />}
      </div>
      <ul className="legenda">
        {partes.map(([c, nome, v]) => (
          <li key={c}><i className={`ponto ${c}`} aria-hidden="true" />{nome}<strong>{v}</strong></li>
        ))}
      </ul>
      {est.doencas_mais_frequentes.length > 0 && (
        <>
          <h3>Doenças mais identificadas</h3>
          <ul className="frequentes">
            {est.doencas_mais_frequentes.map((d) => (
              <li key={d.doenca + d.planta}>{d.doenca} <span className="suave">em {d.planta.toLowerCase()}</span><strong>{d.quantidade}</strong></li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Desempenho({ modelo }) {
  const m = modelo.metricas || {};
  const bin = m.binario_teste_calibrado;
  const det = m.detalhado || {};
  const linhas = [];
  if (bin) {
    linhas.push(["Acerto saudável × doente", pct(bin.acuracia, 1)]);
    linhas.push(["Folhas doentes detectadas (sensibilidade)", pct(bin.sensibilidade_doente, 1)]);
    linhas.push(["Folhas saudáveis reconhecidas (especificidade)", pct(bin.especificidade_saudavel, 1)]);
  }
  if (det.planta_teste?.precisao_quando_exibido != null) linhas.push(["Acerto da planta, quando exibida", pct(det.planta_teste.precisao_quando_exibido, 1)]);
  if (det.doenca_teste?.precisao_quando_exibido != null) linhas.push(["Acerto da doença, quando exibida", pct(det.doenca_teste.precisao_quando_exibido, 1)]);
  return (
    <>
      {linhas.length ? (
        <dl className="metricas">
          {linhas.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
        </dl>
      ) : <p className="suave">Calibração não encontrada. Rode <code>python calibrar.py</code> para medir o desempenho.</p>}
      <p className="nota">
        Teste com 177 fotos que o modelo nunca viu, todas do mesmo banco de imagens (folhas isoladas sobre fundo neutro).
        Em fotos de campo o desempenho tende a ser menor.
      </p>
    </>
  );
}

/* ---------------- Analisar ---------------- */

function Analisar() {
  const [arquivo, setArquivo] = useState(null);
  const [previa, setPrevia] = useState("");
  const [erro, setErro] = useState("");
  const [enviando, setEnviando] = useState(false);
  const [registro, setRegistro] = useState(null);
  const [arrastando, setArrastando] = useState(false);
  const entrada = useRef(null);

  useEffect(() => () => previa && URL.revokeObjectURL(previa), [previa]);

  const escolher = useCallback((f) => {
    setErro(""); setRegistro(null);
    if (!f) return;
    if (!TIPOS_ACEITOS.includes(f.type)) { setErro(`O arquivo “${f.name}” não é uma imagem JPG, PNG, WEBP ou BMP.`); return; }
    if (f.size > LIMITE_MB * 1024 * 1024) { setErro(`A imagem tem ${(f.size / 1048576).toFixed(1)} MB. O limite é ${LIMITE_MB} MB.`); return; }
    setArquivo(f);
    setPrevia(URL.createObjectURL(f));
  }, []);

  async function analisar() {
    if (!arquivo) return;
    setEnviando(true); setErro("");
    try {
      setRegistro(await api.analisar(arquivo));
    } catch (e) {
      setErro(e.message);
    } finally {
      setEnviando(false);
    }
  }

  function recomecar() {
    setArquivo(null); setPrevia(""); setRegistro(null); setErro("");
    if (entrada.current) entrada.current.value = "";
  }

  if (registro) {
    return (
      <section className="pagina">
        <h1 className="titulo-pagina">Resultado da análise</h1>
        <Ficha registro={registro} />
        <div className="acoes">
          <button className="botao primario" onClick={recomecar}>Analisar outra folha</button>
          <a className="botao" href="#/historico">Ver no histórico</a>
        </div>
      </section>
    );
  }

  return (
    <section className="pagina">
      <h1 className="titulo-pagina">Analisar uma folha</h1>
      <p className="lead">
        Para um resultado mais confiável, fotografe uma única folha, bem iluminada, ocupando a maior parte da imagem
        sobre um fundo liso.
      </p>

      <div
        className={`area-envio ${arrastando ? "arrastando" : ""} ${previa ? "com-previa" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setArrastando(true); }}
        onDragLeave={() => setArrastando(false)}
        onDrop={(e) => { e.preventDefault(); setArrastando(false); escolher(e.dataTransfer.files?.[0]); }}
      >
        {previa ? (
          <figure className="previa">
            <img src={previa} alt={`Pré-visualização de ${arquivo.name}`} />
            <figcaption>{arquivo.name} ({(arquivo.size / 1024).toFixed(0)} KB)</figcaption>
          </figure>
        ) : (
          <div className="convite">
            <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M40 8C21 8 8 17 8 33c0 3 .6 5 1.5 7 3-9 10-16 20-21-9 6-15 13-17 22 2 1 4.5 1.5 7 1.5C36 42.5 40 26 40 8z" /></svg>
            <p>Arraste a foto até aqui ou</p>
          </div>
        )}
        <label className="botao">
          {previa ? "Trocar imagem" : "Selecionar imagem"}
          <input ref={entrada} type="file" accept={TIPOS_ACEITOS.join(",")} className="oculto"
            onChange={(e) => escolher(e.target.files?.[0])} />
        </label>
        <p className="suave pequeno">JPG, PNG, WEBP ou BMP, até {LIMITE_MB} MB.</p>
      </div>

      {erro && <Aviso tipo="erro">{erro}</Aviso>}

      <div className="acoes">
        <button className="botao primario" onClick={analisar} disabled={!arquivo || enviando} aria-busy={enviando}>
          {enviando ? <><span className="girando" aria-hidden="true" />Analisando…</> : "Analisar folha"}
        </button>
        {arquivo && !enviando && <button className="botao" onClick={recomecar}>Limpar</button>}
      </div>
      <p className="sr" aria-live="polite">{enviando ? "Analisando a imagem, aguarde." : ""}</p>
    </section>
  );
}

/* ---------------- Ficha de resultado (o "espécime de herbário") ---------------- */

function Ficha({ registro }) {
  const r = registro.resultado;
  const c = r.classificacao;
  return (
    <article className={`ficha ${c}`} aria-labelledby={`ficha-${registro.id}`}>
      <div className="especime">
        <img src={registro.url_imagem} alt={`Folha enviada: ${registro.nome_arquivo}`} />
      </div>

      <div className="etiqueta">
        <p className="numero">Análise nº {registro.id}</p>
        <h2 id={`ficha-${registro.id}`} className="veredito">{NOMES[c]}</h2>

        <dl className="campos">
          <div><dt>Planta</dt><dd>{r.planta ? <>{r.planta.nome} <span className="suave">({pct(r.planta.confianca)})</span></> : <span className="suave">Não identificada com segurança</span>}</dd></div>
          {c === "doente" && (
            <div><dt>Doença</dt><dd>{r.doenca
              ? <>{r.doenca.nome} <span className="suave">({pct(r.doenca.confianca)})</span>{r.doenca.nome_cientifico && <em className="cientifico">{r.doenca.nome_cientifico}</em>}</>
              : <span className="suave">Não identificada com segurança</span>}</dd></div>
          )}
          {c !== "inconclusivo" && (
            <div><dt>Confiança</dt><dd>{r.nivel_confianca[0].toUpperCase() + r.nivel_confianca.slice(1)} <span className="suave">({pct(r.confianca)})</span></dd></div>
          )}
          <div><dt>Data e horário</dt><dd>{formatarData(registro.data_hora)}</dd></div>
          <div><dt>Arquivo</dt><dd className="arquivo">{registro.nome_arquivo}</dd></div>
        </dl>

        {c !== "inconclusivo" && <Regua p={r.probabilidades.doente} limiar={r.limiar_doente} />}

        <p className="mensagem">{r.mensagem}</p>

        <details className="tecnico">
          <summary>Detalhes técnicos</summary>
          <dl>
            <div><dt>Probabilidade de doença (modelo principal)</dt><dd>{pct(r.probabilidades.doente, 1)}{c === "inconclusivo" && " — não utilizada, imagem fora do padrão"}</dd></div>
            <div><dt>Limiar de decisão</dt><dd>{pct(r.limiar_doente, 0)}</dd></div>
            {r.pontuacao_padrao != null && <div><dt>Distância ao padrão de treino</dt><dd>{r.pontuacao_padrao.toFixed(3)} (limite {r.limiar_padrao.toFixed(3)})</dd></div>}
            <div><dt>Tempo de processamento</dt><dd>{Math.round(r.tempo_ms)} ms</dd></div>
          </dl>
          {r.top3_detalhado?.length > 0 && (
            <>
              <h3>Hipóteses do modelo detalhado</h3>
              <ol className="hipoteses">
                {r.top3_detalhado.map((h) => <li key={h.classe}>{h.planta}: {h.condicao} <span className="suave">{pct(h.probabilidade, 1)}</span></li>)}
              </ol>
            </>
          )}
        </details>
        <p className="aviso-pro">{r.aviso}</p>
      </div>
    </article>
  );
}

function Regua({ p, limiar }) {
  return (
    <figure className="regua">
      <figcaption>Probabilidade de doença estimada: <strong>{pct(p, 1)}</strong>
        <span className="suave"> (a marca vertical é o limiar de decisão, {pct(limiar)})</span></figcaption>
      <div className="trilho" role="img" aria-label={`Probabilidade de doença ${pct(p, 1)}; acima de ${pct(limiar)} a folha é classificada como doente`}>
        <span className="preenchido" style={{ width: `${p * 100}%` }} />
        <span className="limiar" style={{ left: `${limiar * 100}%` }} title={`Limiar ${pct(limiar)}`} />
      </div>
      <div className="escala" aria-hidden="true"><span>0%</span><span>Saudável ← → Doente</span><span>100%</span></div>
    </figure>
  );
}

/* ---------------- Histórico ---------------- */

const FILTROS = [["", "Todas"], ["saudavel", "Saudáveis"], ["doente", "Doentes"], ["inconclusivo", "Inconclusivas"]];
const POR_PAGINA = 12;

function Historico() {
  const [filtro, setFiltro] = useState("");
  const [itens, setItens] = useState([]);
  const [total, setTotal] = useState(0);
  const [carregando, setCarregando] = useState(true);
  const [erro, setErro] = useState("");

  const carregar = useCallback(async (deslocamento, acumular) => {
    setCarregando(true); setErro("");
    try {
      const h = await api.historico({ limite: POR_PAGINA, deslocamento, classificacao: filtro });
      setTotal(h.total);
      setItens((antes) => (acumular ? [...antes, ...h.itens] : h.itens));
    } catch (e) {
      setErro(e.message);
    } finally {
      setCarregando(false);
    }
  }, [filtro]);

  useEffect(() => { carregar(0, false); }, [carregar]);

  return (
    <section className="pagina">
      <h1 className="titulo-pagina">Histórico de análises</h1>
      <div className="filtros" role="group" aria-label="Filtrar por resultado">
        {FILTROS.map(([v, nome]) => (
          <button key={v} className="filtro" aria-pressed={filtro === v} onClick={() => setFiltro(v)}>{nome}</button>
        ))}
      </div>
      {erro && <Aviso tipo="erro">{erro}</Aviso>}
      {!carregando && !erro && itens.length === 0 && (
        <p className="vazio">{filtro ? "Nenhuma análise com esse resultado." : <>O histórico está vazio. <a href="#/analisar">Analise uma folha</a> para registrar a primeira.</>}</p>
      )}
      <ListaAnalises itens={itens} />
      <div className="acoes centro">
        {itens.length < total && (
          <button className="botao" disabled={carregando} onClick={() => carregar(itens.length, true)}>
            {carregando ? "Carregando…" : `Carregar mais (${total - itens.length} restantes)`}
          </button>
        )}
      </div>
    </section>
  );
}

function ListaAnalises({ itens }) {
  if (!itens.length) return null;
  return (
    <ul className="lista-analises">
      {itens.map((a) => (
        <li key={a.id}>
          <a href={`#/analise/${a.id}`} className={`item ${a.classificacao}`}>
            <img src={a.url_imagem} alt="" loading="lazy" />
            <span className="item-texto">
              <strong>{NOMES[a.classificacao]}</strong>
              <span>{a.doenca ? `${a.doenca}${a.planta ? ` em ${a.planta.toLowerCase()}` : ""}` : a.planta || "Planta não identificada"}</span>
              <time dateTime={a.data_hora}>{formatarData(a.data_hora)}</time>
            </span>
          </a>
        </li>
      ))}
    </ul>
  );
}

function AnaliseSalva({ id }) {
  const [registro, setRegistro] = useState(null);
  const [erro, setErro] = useState("");

  useEffect(() => { api.analise(id).then(setRegistro).catch((e) => setErro(e.message)); }, [id]);

  async function excluir() {
    if (!window.confirm(`Excluir a análise nº ${id} do histórico? A imagem também será apagada.`)) return;
    try { await api.excluir(id); ir("/historico"); } catch (e) { setErro(e.message); }
  }

  return (
    <section className="pagina">
      <p><a href="#/historico">Voltar ao histórico</a></p>
      {erro && <Aviso tipo="erro">{erro}</Aviso>}
      {registro && (
        <>
          <Ficha registro={registro} />
          <div className="acoes"><button className="botao perigo" onClick={excluir}>Excluir análise</button></div>
        </>
      )}
    </section>
  );
}

function Aviso({ tipo, children }) {
  return <p className={`aviso ${tipo}`} role="alert">{children}</p>;
}
