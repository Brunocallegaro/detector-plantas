# Relatório técnico: análise, correções e resultados

Todos os números deste documento foram obtidos executando o código do projeto (CPU, Python 3.13, PyTorch 2.15) em 18/11/2026. Os arquivos de origem estão em `resultados/` e `docs/resultados_testes.txt`.

## 1. Análise do projeto recebido

| Arquivo | Situação encontrada |
|---|---|
| `config.py` | Caminhos, hiperparâmetros (224 px, lote 32, 10 épocas, LR 1e-3, semente 42), normalização ImageNet, escolha CUDA → MPS → CPU. Correto. |
| `dataset.py` | `ImageFolder` + augmentation no treino (recorte aleatório 70–100%, espelhamentos, rotação ±20°, cor ±20%); avaliação determinística (Resize 255 → CenterCrop 224). Correto e coerente com o treino. |
| `modelo.py` | Fábrica de MobileNetV3-Small, ResNet18 e CNN própria; carregamento com `strict=True` e `.eval()`. Faltava tratar arquivo ausente ou corrompido. |
| `train.py` | AdamW + CosineAnnealing + CrossEntropy, salva o melhor modelo pela acurácia de validação. Correto, mas sobrescrevia o modelo anterior sem backup. |
| `evaluate.py` | Relatório de classificação, matriz de confusão e lista de erros no teste. Correto. |
| `predict.py`, `api.py` | Mesmo pré-processamento da avaliação, `no_grad`, softmax. Sem histórico, sem CORS, sem correção de rotação EXIF, sem tratamento de transparência, modelo carregado na importação (a API não subia se faltasse o `.pth`). |
| `run_pipeline.py` | Venv, instalação, detecção de mudanças e treino automático. **Dois bugs graves** (seção 2). |
| `melhor_modelo.pth` | Carrega corretamente: MobileNetV3-Small, 244 tensores, classes `['doente', 'saudavel']`, arquitetura idêntica à de `criar_modelo()` (camadas e dimensões conferidas em teste automatizado). |

**Dataset:** 1.068 imagens JPEG 256×256 RGB (subconjunto do PlantVillage), 9 culturas, 24 doenças e 9 classes saudáveis.

| Divisão | Doente | Saudável | Total |
|---|---|---|---|
| Treino | 360 (15 por doença) | 378 (42 por cultura) | 738 |
| Validação | 72 | 81 | 153 |
| Teste | 96 | 81 | 177 |

As classes binárias estão equilibradas (49% × 51%). Verificação por hash MD5: **nenhuma imagem repetida entre treino, validação e teste**, ou seja, sem vazamento de dados.

**Desempenho do modelo original** (`evaluate.py`, teste): acurácia de 96,0%. **Os 7 erros são todos folhas doentes classificadas como saudáveis**, com P(doente) entre 0,02 e 0,40. Como o desempenho é bom, **o modelo não foi substituído nem retreinado**: o arquivo `melhor_modelo.pth` entregue é byte a byte idêntico ao original (conferido com `cmp`).

## 2. Problemas encontrados e correções

| # | Problema | Impacto | Correção |
|---|---|---|---|
| 1 | `run_pipeline.hash_dados()` ordenava com `sorted(Path)`, que **ignora maiúsculas no Windows e não no Linux/macOS** (ex.: `Tomato_Tomato_mosaic` × `Tomato_Tomato_Yellow`) | Fora do Windows, o pipeline achava que as fotos tinham mudado e retreinava | Ordenação explícita por caminho em minúsculas. O hash agora é igual ao salvo no treino original em qualquer sistema (teste `test_dados_identicos_ao_treino_original`) |
| 2 | `run_pipeline` **apagava** `melhor_modelo.pth` e `resultados/` antes de retreinar | Perda do modelo treinado | `fazer_backup_modelo_antigo()` move tudo para `modelos/backup/AAAAMMDD_HHMMSS/` |
| 3 | `hash_codigo()` usava `ast.dump`, cujo formato muda entre versões do Python | Retreino indevido ao trocar de versão do Python | `ast.unparse` (continua ignorando comentários e docstrings) |
| 4 | `train.py` sobrescrevia o modelo sem cópia | Perda do modelo anterior | Backup automático em `modelos/backup/` |
| 5 | Classificador binário responde "doente"/"saudável" para **qualquer** imagem (um gato saiu "doente 99,9%") | Diagnóstico inventado | Verificação de padrão → "Inconclusivo" (seção 4) |
| 6 | Fotos de celular giradas (EXIF) e PNG transparente (fundo preto) | Entrada diferente da do treino | `ImageOps.exif_transpose` e composição sobre fundo branco |
| 7 | Validação de entrada só pela abertura do arquivo | Mensagens pouco claras | Formato real pelo conteúdo, tamanho, resolução mínima, arquivo truncado, imagem de uma cor só, proteção contra "bomba de descompressão" |
| 8 | `torch.load` sem `weights_only=True` | Um `.pth` malicioso poderia executar código | Carregamento seguro e mensagens de erro claras (`ErroModelo`) |
| 9 | Limiar de 50% favorece falsos negativos (seção 3) | Folhas doentes passam como saudáveis | Limiar calibrado na validação |
| 10 | API caía se o modelo não existisse | Sem diagnóstico do problema | API sobe e informa o erro em `/api/saude`; `/api/analisar` responde 503 com a causa |
| 11 | Interface: com um resultado aberto, clicar em "Analisar" não fazia nada (encontrado no teste E2E) | Botão sem efeito | A página é recriada ao clicar na aba já ativa |

Arquivos alterados: `config.py` (só constantes novas), `modelo.py` (carregamento), `train.py` (backup), `run_pipeline.py`, `predict.py`, `api.py`, `requirements.txt`. Como as alterações em `config.py`, `modelo.py` e `train.py` não mudam o resultado do treino, o hash de código em `treino_info.json` foi atualizado (o original está em `modelos/backup/original_entregue/`). O treino continua funcionando: foi executado numa cópia do projeto e gerou o backup antes de sobrescrever.

## 3. Calibração do limiar de decisão (`calibrar.py`)

O modelo devolve P(doente). Em vez de usar 0,5, o limiar é o valor que **maximiza a acurácia na validação**. Os limiares de 0,05 a 0,07 empatam com 99,35%, e foi escolhido o centro do intervalo: **0,06**. **O conjunto de teste não participa da escolha.**

| Conjunto | Limiar | Acurácia | Sensibilidade (doente) | Especificidade (saudável) | Erros |
|---|---|---|---|---|---|
| Validação | 0,50 | 97,4% | 95,8% | 98,8% | 3 FN, 1 FP |
| Validação | 0,06 | 99,3% | 100% | 98,8% | 0 FN, 1 FP |
| **Teste** | 0,50 | 96,0% | 92,7% | 100% | 7 FN, 0 FP |
| **Teste** | **0,06** | **97,7%** | **99,0%** | **96,3%** | 1 FN, 3 FP |

Justificativa: em fitossanidade, deixar passar uma folha doente costuma ser pior do que pedir uma inspeção a mais. O custo é aceitar mais falsos positivos (3 folhas saudáveis de pêssego, pimentão e morango). Um limiar tão baixo também mostra que o modelo é **pouco confiante** em algumas doenças: por isso a interface exibe a probabilidade e o nível de confiança, em vez de só "doente".

Limitação honesta: a validação tem só 153 imagens, e uma única imagem muda a acurácia em 0,65 ponto percentual. O valor exato do limiar tem incerteza.

## 4. Verificação de "imagem fora do padrão"

Para cada foto, o vetor de características (saída do `avgpool` da MobileNet, 576 valores, normalizado) é comparado com os das 738 imagens de treino. A pontuação é **1 − similaridade de cosseno com a imagem de treino mais parecida**. Acima do **percentil 99 da validação (0,304)**, o resultado é "Inconclusivo", e o sistema não mostra diagnóstico, planta nem doença.

Comparação feita durante o desenvolvimento, com 18 fotos que não são folhas (imagens de exemplo do scikit-image e do scikit-learn):

| Características usadas | Não-folhas detectadas | Folhas do teste rejeitadas |
|---|---|---|
| Modelo principal (binário), k=5 | 11/18 | 1,7% |
| **Modelo detalhado (33 classes), k=1** | **17/18** | **1,1%** |

Foi adotado o modelo detalhado: as características do modelo binário foram "comprimidas" para separar só doente × saudável, e um gato ficava parecido com folhas doentes. **Limitação:** fotos reais de folhas tiradas no campo, com fundo de solo, outras folhas ou mão, também tendem a ser marcadas como inconclusivas. Isso é intencional, porque o modelo não foi treinado com esse tipo de imagem.

## 5. Modelo auxiliar: planta e doença (`train_detalhado.py`)

Os nomes dos arquivos trazem a espécie e a condição (ex.: `Potato_Late_blight_020.jpg`), o que permite treinar um classificador de **33 classes** com **as mesmas divisões** de treino, validação e teste.

- **Ponto de partida:** os pesos do modelo principal (que vieram do ImageNet). Só a última camada é trocada (2 → 33 saídas).
- **Balanceamento:** pesos inversamente proporcionais à frequência (classes saudáveis têm 42 fotos, cada doença tem 15) e *label smoothing* de 0,05.
- **Treino:** 15 épocas, AdamW com LR 5e-4, cosseno. Melhor validação: 92,8% (época 12).

| Teste (177 imagens) | Acerto |
|---|---|
| Classe exata (planta + condição) | 91,5% |
| Planta | 96,6% (Macieira 81%, Morangueiro 92%, demais 98–100%) |
| Doença, entre as folhas doentes | 87,5% |

**Regras de exibição** (limiares escolhidos na validação para precisão mínima de 95% na planta e 90% na doença): a planta aparece quando a probabilidade somada da espécie é ≥ 0,30. A doença aparece só se o modelo principal disser "doente", a classe mais provável for uma doença com probabilidade ≥ 0,40 e for coerente com a planta. Caso contrário, a interface mostra "Não identificada com segurança". No teste, a doença foi exibida em 94% das folhas doentes detectadas, com 88,8% de acerto.

O modelo principal continua sendo **o único que decide saudável × doente**.

## 6. Arquitetura do sistema

```
 Navegador (React)                    Backend Python (FastAPI, porta 8000)
 ┌──────────────────┐   HTTP/JSON   ┌────────────────────────────────────────────┐
 │ Painel           │ ────────────▶ │ api.py        rotas, erros JSON, logs       │
 │ Analisar         │  multipart    │   │                                         │
 │ Resultado        │ ◀──────────── │   ├─ inferencia.py                          │
 │ Histórico        │               │   │    validar_e_abrir()   (Pillow)         │
 └──────────────────┘               │   │    modelo principal → P(doente)         │
                                    │   │    modelo detalhado → planta/doença     │
                                    │   │    distância ao treino → inconclusivo   │
                                    │   │    mensagem explicativa                 │
                                    │   └─ banco.py  → instancia/historico.db     │
                                    │                  instancia/uploads/*.jpg    │
                                    └────────────────────────────────────────────┘
```

A API também serve a interface compilada (`frontend/dist`), então só existe um processo para executar. `inferencia.Analisador` é usado pela API, pelo `predict.py`, pela calibração e pelos testes, o que garante exatamente o mesmo pré-processamento em todos os caminhos.

## 7. Testes (resultado real)

`python -m pytest -v` → **43 aprovados, 1 ignorado** (GPU indisponível na máquina de teste), em cerca de 9 s. Saída completa em `docs/resultados_testes.txt`. Também foram verificados manualmente: o fluxo completo no Chrome (capturas em `docs/imagens/`), o layout em tela de celular (390 px), o `run_pipeline.py` detectando "nada mudou" sem retreinar e o `train.py` gerando backup antes de sobrescrever.

## 8. Limitações

1. Imagens de laboratório (PlantVillage): folha isolada, fundo neutro, boa iluminação. Fotos de campo não estão representadas.
2. Conjuntos pequenos (15 fotos de treino por doença, 3 a 4 de teste), então as métricas por doença têm alta incerteza.
3. Só 9 culturas e 24 doenças: uma doença fora da lista nunca será nomeada corretamente.
4. O teste de GPU não pôde ser executado (máquina sem CUDA). O código segue o padrão do PyTorch (`.to(device)` no modelo e nos dados).
