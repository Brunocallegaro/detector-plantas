# Roteiro de apresentação: Detector de Doenças em Plantas

Duração sugerida: **12 a 15 minutos** + perguntas. Deixe a API aberta (`iniciar_windows.bat`) e o navegador em <http://localhost:8000> **antes** de começar. Na primeira execução, a instalação demora.

## Preparação (no dia anterior)

- [ ] Rodar `iniciar_windows.bat` uma vez (instala tudo e confere se a API sobe).
- [ ] Rodar `testar_windows.bat` e deixar o resultado na tela, ou abrir `docs/resultados_testes.txt`.
- [ ] Separar numa pasta da área de trabalho: 2 folhas saudáveis e 3 doentes de `data\test\`, uma foto que não é folha (ex.: um animal de estimação) e um arquivo `.txt` renomeado para `.jpg`.
- [ ] Opcional: apagar `instancia\` para começar com o histórico vazio.

Sugestões de fotos de `data\test\`, que dão bons resultados e mostram planta e doença:

| Arquivo | Resultado esperado |
|---|---|
| `doente\Potato_Late_blight_020.jpg` | Doente, batata, requeima |
| `doente\Grape_Black_rot_019.jpg` | Doente, videira, podridão-negra |
| `doente\Corn_maize_Common_rust_018.jpg` | Doente, milho, ferrugem-comum |
| `saudavel\Tomato_healthy_051.jpg` | Saudável, tomateiro |
| `doente\Apple_Apple_scab_018.jpg` | Doente com **confiança baixa** (P = 40%). Bom para discutir o limiar |

Confira esses resultados no seu computador antes da apresentação com `python predict.py <arquivo>`.

## Roteiro

### 1. Problema e objetivo (1 min)

Doenças foliares causam perdas importantes na agricultura, e o diagnóstico visual precoce depende de especialistas. O objetivo do projeto é um sistema que recebe a foto de uma folha e indica se ela parece saudável ou doente, com a probabilidade e, quando possível, a planta e a doença. Deixe claro desde o início: **é uma ferramenta de apoio, não substitui o engenheiro agrônomo**.

### 2. Dataset (2 min)

- 1.068 fotos do PlantVillage: 9 culturas (macieira, cerejeira, milho, videira, pessegueiro, pimentão, batata, morangueiro, tomateiro), 24 doenças e folhas saudáveis.
- Divisão **treino 738 / validação 153 / teste 177**, classes equilibradas.
- Ponto forte: foi verificado por hash que **nenhuma imagem se repete entre os conjuntos** (sem vazamento).
- Ponto fraco, dito por você antes de alguém perguntar: as fotos são de laboratório, com fundo neutro.

### 3. O modelo (3 min)

- **CNN MobileNetV3-Small** com *transfer learning*: a rede já sabia reconhecer bordas, texturas e formas (ImageNet, 1,2 milhão de imagens). Trocamos só a última camada para 2 classes e ajustamos com as nossas fotos.
- Por que a MobileNet: é leve (6 MB, cerca de 30 ms por foto em CPU) e roda em computadores comuns ou celulares.
- **Treino:** imagens redimensionadas para 224×224, normalizadas com média e desvio do ImageNet, *data augmentation* (recortes, espelhamentos, rotação, brilho e contraste) para reduzir o *overfitting*. AdamW, 10 épocas, e o melhor modelo é escolhido pela validação.
- **Inferência:** mesmo pré-processamento da avaliação, modelo em `eval()` e softmax, que gera as probabilidades.
- Mostre `resultados/matriz_confusao.png`.

### 4. Melhorias feitas na revisão (2 min)

Escolha três para contar como uma história de engenharia:

1. **Os erros eram todos do mesmo tipo:** os 7 erros do teste eram folhas doentes vistas como saudáveis. Calibramos o limiar na validação (6%) e a sensibilidade subiu de 92,7% para 99,0%, com acurácia de 96,0% → 97,7%. O teste não foi usado para escolher o limiar.
2. **"Um gato saía doente com 99,9%":** classificadores sempre respondem uma das classes. Medimos a distância da foto às imagens de treino e, se for muito diferente, o sistema responde "Inconclusivo".
3. **Bug que apagava o modelo:** o pipeline apagava o modelo e retreinava fora do Windows por causa da ordenação de arquivos. Corrigido, e agora sempre há backup.

Bônus: um segundo modelo (33 classes), treinado a partir do primeiro, identifica a planta (96,6%) e a doença (87,5%) **só quando tem confiança suficiente**.

### 5. Demonstração ao vivo (4 min)

1. **Painel:** mostre os contadores e o quadro "Desempenho medido no conjunto de teste".
2. **Folha doente** (`Potato_Late_blight_020.jpg`): selecione, mostre a pré-visualização e clique em **Analisar folha**. Mostre a classificação, a planta, a doença com o nome científico, a régua com o limiar e a mensagem. Abra **Detalhes técnicos** e mostre as 3 hipóteses.
3. **Folha saudável** (`Tomato_healthy_051.jpg`).
4. **Confiança baixa** (`Apple_Apple_scab_018.jpg`): o modelo diz "doente" com P = 40%. "O sistema não esconde a incerteza."
5. **Foto que não é folha** (ex.: um gato) → **Inconclusivo**. Este costuma ser o momento mais marcante.
6. **Arquivo inválido** (`.txt` renomeado) → mensagem de erro clara.
7. **Histórico:** use o filtro "Doentes". Feche a janela da API, abra de novo e mostre que **o histórico continua lá** (SQLite).
8. Opcional: <http://localhost:8000/docs> para mostrar a API documentada e testável pelo navegador.

### 6. Testes e qualidade (1 min)

44 testes automatizados (43 aprovados, 1 ignorado por falta de GPU): carregamento do modelo, arquitetura idêntica à do treino, acurácia no teste completo, validação de 10 tipos de entrada inválida, todos os endpoints, persistência após reiniciar e um teste que abre o navegador e usa a interface de verdade.

### 7. Limitações e trabalhos futuros (1 min)

- Treinar com fotos de campo (ex.: PlantDoc) e com celulares diferentes.
- Mais imagens por doença (hoje são 15 de treino por doença).
- Mapas de ativação (Grad-CAM) para mostrar **onde** o modelo "olhou".
- Validar em campo junto com agrônomos.

## Perguntas prováveis

**Por que 97,7% e não 100%?** O teste tem 177 imagens e 4 erros: 1 cerejeira com oídio vista como saudável e 3 folhas saudáveis vistas como doentes. Com o limiar padrão seriam 7 erros, todos doenças não detectadas.

**O limiar de 6% não é baixo demais?** Ele foi escolhido na validação, onde maximizou a acurácia, e confirmado no teste. Um valor baixo indica que o modelo tende a subestimar a probabilidade de doença em algumas culturas. Preferimos errar pedindo uma inspeção a mais. A interface mostra a probabilidade e o nível de confiança, nunca só o rótulo.

**Vocês usaram o conjunto de teste para ajustar algo?** Não. O limiar de decisão, o limiar de "fora do padrão" e os limiares de planta e doença vêm só da validação. O teste foi usado apenas para medir.

**O que é "confiança"?** É a probabilidade que o modelo atribui à classe escolhida. Ela não é uma garantia de acerto: redes neurais podem estar confiantes e erradas, especialmente com imagens diferentes das do treino.

**Funciona com foto tirada no campo?** Pode funcionar, mas não foi validado. Por isso o sistema prefere responder "Inconclusivo" quando a foto é muito diferente das de treino.

**Por que dois modelos?** O modelo principal (binário) foi preservado, como pedia o projeto. O detalhado aproveita o que o principal aprendeu e acrescenta a planta e a doença sem alterar a decisão saudável × doente.

**Como funciona a detecção de "fora do padrão"?** Cada foto vira um vetor de 576 números (as "características" que a rede extrai). Comparamos esse vetor com os das 738 fotos de treino. Se até a foto mais parecida estiver longe, além do que acontece com 99% das fotos de validação, a imagem é tratada como fora do padrão.
