import torch  # Importa o PyTorch, a biblioteca de deep learning usada para criar e treinar redes neurais
from pathlib import Path  # Importa a classe Path, que manipula caminhos de arquivos/pastas de forma independente do sistema operacional

# ------------------------------------------------------------
# CAMINHOS DO PROJETO
# ------------------------------------------------------------

RAIZ: Path = Path(__file__).resolve().parent
# __file__ é o caminho deste arquivo .py
# .resolve() transforma em caminho absoluto (completo), resolvendo atalhos como "..".
# .parent pega a pasta onde este arquivo está, que vira a "raiz" do projeto.
# Obs.: __file__ não existe em notebooks Jupyter/Colab; lá isso geraria erro.

DIR_DADOS: Path = RAIZ / "data"
# Pasta onde ficam os dados (ex.: imagens de treino e teste).
# O operador "/" com objetos Path junta caminhos: RAIZ/data

DIR_MODELOS: Path = RAIZ / "modelos"
# Pasta onde os modelos treinados (pesos da rede) serão salvos.

DIR_RESULTADOS: Path = RAIZ / "resultados"
# Pasta para salvar resultados: gráficos, métricas, relatórios etc.

# ------------------------------------------------------------
# HIPERPARÂMETROS DE TREINAMENTO
# ------------------------------------------------------------

TAMANHO_IMAGEM: int = 224
# Tamanho (em pixels) para o qual as imagens serão redimensionadas: 224x224.
# É o tamanho padrão esperado por redes clássicas como ResNet, VGG, etc.

BATCH_SIZE: int = 32
# Quantidade de imagens processadas de uma vez antes de atualizar os pesos.
# Batches maiores usam mais memória da GPU; menores deixam o treino mais "ruidoso".

EPOCAS: int = 10
# Número de épocas: quantas vezes o modelo vai percorrer TODO o conjunto de treino.

LR: float = 1e-3
# Learning rate (taxa de aprendizado) = 0.001.
# Controla o tamanho do "passo" dado ao ajustar os pesos a cada atualização.
# Muito alta: o treino fica instável; muito baixa: o treino fica lento.

SEED: int = 42
# Semente aleatória. Fixá-la permite reproduzir os mesmos resultados
# (mesma inicialização dos pesos, mesma ordem dos dados embaralhados etc.).
# Aqui ela só é definida; precisa ser aplicada em outro ponto do código (ex.: torch.manual_seed(SEED)).

# ------------------------------------------------------------
# NORMALIZAÇÃO DAS IMAGENS
# ------------------------------------------------------------

MEDIA: list[float] = [0.485, 0.456, 0.406]
# Média de cada canal de cor (R, G, B) calculada no dataset ImageNet.
# Usada para normalizar as imagens: pixel_normalizado = (pixel - média) / desvio.

DESVIO: list[float] = [0.229, 0.224, 0.225]
# Desvio padrão de cada canal (R, G, B), também do ImageNet.
# Usar esses valores é importante ao aproveitar modelos pré-treinados no ImageNet
# (transfer learning), pois eles foram treinados com imagens normalizadas assim.

# ------------------------------------------------------------
# ESCOLHA DO HARDWARE (CPU / GPU)
# ------------------------------------------------------------

def dispositivo() -> torch.device:
    # Define uma função que retorna o melhor dispositivo disponível para rodar o modelo.
    # "-> torch.device" é uma anotação de tipo: indica que a função retorna um torch.device.

    if torch.cuda.is_available():
        # Verifica se há uma GPU NVIDIA com CUDA disponível.
        return torch.device("cuda")
        # Se houver, usa a GPU NVIDIA (a opção mais rápida para deep learning).

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # hasattr(...) verifica se esta versão do PyTorch tem suporte a MPS
        # (versões antigas não têm, e acessar direto daria erro).
        # MPS = Metal Performance Shaders: a GPU dos Macs com chip Apple (M1, M2, M3...).
        return torch.device("mps")
        # Se disponível, usa a GPU do Mac.

    return torch.device("cpu")
    # Se nenhuma GPU estiver disponível, usa o processador (CPU).
    # Funciona em qualquer máquina, mas o treino é bem mais lento.

# ------------------------------------------------------------
# ARQUIVOS DE MODELO E CALIBRAÇÃO (usados na inferência / API)
# ------------------------------------------------------------

CAMINHO_MODELO: Path = DIR_MODELOS / "melhor_modelo.pth"
# Modelo principal (binário: doente x saudavel). É o modelo treinado pelo train.py.

CAMINHO_MODELO_DETALHADO: Path = DIR_MODELOS / "modelo_detalhado.pth"
# Modelo auxiliar (opcional) que tenta identificar a PLANTA e a DOENÇA (33 classes).
# Gerado por train_detalhado.py. Se não existir, o sistema funciona só com o modelo principal.

CAMINHO_CALIBRACAO: Path = DIR_MODELOS / "calibracao.pt"
# Limiar de decisão e referência de "imagem fora do padrão", calculados por calibrar.py
# usando SOMENTE os conjuntos de treino e validação (o teste nunca é usado para ajustar nada).

DIR_BACKUP_MODELOS: Path = DIR_MODELOS / "backup"
# Cópias de segurança de modelos antigos (feitas antes de qualquer novo treino).

# ------------------------------------------------------------
# APLICAÇÃO WEB (API + BANCO DE DADOS)
# ------------------------------------------------------------

DIR_INSTANCIA: Path = RAIZ / "instancia"
# Dados gerados pelo uso do sistema (banco, imagens enviadas, logs).

CAMINHO_BANCO: Path = DIR_INSTANCIA / "historico.db"
# Banco SQLite com o histórico de análises (persiste entre reinicializações).

DIR_UPLOADS: Path = DIR_INSTANCIA / "uploads"
# Cópia das imagens analisadas, para exibir no histórico.

DIR_LOGS: Path = DIR_INSTANCIA / "logs"
# Arquivos de log da API.

DIR_FRONTEND: Path = RAIZ / "frontend" / "dist"
# Interface React já compilada (npm run build). A API serve esses arquivos em http://localhost:8000

TAMANHO_MAX_MB: int = 10
# Tamanho máximo aceito para uma imagem enviada.

LADO_MINIMO_PX: int = 64
# Imagens menores que 64x64 pixels não têm detalhe suficiente para análise.

FORMATOS_ACEITOS: dict[str, str] = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "BMP": "image/bmp"}
# Formatos de imagem aceitos (detectados pelo CONTEÚDO do arquivo, não pela extensão).
