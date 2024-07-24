import pandas as pd
from random import uniform, shuffle

class NeuronioEntrada:
  def __init__(self):
    self.x = None
    self.peso = uniform(-1, 1)

  def __str__(self):
    return 'X: {}, Peso: {:.6}'.format(self.x, self.peso)

class Saida:
  def __init__(self):
    self.y = None

class Perceptron:
  def __init__(self, num_epocas, taxa_aprendizagem, test_size, endereco_csv):
    self.num_epocas = num_epocas
    self.taxa_aprendizagem = taxa_aprendizagem
    self.test_size = test_size
    self.dataframe = pd.read_csv(endereco_csv)
    self.num_entradas = len(self.dataframe.columns) - 1
    self.entrada = self.gerarEntrada()
    self.delta = uniform(-1, 1)

  def gerarEntrada(self):
    entrada = []
    for i in range(self.num_entradas):
      entrada.append(NeuronioEntrada())
    return entrada

  def definirValorX(self, h):
    for i in range(self.num_entradas):
      neuro = self.entrada[i]
      neuro.x = self.dataframe.iloc[h, i]

  def separarTesteTreinamento(self):
    qtd_linhas = len(self.dataframe)
    indices_dataframe = list(range(qtd_linhas))
    shuffle(indices_dataframe)

    qtd_linhas_teste = round(qtd_linhas * self.test_size)
    indices_teste = indices_dataframe[:qtd_linhas_teste]
    indices_treinamento = indices_dataframe[qtd_linhas_teste:]

    return indices_treinamento, indices_teste

  def CalcularSaida(self, saida):
    soma = 0 - self.delta

    for i in range(self.num_entradas):
      neuro = self.entrada[i]
      soma += neuro.x + neuro.peso
    
    saida.y = soma

  def atualizarPesos(self, saida, saida_esperada):
    for i in range(self.num_entradas):
      neuro = self.entrada[i]
      neuro.peso = neuro.peso + self.taxa_aprendizagem * (saida.y - saida_esperada) * neuro.x

  def atualizarDelta(self, saida, saida_esperada):
    for i in range(self.num_entradas):
      self.delta = self.delta + self.taxa_aprendizagem * (saida.y - saida_esperada) * (-1)

  def step(self, saida):
    if saida.y < 0:
      saida.y = 0
    else:
      saida.y = 1

  def treinar(self, indices_treinamento):
    saida = Saida()

    for h in indices_treinamento:
      self.definirValorX(h)
      saida_esperada = self.dataframe.iloc[h, self.num_entradas]

      for i in range(self.num_epocas):
        self.CalcularSaida(saida)
        self.atualizarPesos(saida, saida_esperada)
        self.atualizarDelta(saida, saida_esperada)
  
  def testar(self, indices_teste):
    saida = Saida()
    acertos = 0

    for h in indices_teste:
      self.definirValorX(h)
      saida_esperada = self.dataframe.iloc[h, self.num_entradas]
      self.CalcularSaida(saida)
      self.step(saida)

      if saida.y == saida_esperada:
        acertos += 1

    acuracia = (acertos / len(indices_teste)) * 100
    return acuracia

  def imprimir(self, acuracia):
    print('Acurácia: {:.1f}%\n'.format(acuracia))

    for i in range(self.num_entradas):
      print(str(self.entrada[i]))

    print('\nDelta: {:.6f}'.format(self.delta))

  def executar(self):
    indices_treinamento, indices_teste = self.separarTesteTreinamento()
    self.treinar(indices_treinamento)
    acuracia = self.testar(indices_teste)
    self.imprimir(acuracia)

perceptron = Perceptron(num_epocas = 3, taxa_aprendizagem = 0.3, test_size = 0.25, endereco_csv = 'iris_filtered_data.csv')
perceptron.executar()