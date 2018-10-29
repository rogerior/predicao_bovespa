import pandas as pd
import glob
import json
from sklearn.externals import joblib
from datetime import datetime, timedelta

# Função que recebe uma data, a quantidade de dias, e retorna uma lista com as datas seguintes de acordo com a quantidade de dias definidos.
# Exemplo: data = 25102018, dias = 2, return = ['20181025', '20181026']
def retornaConjuntoDeDias(data, dias):
    data = datetime.strptime(str(data), "%d%m%Y")
    datas = [datetime.strftime(data + timedelta(days=i), "%Y%m%d") for i in range(0,dias)]
    return datas

# Função que recebe uma lista de datas, e retorna os atributos daquela data para ser utilizado na predição
def retornaDadosDatas(datas):
    dimensao_data = pd.read_csv('../dados/dimensao_data_tratado.csv', sep=';')
    dados = dimensao_data.loc[dimensao_data['DATA'].isin(datas)].copy()    
    
    return dados

# Função que recebe uma ação, e a quantidade de técnicas (n_tecnicas) melhores que é para ser usado, 
# e retorna o local e nome dos arquivos que estão salvos o modelo
# Exemplo: acao = 'MGEL4F', n_tecnicas = 3, return = ['../modelos/MGEL4F_AdaBoostClassifier.pkl',
#                                                     '../modelos/MGEL4F_BaggingClassifier.pkl',
#                                                     '../modelos/MGEL4F_RidgeClassifierCV.pkl']
def retornaMelhoresModelos(acao, n_tecnicas):
    resultados = pd.read_csv('../melhores_resultados_tecnicas.csv', sep=';', decimal=',')
    melhores = resultados.loc[resultados['acao'] == acao, ['acao', 'accuracy', 'tecnica']].sort_values('accuracy', ascending=False).head(n_tecnicas)
    tecnicas = melhores['tecnica'].values
    pkls = []
    for tecnica in tecnicas:
        pkls.append('../modelos/'+acao+'_'+tecnica+'.pkl')
    return pkls


# Função que recebe o nome da ação, os dados que serão utilizados para a predição, e a quantidade de melhores técnicas que deverá ser usado, 
# realiza a predição em todas as melhores técnicas selecionadas, e retorna a predicao 
def predict(acao, dados_predicao, n_tecnicas):
    resultado = {}
    
    pkls = retornaMelhoresModelos(acao, n_tecnicas)
    for pkl in pkls:

        tecnica = pkl.split('_')
        tecnica = tecnica[1].split('.')
        tecnica = tecnica[0]

        modelo = joblib.load(pkl)

        identificador = ''

        for i in range(1,len(dados_predicao)+1):

            dado = dados_predicao.iloc[(i-1):i].copy()
            data = str(dado['DATA'].values[0])
            data = datetime.strptime(data, "%Y%m%d")
            data = data.strftime('%d%m%Y')
            identificador = data + '_' + str(tecnica)

            dado.drop('DATA', axis=1, inplace=True)
            
            predict = modelo.predict(dado)

            if predict[0] == 1:
                predict_traduzido = 'Positivo'
            else:
                predict_traduzido = 'Negativo'
                predict = 0

            try:
                predict_proba = modelo.predict_proba(dado)                

                predicao = predict_traduzido + ', probabilidade: ' + str( predict_proba[0][predict]*100 ) + '%'

                resultado.update({identificador: predicao})
            except AttributeError:

                predicao = predict_traduzido + ', probabilidade: Não consta'
                resultado.update({identificador: predicao})

            except IndexError:

                predicao = predict_traduzido + ', probabilidade: Não consta'
                resultado.update({identificador: predicao})


    return resultado

# Função central que recebe um novo pedido de predição, realiza todas as operações necessárias, e retorna um json com a predição de acordo com os parâmetros
def realizarPredicao(acao, data, dias, n_tecnicas):
    datas = retornaConjuntoDeDias(data, int(dias))
    dados_predicao = retornaDadosDatas(datas)
    resultados = predict(acao, dados_predicao, int(n_tecnicas))
    return json.dumps(resultados, ensure_ascii=False).encode('utf8')