import pandas as pd
import random
import sklearn
import csv
from sklearn import tree, svm, naive_bayes, neighbors, ensemble, calibration, gaussian_process, semi_supervised, discriminant_analysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings('ignore')


# Lendo os arquivos tratados de todas as ações
dados = pd.read_csv('dados/dados_tratados_acoes_atuais.csv')
dados.sort_values('DATA', ascending=True, inplace=True)


# Lista com todas as técnicas para serem executadas
tecnicas = [ ('AdaBoostClassifier', sklearn.ensemble.weight_boosting.AdaBoostClassifier()),
             ('BaggingClassifier', sklearn.ensemble.bagging.BaggingClassifier()),
             ('BernoulliNB', sklearn.naive_bayes.BernoulliNB()),
             ('CalibratedClassifierCV', sklearn.calibration.CalibratedClassifierCV()),
             ('DecisionTreeClassifier', sklearn.tree.tree.DecisionTreeClassifier()),
             ('ExtraTreeClassifier', sklearn.tree.tree.ExtraTreeClassifier()),
             ('ExtraTreesClassifier', sklearn.ensemble.forest.ExtraTreesClassifier()),
             #('GaussianNB', sklearn.naive_bayes.GaussianNB()),
             ('GaussianProcessClassifier', sklearn.gaussian_process.gpc.GaussianProcessClassifier()),
             ('GradientBoostingClassifier', sklearn.ensemble.gradient_boosting.GradientBoostingClassifier()),
             ('KNeighborsClassifier', sklearn.neighbors.classification.KNeighborsClassifier()),
             ('LabelPropagation', sklearn.semi_supervised.label_propagation.LabelPropagation()),
             ('LabelSpreading', sklearn.semi_supervised.label_propagation.LabelSpreading()),
             ('LinearDiscriminantAnalysis', sklearn.discriminant_analysis.LinearDiscriminantAnalysis()),
             ('LinearSVC', sklearn.svm.classes.LinearSVC()),
             ('LogisticRegression', sklearn.linear_model.logistic.LogisticRegression(penalty='l2')),
             ('LogisticRegressionCV', sklearn.linear_model.logistic.LogisticRegressionCV()),
             ('MLPClassifier', sklearn.neural_network.multilayer_perceptron.MLPClassifier()),
             ('MultinomialNB', sklearn.naive_bayes.MultinomialNB()),
             ('NearestCentroid', sklearn.neighbors.nearest_centroid.NearestCentroid()),
             #('NuSVC', sklearn.svm.classes.NuSVC()), # erro de outlier
             ('PassiveAggressiveClassifier', sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier()),
             ('Perceptron', sklearn.linear_model.perceptron.Perceptron()),
             ('QuadraticDiscriminantAnalysis', sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()),
             ('RadiusNeighborsClassifier', sklearn.neighbors.classification.RadiusNeighborsClassifier()),
             ('RandomForestClassifier', sklearn.ensemble.forest.RandomForestClassifier()),
             ('RidgeClassifier', sklearn.linear_model.ridge.RidgeClassifier()),
             ('RidgeClassifierCV', sklearn.linear_model.ridge.RidgeClassifierCV()),
             ('SGDClassifier', sklearn.linear_model.stochastic_gradient.SGDClassifier())#,
             #('SVC', sklearn.svm.classes.SVC())
           ]

    



# Função para gravar o log das etapas realizadas
def gravarLog(tipo, mensagem): 
    
    #print(str(datetime.now())+'; ['+tipo+']; ' + str(mensagem))
    
    if tipo == 'resultado':
        nomeArquivo = 'resultados_tecnicas.csv'
        with open(nomeArquivo, 'a', newline='') as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=['acao','accuracy','roc_auc_score','f1_score','log_loss','precision','recall','tecnica','tipo_split','pipeline','parametros','best_estimator','best_params','random_state_split','random_state_random_search','cross_validation_random_search','tamanho_base_treino','tamanho_base_teste','tamanho_base_completa','tem_dados_de_teste_nos_dados_de_treino'], delimiter=';')
            if arquivo.tell() == 0:
                writer.writeheader()
            #writer.writerow({'mensagem': str(mensagem['tecnica'])})  
            
            writer.writerow({
            'acao': mensagem['acao'], 
            'accuracy': mensagem['accuracy'],
            'roc_auc_score': mensagem['roc_auc_score'],
            'f1_score': mensagem['f1_score'],
            'log_loss': mensagem['log_loss'],
            'precision': mensagem['precision'],
            'recall': mensagem['recall'],
            'tecnica': mensagem['tecnica'],
            'tipo_split': mensagem['tipo_split'],
            'pipeline':mensagem['pipeline'],
            'parametros':mensagem['parametros'],
            'best_estimator':mensagem['best_estimator'],
            'best_params':mensagem['best_params'],
            'random_state_split':mensagem['random_state_split'],
            'random_state_random_search':mensagem['random_state_random_search'],
            'cross_validation_random_search':mensagem['cross_validation_random_search'],
            'tamanho_base_treino':mensagem['tamanho_base_treino'],
            'tamanho_base_teste':mensagem['tamanho_base_teste'],
            'tamanho_base_completa':mensagem['tamanho_base_completa'],
            'tem_dados_de_teste_nos_dados_de_treino':mensagem['tem_dados_de_teste_nos_dados_de_treino']
            })
            
            
    
    nomeArquivo = 'log/log.csv'
    with open(nomeArquivo, 'a', newline='') as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=['horario', 'tipo', 'mensagem'], delimiter=';')
        if arquivo.tell() == 0:
            writer.writeheader()
        writer.writerow({'horario': str(datetime.now()),'tipo': tipo, 'mensagem': mensagem})     
        
        
# Função para retornar os dados de uma ação já separados em X e y
def getDadosAcao(acao):
    dados_acao = dados.loc[dados['CODNEG'] == acao]
    
    dados_acao.sort_values('DATA', ascending=True, inplace=True)

    X = dados_acao.copy()
    X.drop(['CODNEG', 'DATA', 'STATUS_POSITIVO'], axis=1, inplace=True)
    y = dados_acao['STATUS_POSITIVO']
    
    return X, y


# Função para separar os dados de treino e teste, escolhendo a forma de separar os dados aleatoriamente
def splitDados(X, y):
    
    aleatorio = getNumeroAleatorio('simples', 6)
    
    random_state = getNumeroAleatorio('random_state')
    
    if aleatorio == 0:
        tipo_split = 'train_test_split 20%'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    elif aleatorio == 1:
        tipo_split = 'train_test_split 25%'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    elif aleatorio == 2:
        tipo_split = 'train_test_split 30%'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    elif aleatorio == 3:
        tipo_split = '30 ultimos dias'
        X_train = X[:-30]
        y_train = y[:-30]
        X_test = X.iloc[-30:]
        y_test = y.iloc[-30:]
    elif aleatorio == 4:
        tipo_split = '60 ultimos dias'
        X_train = X[:-60]
        y_train = y[:-60]
        X_test = X.iloc[-60:]
        y_test = y.iloc[-60:]
    elif aleatorio == 5:
        tipo_split = '90 ultimos dias'
        X_train = X[:-90]
        y_train = y[:-90]
        X_test = X.iloc[-90:]
        y_test = y.iloc[-90:]
    elif aleatorio == 6:
        tipo_split = '180 ultimos dias'
        X_train = X[:-180]
        y_train = y[:-180]
        X_test = X.iloc[-180:]
        y_test = y.iloc[-180:]
    
    
    return X_train, X_test, y_train, y_test, tipo_split, random_state


# Retorna aleatoriamente qual normalizador será utilizado
def getNormalizador():
    normalizador = [(),
                    ('standard_scaler', sklearn.preprocessing.StandardScaler()),
                    ('robust_scaler', sklearn.preprocessing.RobustScaler()),
                    ('min_max_scaler', sklearn.preprocessing.MinMaxScaler()),
                    ('normalizer', sklearn.preprocessing.Normalizer())
                   ]
    
    aleatorio = getNumeroAleatorio('simples', len(normalizador)-1)
    
    if aleatorio == 0:
        parametros = {}
    elif normalizador[aleatorio][0] == 'standard_scaler':
        parametros = {
            'standard_scaler__with_mean': [True, False],
            'standard_scaler__with_std': [True, False]
        }
    elif normalizador[aleatorio][0] == 'robust_scaler':
        parametros = {
            'robust_scaler__with_centering': [True, False],
            'robust_scaler__with_scaling': [True, False]
        }
    elif normalizador[aleatorio][0] == 'min_max_scaler':
        parametros = {
            'min_max_scaler__feature_range': [(0,1), (1,10), (1,100)]
        }
    elif normalizador[aleatorio][0] == 'normalizer':
        parametros = {
            'normalizer__norm': ('l1', 'l2', 'max')
        }
        
    
    return normalizador[aleatorio], parametros


# Retorna aleatoriamente qual redutor de dimensionalidade será utilizado
def getRedutorDimensionalidade():
    redutor_dimensionalidade = [(),
                                ('pca', PCA()),
                                ('nmf', NMF()),
                                ('FastICA', FastICA())]
    
    aleatorio = getNumeroAleatorio('simples', len(redutor_dimensionalidade)-1)
    
    if aleatorio == 0:
        parametros = {}
    elif redutor_dimensionalidade[aleatorio][0] == 'pca':
        parametros = {
            'pca__n_components': [None, 3, 5, 7, 9, 11, 13],
            'pca__whiten': [True, False],
            'pca__svd_solver': ('auto', 'full', 'randomized')
        }
    elif redutor_dimensionalidade[aleatorio][0] == 'nmf':
        parametros = {
            'nmf__n_components': [None, 3, 5, 7, 9, 11, 13],
            'nmf__init': ('random', 'nndsvd', 'nndsvda', 'nndsvdar'),
            'nmf__solver': ('cd', 'mu')
        }
    elif redutor_dimensionalidade[aleatorio][0] == 'FastICA':
        parametros = {
            'FastICA__n_components': [None, 3, 5, 7, 9, 11, 13],
            'FastICA__algorithm': ('parallel', 'deflation'),
            'FastICA__whiten': [True, False]
        }
        
    
    return redutor_dimensionalidade[aleatorio], parametros


# Recebe uma técnica e retorna os parametros da técnica
def getParametrosTecnica(tecnica):
    if tecnica == 'AdaBoostClassifier':
        parametros = {
            'AdaBoostClassifier__n_estimators': [5,25,50,75,100],
            'AdaBoostClassifier__learning_rate': [0.5, 1.0, 1.5],
            'AdaBoostClassifier__algorithm': ('SAMME', 'SAMME.R')
        }
    elif tecnica == 'BaggingClassifier':
        parametros = {
            'BaggingClassifier__n_estimators': [5, 10, 15],
            #'BaggingClassifier__bootstrap': [True, False],
            'BaggingClassifier__bootstrap_features': [True, False],
            'BaggingClassifier__oob_score': [True, False]
        }
    elif tecnica == 'BernoulliNB':
        parametros = {
            'BernoulliNB__alpha': [0, 0.5, 1.0, 1.5, 2],
            'BernoulliNB__fit_prior': [True, False]
        }
    elif tecnica == 'CalibratedClassifierCV':
        parametros = {
            'CalibratedClassifierCV__method': ('sigmoid', 'isotonic'),
            'CalibratedClassifierCV__cv': [3, 5, 10]
        }
    elif tecnica == 'DecisionTreeClassifier':
        parametros = {
            'DecisionTreeClassifier__criterion': ('gini', 'entropy'),
            'DecisionTreeClassifier__splitter': ('best', 'random')            
        }
    elif tecnica == 'ExtraTreeClassifier':
        parametros = {
            'ExtraTreeClassifier__criterion': ('gini', 'entropy'),
            'ExtraTreeClassifier__max_features': ('auto', 'sqrt', 'log2')
        }
    elif tecnica == 'ExtraTreesClassifier':
        parametros = {
            'ExtraTreesClassifier__criterion': ('gini', 'entropy'),
            'ExtraTreesClassifier__max_features': ('auto', 'sqrt', 'log2')
            #
        }
    elif tecnica == 'GaussianProcessClassifier':
        parametros = {
            'GaussianProcessClassifier__n_restarts_optimizer': [0,5,10,15],
            'GaussianProcessClassifier__max_iter_predict': [10, 100, 1000]
        }  
    elif tecnica == 'GradientBoostingClassifier':
        parametros = {
            'GradientBoostingClassifier__loss': ('deviance', 'exponential'),
            'GradientBoostingClassifier__learning_rate': [0.1,0.2,0.001],
            'GradientBoostingClassifier__criterion': ('friedman_mse', 'mse', 'mae'),
            'GradientBoostingClassifier__max_features': ('auto', 'sqrt', 'log2')
        }
    elif tecnica == 'KNeighborsClassifier':
        parametros = {
            'KNeighborsClassifier__n_neighbors': [3,5,7,9],
            'KNeighborsClassifier__weights': ('uniform', 'distance'),
            'KNeighborsClassifier__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
            'KNeighborsClassifier__p': [1,2]
        }
    elif tecnica == 'LabelPropagation':
        parametros = {
            'LabelPropagation__kernel': ('knn', 'rbf'),
            'LabelPropagation__n_neighbors': [3,5,7,9],
            'LabelPropagation__max_iter': [100, 1000, 10000],
            'LabelPropagation__tol': [0.01, 0.001, 0.0001]
        }
    elif tecnica == 'LabelSpreading':
        parametros = {
            'LabelSpreading__kernel': ('knn', 'rbf'),
            'LabelSpreading__n_neighbors': [3,5,7,9],
            'LabelSpreading__max_iter': [100, 1000, 10000],
            'LabelSpreading__tol': [0.01, 0.001, 0.0001]
        }
    elif tecnica == 'LinearDiscriminantAnalysis':
        parametros = {
            'LinearDiscriminantAnalysis__solver': ('svd', 'lsqr'),
            #'LinearDiscriminantAnalysis__shrinkage': (None, 'auto'),
            'LinearDiscriminantAnalysis__tol': [0.001, 0.0001, 0.00001]
        }
    elif tecnica == 'LinearSVC':
        parametros = {
            #'LinearSVC__penalty': ('l1', 'l2'),
            'LinearSVC__loss': ('hinge', 'squared_hinge'),
            #'LinearSVC__dual': [True, False],
            'LinearSVC__tol': [0.001, 0.0001, 0.00001],
            'LinearSVC__multi_class': ('ovr', 'crammer_singer'),
            'LinearSVC__fit_intercept': [True, False],
            'LinearSVC__max_iter': [100, 1000, 10000]
        }
    elif tecnica == 'LogisticRegression':
        parametros = {
            #'LogisticRegression__penalty': ('l2'),
            'LogisticRegression__fit_intercept': [True, False],
            'LogisticRegression__solver': ('newton-cg', 'sag', 'lbfgs'),
            'LogisticRegression__max_iter': [10, 100, 1000, 10000],
            'LogisticRegression__warm_start': [True, False]
        }
    elif tecnica == 'LogisticRegressionCV':
        parametros = {
            'LogisticRegressionCV__fit_intercept': [True, False],
            'LogisticRegressionCV__cv': ['warn', 3, 5, 7, 9],
            'LogisticRegressionCV__solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
            'LogisticRegressionCV__refit': [True, False],
            'LogisticRegressionCV__multi_class': ('ovr', 'auto')
        }
    elif tecnica == 'MLPClassifier':
        parametros = {
            'MLPClassifier__hidden_layer_sizes': [(1,), (100,), (500,), (1,3), (100,3), (500,3)],
            'MLPClassifier__activation': ('identity', 'logistic', 'tanh', 'relu'),
            'MLPClassifier__solver': ('lbfgs', 'sgd', 'adam'),
            'MLPClassifier__learning_rate': ('constant', 'invscaling', 'adaptive'),
            'MLPClassifier__shuffle': [True, False],
            'MLPClassifier__max_iter': [100, 500, 1000],
            'MLPClassifier__tol': [0.001, 0.0001, 0.00001]
        }
    elif tecnica == 'MultinomialNB':
        parametros = {
            'MultinomialNB__alpha': [0.0, 1.0],
            'MultinomialNB__fit_prior': [True, False]
        }
    elif tecnica == 'NuSVC':
        parametros = {
            'NuSVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
            'NuSVC__gamma': ('rbf', 'poly', 'sigmoid'),
            'NuSVC__shrinking': [True, False],
            'NuSVC__decision_function_shape': ('ovo', 'ovr')
        }
    elif tecnica == 'PassiveAggressiveClassifier':
        parametros = {
            'PassiveAggressiveClassifier__fit_intercept': [True, False],
            'PassiveAggressiveClassifier__shuffle': [True, False],
            'PassiveAggressiveClassifier__loss': ('hinge', 'squared_hinge'),
            'PassiveAggressiveClassifier__average': [True, False]
        }
    elif tecnica == 'Perceptron':
        parametros = {
            'Perceptron__penalty': (None, 'l1', 'l2', 'elasticnet'),
            'Perceptron__shuffle': [True, False]
        }
    elif tecnica == 'QuadraticDiscriminantAnalysis':
        parametros = {
            'QuadraticDiscriminantAnalysis__store_covariance': [True, False],
            'QuadraticDiscriminantAnalysis__store_covariances': [True, False],
            'QuadraticDiscriminantAnalysis__tol': [0.001, 0.0001, 0.00001]
        }
    elif tecnica == 'RadiusNeighborsClassifier':
        parametros = {
            'RadiusNeighborsClassifier__weights': ('uniform', 'distance'),
            'RadiusNeighborsClassifier__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto'),
            'RadiusNeighborsClassifier__p': [1, 2]
        }
    elif tecnica == 'RandomForestClassifier':
        parametros = {
            'RandomForestClassifier__criterion': ('gini', 'entropy'),
            'RandomForestClassifier__max_features': ('auto', 'sqrt', 'log2', None),
            #'RandomForestClassifier__bootstrap': [True, False],
            #'RandomForestClassifier__oob_score': [True, False],
            'RandomForestClassifier__warm_start': [True, False],
            'RandomForestClassifier__class_weight': ('balanced', 'balanced_subsample', None)
        }
    elif tecnica == 'RidgeClassifier':
        parametros = {
            'RidgeClassifier__fit_intercept': [True, False],
            'RidgeClassifier__normalize': [True, False],
            'RidgeClassifier__solver': ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'),
            'RidgeClassifier__tol': [0.01, 0.001, 0.0001]
        }
    elif tecnica == 'RidgeClassifierCV':
        parametros = {
            'RidgeClassifierCV__fit_intercept': [True, False],
            'RidgeClassifierCV__normalize': [True, False],
            'RidgeClassifierCV__cv': [3,5,7,9]
        }
    elif tecnica == 'SGDClassifier':
        parametros = {
            'SGDClassifier__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
            'SGDClassifier__shuffle': [True, False],
            #'SGDClassifier__learning_rate': ['constant', 'optimal', 'invscaling'],
            'SGDClassifier__average': [True, False]
        }
    
        
    return parametros



# Função para retornar número aleatório
def getNumeroAleatorio(tipo, maximo=None):
    if tipo == 'cross_validation':
        return random.randint(3,10)
    elif tipo == 'random_state':
        return random.randint(1, 42)
    elif tipo == 'simples':
        return random.randint(0,maximo)

    
# Função para ler e retornar em pandas o csv contendo os melhores resultados das técnicas
def getMelhoresResultados():
    return pd.read_csv('melhores_resultados_tecnicas.csv', sep=';', decimal=',')
    
    
# Função que lê o arquivo (csv) dos melhores resultados das técnicas e retorna o melhor resultado já alcançado pela técnica naquela ação 
def getMelhorResultadoTecnica(acao, tecnica):
    
    melhor_resultado = getMelhoresResultados()
    
    return melhor_resultado.loc[(melhor_resultado['acao'] == acao) & (melhor_resultado['tecnica'] == tecnica)]


# Função para gravar no arquivo (csv) dos melhores resultados das técnicas, o novo resultado/desempenho melhor alcançado/encontrado
def salvarResultadoMelhor(tipo, resultado_melhor):
    gravarLog('desempenho melhor', 'ação: ' + str(resultado_melhor.iloc[0]['acao']) + ', técnica: ' + str(resultado_melhor.iloc[0]['tecnica']))
    
    resultados = getMelhoresResultados()
    
    if tipo == 'novo': # Se o resultado melhor for do tipo novo, significa que não existe nenhum resultado já salvo para a ação e técnica, será adicionado esse novo resultado no arquivo de controle
        resultados['n_execucoes_melhores'] = int(1) # Iniciando contador que vai identificar quantas execuções teve na ação e técnica
        resultados = pd.concat([resultados, resultado_melhor])
    elif tipo == 'melhor': # Se o resultado melhor for do tipo novo, significa que já existe salvo um resultado anterior para a ação e técnica, então esse resultado anterior já salvo será substituido pelo novo resultado melhor
        # Incrementando contador que identifica quantas execuções teve na ação e técnica
        resultados.loc[(resultados['acao'] == resultado_melhor.iloc[0]['acao']) & (resultados['tecnica'] == resultado_melhor.iloc[0]['tecnica']), 'n_execucoes_melhores'] += 1

        for coluna in resultados.columns: # Percorrer todas as colunas, e salvar nas colunas que representa os avaliadores de desempenho, o novo melhor resultado
            if ((coluna != 'tecnica') and (coluna != 'acao') and (coluna != 'tipo_split') and (coluna != 'pipeline') and (coluna != 'parametros') 
                and (coluna != 'best_estimator') and (coluna != 'best_params') and (coluna != 'n_execucoes_melhores')):
                resultados.loc[(resultados['acao'] == resultado_melhor.iloc[0]['acao']) & (resultados['tecnica'] == resultado_melhor.iloc[0]['tecnica']), coluna] = resultado_melhor.iloc[0][coluna]
        
    # Salvar o arquivo com o novo melhor resultado incluso
    arquivo = open('melhores_resultados_tecnicas.csv', 'w')
    resultados.to_csv(arquivo, sep=';', index=False, decimal=',')
    arquivo.close()
    
    
# Função que recebe o y_true e y_pred, aplica os avaliadores de desempenho, faz o comparativo se o desempenho melhorou, e se tiver melhorado chama a função para salvar o novo desempenho no arquivo de controle (csv), posteriormente o modelo desse novo desempenho será salvo pelo joblib (.pkl)
def avaliarDesempenho(acao, tecnica, y_true, y_pred):
    
    resultados = {}
    resultados['acao'] = [acao]
    resultados['tecnica'] = [tecnica]
    # Aplicando avaliadores de desempenho
    resultados['accuracy'] = [sklearn.metrics.accuracy_score(y_true, y_pred)]
    resultados['roc_auc_score'] = [sklearn.metrics.roc_auc_score(y_true, y_pred)]
    resultados['f1_score'] = [sklearn.metrics.f1_score(y_true, y_pred)]
    resultados['log_loss'] = [sklearn.metrics.log_loss(y_true, y_pred)]
    resultados['precision'] = [sklearn.metrics.precision_score(y_true, y_pred)]
    resultados['recall'] = [sklearn.metrics.recall_score(y_true, y_pred)]
    
    
    pd_resultados = pd.DataFrame(data=resultados)
    
    melhor_resultado_anterior = getMelhorResultadoTecnica(acao, tecnica)
    
    melhorou = False
    tipo = ''
    
    if len(melhor_resultado_anterior) > 0: # Verificar se existe algum resultado já salvo para aquela ação e técnica, se já existir é feito o comparativo para verificar se o desempenho atual é melhor do que o melhor desempenho já salvo/encontrado
        
        contador = 0 # Váriavel para contar em quantos avaliadores de desempenho o modelo atual é melhor se comparado com o melhor modelo já salvo/encontrado
        for avaliador in pd_resultados.columns:
            if (avaliador != 'tecnica') and (avaliador != 'acao') and (avaliador != 'accuracy'): # Verificar se a coluna é um avaliador ou não, somente passará se a coluna for um avaliador. A acurácia não será usada no comparativo pois ela não é uma boa metrica para quando se tem dados desbalanceados
                
                if avaliador == 'log_loss': # Para o avaliador log_loss, quanto menor o valor melhor é o desempenho
                    if float(pd_resultados[avaliador]) < float(melhor_resultado_anterior[avaliador]):
                        contador += 1
                elif float(pd_resultados[avaliador]) > float(melhor_resultado_anterior[avaliador]): #Para o restando dos avaliadores, quanto maior o valor melhor é o desempenho
                    contador += 1

        
        metade = 5 / 2 # quantidade de avaliadores (que estamos usando 5) dividido por 2
        if contador > metade: # Se teve mais da metade de avaliadores com desempenho melhor, então consideramos que o modelo é melhor do que o melhor modelo já salvo/encontrado anteriormente
            melhorou = True
            pd_resultados['quantos_avaliadores_melhores_que_o_antecessor'] = contador
            tipo = 'melhor'
    else: # Caso não existe nenhum resultado já salvo para a ação e técnica, então o primeiro modelo será considerado como melhor
        
        melhorou = True
        pd_resultados['quantos_avaliadores_melhores_que_o_antecessor'] = -1 # -1 representa que não existe número para este atributos
        tipo = 'novo'

    return melhorou, tipo, pd_resultados



# Função que executa todas as etapas necessárias para ler os dados da ação, montar o pipeline, coletar os parâmetros, realizar o random search e avaliar o desempenho da técnica
def robo(acao):

    try:

        gravarLog('info', 'Iniciando execução com a ação: ' + acao)

        X, y = getDadosAcao(acao) # Coletando os dados referente a ação
        tamanho_base_completa = len(X) # Salvar quantidade/tamanho de linhas/tuplas

        X_train, X_test, y_train, y_test, tipo_split, random_state_split = splitDados(X, y) # Separando os dados em treino e teste
        tem_dados_de_teste_nos_dados_de_treino = X_test.isin(X_train).values.any()

        # Limpar mémoria apagando váriavel que não será mais utilizada
        del(X)
        del(y)

        pipeline = []
        parametros = {}

        normalizador_pipeline, normalizador_parametros = getNormalizador() # Coletando o normalizador, que será definido aleatoriamente
        if len(normalizador_pipeline) > 0: # Se for definido que terá normalizador, o normalizador é adicionado no pipeline e no array que salva os parâmetros
            pipeline += [normalizador_pipeline]
            parametros.update(normalizador_parametros)

        redutor_dimensionalidade_pipeline, redutor_dimensionalidade_parametros = getRedutorDimensionalidade() # Coletando o redutor de dimensionalidade, que será definido aleatoriamente
        if len(redutor_dimensionalidade_pipeline) > 0: # Se for definido que terá redutor de dimensionalidade, o redutor de dimensionalidade é adicionado no pipeline e no array que salva os parâmetros
            pipeline += [redutor_dimensionalidade_pipeline]
            parametros.update(redutor_dimensionalidade_parametros)


        for tecnica, model in tecnicas: # Para cada técnica presente na variável "tecnicas", coletar os parametros da técnica, realizar o random search e avaliar o seu desempenho

            try:

                pipeline_tecnica = pipeline.copy()
                pipeline_tecnica += [(tecnica, model)] # Adicionando a técnica no pipeline

                parametros_tecnica = parametros.copy()
                parametros_tecnica.update(getParametrosTecnica(tecnica)) # Coletando os parâmetros da técnica e adicionando no array de parametros

                pipeline_tecnica_final = Pipeline(pipeline_tecnica)

                cross_validation = getNumeroAleatorio('cross_validation') # Coletando um número aleatório para representar o cross_validation do random_search
                random_state = getNumeroAleatorio('random_state') # Coletando um número aleatório para representar o random_state do random_search

                # Executando o Random Search com 4 iterações
                modelo = RandomizedSearchCV(n_iter=4, estimator=pipeline_tecnica_final, param_distributions=parametros_tecnica, cv=cross_validation, random_state=random_state)

                modelo.fit(X_train, y_train)

                y_pred = modelo.predict(X_test) # Realizando a predição para os dados de teste

                # Chamando a função que avalia o desempenho da técnica e retorna se o desempenho foi melhor ou não (True ou False)
                melhorou, tipo, desempenho_tecnica = avaliarDesempenho(acao, tecnica, y_test, y_pred)

                # Adicionando informações relevantes e informações para identificar como é o modelo gerado, permitindo que seja reproduzido manualmente caso necessário
                desempenho_tecnica['tipo_split'] = [tipo_split]
                desempenho_tecnica['pipeline'] = [str(pipeline_tecnica)]
                desempenho_tecnica['parametros'] = [str(parametros_tecnica)]
                desempenho_tecnica['best_estimator'] = [modelo.best_estimator_]
                desempenho_tecnica['best_params'] = [modelo.best_params_]
                desempenho_tecnica['random_state_split'] = [random_state_split]
                desempenho_tecnica['random_state_random_search'] = [random_state]
                desempenho_tecnica['cross_validation_random_search'] = [cross_validation]
                desempenho_tecnica['tamanho_base_treino'] = [len(X_train)]
                desempenho_tecnica['tamanho_base_teste'] = [len(X_test)]
                desempenho_tecnica['tamanho_base_completa'] = [tamanho_base_completa]
                desempenho_tecnica['tem_dados_de_teste_nos_dados_de_treino'] = [tem_dados_de_teste_nos_dados_de_treino]

                gravarLog('resultado', desempenho_tecnica)


                if melhorou is True: # Se o desempenho tiver sido melhor, o modelo da técnica será salvo pelo joblib (.pkl)

                    nome_arquivo = 'modelos/' + acao + '_' + tecnica + '.pkl' # Criando o nome do arquivo, ex.: PETR4_KNeighborsClassifier.pkl
                    sklearn.externals.joblib.dump(modelo.best_estimator_, nome_arquivo) # Salvando o modelo
                    salvarResultadoMelhor(tipo, desempenho_tecnica) # Salvando novo resultado melhor
                    print('----------------- ' + str(datetime.now()) + ', ação: ' + acao + ', técnica: ' + tecnica + ', melhorou -------------')

                    # Salvando y_test e y_pred para futuras verificações caso necessário
                    y_test.to_csv('log/y_test_pred/' + acao + '_' + tecnica + '_y_test.csv', index=False, header=True)
                    pd.DataFrame(y_pred, columns=['y_pred']).to_csv('log/y_test_pred/' + acao + '_' + tecnica + '_y_pred.csv', index=False)

            except:
                print('*********** ' + str(datetime.now()) + ', ação: ' + acao + ', técnica: ' + tecnica + ', erro_fit ***********')
                gravarLog('erro', 'erro durante o fit, ação: ' + acao + ', técnica: ' + tecnica)
                pass

            del(pipeline_tecnica)
            del(parametros_tecnica)

        gravarLog('info', 'Finalizado execução com a ação: ' + acao)

    except Exception:

        print('*********** erro_geral, ação: ' + acao + ' ***********')
        gravarLog('erro', 'erro geral, ação: ' + acao)
        pass

    
    
# Executar robô    
acoes = dados.CODNEG.unique()

for i in range(1,100000000000000000000000000000):
    random.shuffle(acoes)
    for acao in acoes:
        robo(acao)
    #Parallel(n_jobs=multiprocessing.cpu_count())(delayed(teste)(acao) for acao in acoes)