from flask import Flask 
import predicao

app = Flask(__name__)

@app.route('/') 
def apresentacao(): # Retorna texto explicando como requisitar uma predição
    return u"""
    <html>
       <head><title>robo</title></head>
       <body>
          <h1>Como realizar uma predição?</h1>
          <ul>
            <li> */acao/data/quantidade_de_dias* para ter a predição do desempenho da ação, a partir da data informada. </li>
            <li> Exemplo: /PETR4/22102018/5 </li>
          </ul>
       </body>
    </html>
    """

# Recebe como parâmetros o nome da ação (acao), a data, 
# a quantidade de dias para realizar predição (por padrão é definido como 1), 
# e a quantidade de melhores técnicas que deverá ser feita a predição (por padrão é definido como 1).
# Exemplo: localhost:5000/PETR4/25102018
# Exemplo: localhost:5000/PETR4/25102018/2
# Exemplo: localhost:5000/PETR4/25102018/2/3
@app.route('/<acao>/<data>', defaults={'dias':1, 'n_tecnicas':1})
@app.route('/<acao>/<data>/<dias>', defaults={'n_tecnicas':1})
@app.route('/<acao>/<data>/<dias>/<n_tecnicas>')
def realizarPredicao(acao, data, dias, n_tecnicas=1):
    return predicao.realizarPredicao(acao, data, dias, n_tecnicas)

app.run(host='0.0.0.0', port=5000)