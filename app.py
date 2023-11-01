from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def scale_prediction(prediction):
  if prediction < 0.5:
    return 0
  elif prediction < 1:
    return 1
  elif prediction < 5:
    return 2
  elif prediction < 10:
    return 3
  elif prediction < 35:
    return 4
  elif prediction < 60:
    return 5
  elif prediction < 100:
    return 6
  elif prediction < 200:
    return 7
  elif prediction < 350:
    return 8
  elif prediction < 550:
    return 9
  else:
    return 10

@app.route('/', methods=['GET', 'POST'])
def test():
    data = {}
    print(request.args)
    data['FFMC'] = [float(request.args['FFMC'])]
    
    data['DMC'] = [float(request.args['DMC'])]
    data['DC'] = [float(request.args['DC'])]
    data['ISI'] = [float(request.args['ISI'])]
    data['temp'] = [float(request.args['temp'])]
    data['RH'] = [int(request.args['RH'])]
    data['wind'] = [float(request.args['wind'])]
    print(data)
    # rain = float(request.args['rain'])
    request_df = pd.DataFrame(data)
    print(request_df)

    prediction = model.predict(request_df)[0]
    factor = scale_prediction(prediction)

    return (jsonify({'predictions' : prediction, 'factor' : factor}));
    # return data

if __name__ == "__main__":
    app.run(debug=True)

