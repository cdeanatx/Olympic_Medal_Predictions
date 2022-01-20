#Import main library
import numpy as np

#Import Flask modules
from flask import Flask, request, render_template

#Import joblib to open our machine learning model
import joblib 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'templates')

#create our "home" route using the "parent.html" page
@app.route('/')
def home():
    return render_template('parent.html')


#create our other routes

    
@app.route('/trends')
def trends():
    return render_template('trends.html')


@app.route("/interact")
def interact():
	return render_template("interact.html")


# prediction function for summer
def ValuePredictors(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    model1 = joblib.load('easyensembles_jlib')
    result = model1.predict(to_predict)
    return result[0]

#Set a post method to yield predictions on page for summer 
@app.route("/interact/info1", methods = ['POST'])
def predict1():
    if request.method == 'POST':
        to_predict_list = list(request.form.values())
        result = ValuePredictors(to_predict_list)       
        if result== 0:
            return render_template('interact.html', prediction_text1 = "Sorry, you would not medal in this summer sport")
        else:
            return render_template('interact.html', prediction_text1 = "Congratulations, you would medal in this summer sport!")         
        

# prediction function for winter
def ValuePredictorw(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    model2 = joblib.load('easyensemblew_jlib')
    result = model2.predict(to_predict)
    return result[0]


#Set a post method to yield predictions on page for winter 
@app.route("/interact/info2", methods = ['POST'])
def predict2():
    if request.method == 'POST':
        to_predict_list = list(request.form.values())
        result = ValuePredictorw(to_predict_list)       
        if result== 0:
            return render_template('interact.html', prediction_text2 = "Sorry, you would not medal in this winter sport")
        else:
            return render_template('interact.html', prediction_text2 = "Congratulations, you would medal in this winter sport!")         




#Run app
if __name__ == "__main__":
    app.run(debug=True)
