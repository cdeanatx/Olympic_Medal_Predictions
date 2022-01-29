#Import main library
import numpy as np

#Import Flask modules
from flask import Flask, request, render_template, redirect

#Import joblib to open our machine learning model
import joblib 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'templates')

#create our "home" route using the "parent.html" page
@app.route('/')
def home():
    return redirect("/index")

#create our other routes

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/trends')
def trends():
    return render_template('trends.html')

@app.route("/interact")
def interact():
	return render_template("interact.html")

# WINTER 

# prediction function for winter
def ValuePredictorw(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    modelw = joblib.load('easyensemblew_jlib')
    result = modelw.predict(to_predict)
    return result[0]


#Set a post method to yield predictions on page for winter 
@app.route("/interact/infow", methods = ['POST'])

def predictw():
    if request.method == 'POST':

        # Obtain values for conversion and print to double-check progress
        raw_list = list(request.form.values())
        print(raw_list)
        weight_option = request.form['wtoptions']
        print(weight_option)
        print(type(weight_option))
        weight = request.form['weight-data']
        print(weight)
        height_option = request.form['htoptions']
        print(height_option)
        height = request.form['height-data']
        print(height)
        
        # Engage in preprocessing to prepare "to_predict_list" for prediction function
        to_predict_list = raw_list.copy()
        del to_predict_list[7]
        del to_predict_list[5]
        print(to_predict_list)

       # Conversion for weight
        if weight_option == "1":
            new_weight = "{:.2f}".format(float(weight) / 2.205)
            print(new_weight)
            to_predict_list[5] = new_weight
            print(to_predict_list)                       

       # Conversion for height
        if height_option == "1":
            new_height = "{:.2f}".format(float(height) * 2.54)
            print(new_height)
            to_predict_list[4] = new_height
            print(to_predict_list)

        # Return the result
        result = ValuePredictorw(to_predict_list)   
        if result == 0:
            return render_template('interact.html', prediction_textw = "Sorry, you would not medal in this winter sport")
        else:
            return render_template('interact.html', prediction_textw = "Congratulations, you would medal in this winter sport!")         


# SUMMER

# prediction function for summer
def ValuePredictors(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    models = joblib.load('easyensembles_jlib')
    result = models.predict(to_predict)
    return result[0]

#Set a post method to yield predictions on page for summer 
@app.route("/interact/infos", methods = ['POST'])

def predicts():
    if request.method == 'POST':

        # Obtain values for conversion and print to double-check progress
        raw_list = list(request.form.values())
        print(raw_list)
        weight_option = request.form['wtoptions']
        print(weight_option)
        print(type(weight_option))
        weight = request.form['weight-data']
        print(weight)
        height_option = request.form['htoptions']
        print(height_option)
        height = request.form['height-data']
        print(height)
        
        # Engage in preprocessing to prepare "to_predict_list" for prediction function
        to_predict_list = raw_list.copy()
        del to_predict_list[7]
        del to_predict_list[5]
        print(to_predict_list)

       # Conversion for weight
        if weight_option == "1":
            new_weight = "{:.2f}".format(float(weight) / 2.205)
            print(new_weight)
            to_predict_list[5] = new_weight
            print(to_predict_list)                      

       # Conversion for height
        if height_option == "1":
            new_height = "{:.2f}".format(float(height) * 2.54)
            print(new_height)
            to_predict_list[4] = new_height
            print(to_predict_list)

        # Return the result
        result = ValuePredictors(to_predict_list)       
        if result == 0:
            return render_template('interact.html', prediction_texts = "Sorry, you would not medal in this summer sport")
        else:
            return render_template('interact.html', prediction_texts = "Congratulations, you would medal in this summer sport!")         
        
#Run app
if __name__ == "__main__":
    app.run(debug=True)
