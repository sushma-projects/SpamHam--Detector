from flask import Flask,render_template,url_for,request
import pickle
#from sklearn.externals import joblib

#loading model and vector
cv = pickle.load(open('cv.pkl','rb')) #loading cv
model = pickle.load(open("spam.pkl","rb")) #loading model

app = Flask(__name__)  #defining flask name

@app.route('/')  #home route
def home():
	return render_template('home.html') #at home route returning home.html to show

@app.route('/predict',methods=['POST'])  #on post request /predict 
def predict():

	if request.method == 'POST':
		message = request.form['message'] #requesting the content of the text field
		data = [message]  #converting text into a list
		vect = cv.transform(data).toarray() #transforming list of sentences into vector form
		my_prediction = model.predict(vect)  #predicting the class (1=spam,0=ham)
	return render_template('result.html',prediction = my_prediction)  #returning result.html with prediction var value as 0 or 1



if __name__ == '__main__':
	app.run(debug=True)  #running the flask app as debug=True