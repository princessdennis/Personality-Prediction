import flask
from flask import request, jsonify, render_template, Flask

#For model prediction
from static.clean_text import dineise_clean_text  #Must have this for dineise_clean_text to be in "__main__" module
from predict_api import dineise_predict

# # Initialize the app
app = Flask(__name__)


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/result", methods=["POST"])
def receive_texts():
    if request.method == 'POST':
        # print(request.form.to_dict())
        input_texts = request.form["input_texts"]
        name = request.form["name"]

        preds = dineise_predict(input_texts)

        return render_template("result.html", name=name, input_texts=input_texts, pred1=preds[0][1], pred2=preds[1][1], pred3=preds[2][1])

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug=False)










 # <input type="text" name="chat_in" maxlength="1000" >
 #    <!-- Submit button -->
 #    <input type="submit" value="Submit" method="get" >
 #
 #    <p> Here are my predictions!
 #    <br>
 #    {{ chat_in }}
 #    {{ prediction[0]['predictions'] }}
 #    </p>