from flask import Flask, render_template, request
import generateImages

app = Flask(__name__)


@app.route("/",methods=['POST','GET'])
def home():
    name=''
    if request.method == 'POST':
        output = request.form.to_dict()
        print(output)
        name = output["name"]
        #debug
        name='ok'
        generateImages.makePic(name)
        return render_template("home.html",name=name)
    elif request.method == 'GET':
        return render_template("home.html",name=name)
    return render_template("home.html")




