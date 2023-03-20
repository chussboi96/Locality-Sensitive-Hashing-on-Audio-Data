from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import functions


features = functions.read_file('audio.pkl')
shingles = functions.read_file('shingles.pkl')
matrix = functions.read_file('shingle_matrix.pkl')
hash_mat = functions.read_file('hash_matrix.pkl')
buckets = functions.read_file('Buckets.pkl')

# the first argument should be the path of the audio file that we get from flask
# don't change the rest of the arguments
# will return a tuple where jac[0]= similarity and jac[1]= file name
# jac = functions.query('C:/Users/lanovo/AudioFiles\\001\\001039.mp3', buckets, hash_mat, shingles, 20, 10)

# send the tuple in this to get x: song details and y:jacquard similarity
# display x and y on result html page
# x, y= get_details(jac, features)


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "SECRET KEY"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['ogg', 'mp3'])


@app.route('/index.html')
def home():
    return render_template("index.html")


@app.route('/lsh.html')
def lsh():
    return render_template("lsh.html")


@app.route('/about.html')
def about():
    return render_template("about.html")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_audio():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            flash('Allowed audio types are mp3 and ogg')
            return redirect(request.url)
    jac = functions.query("static/uploads/" + file_names[0], buckets, hash_mat, shingles, 20, 10)
    x, y = functions.get_details(jac, features)

    ### DISPLAY X AND Y ON WEBPAGE###

    return render_template('index.html', filenames=file_names, x=x, y=str(y*100)+'%')


@app.route('/display/<filename>')
def play_audio(filename):
    # print('play_audio filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
