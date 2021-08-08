import os
import zipfile
from Main import suicide
from flask import Flask, render_template, request, redirect, url_for ,send_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html') 

upload_dir = os.getcwd() + "/upload_dir"

@app.route("/upload", methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        upload_file = request.files['file']
        if upload_file.filename != '':
            upload_file.save(os.path.join(upload_dir, upload_file.filename))
            return redirect(url_for('download'))
    return render_template('upload.html')


@app.route("/download")
def download():
    suicide()
    return render_template('download.html')


@app.route("/download_files", methods=['GET', 'POST'])
def download_files():
    zipfolder = zipfile.ZipFile('output.zip', 'w', compression=zipfile.ZIP_STORED)
    for root,dirs, files in os.walk('output'):
        for file in files:
            zipfolder.write("output/"+file)
        zipfolder.close()
    return send_file('output.zip', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
