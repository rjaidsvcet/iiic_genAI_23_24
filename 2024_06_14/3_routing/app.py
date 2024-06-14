from flask import Flask, render_template

app = Flask (__name__)

@app.route ('/greeting/<name>')
def genericGreeting (name):
    return render_template ('index.html', firstname = name)

@app.route ('/firstpage')
def firstFunction ():
    return render_template ('firstPage.html')

@app.route ('/secondpage')
def secondFunction ():
    return render_template ('secondPage.html')

if __name__ == '__main__':
    app.run (debug=True)