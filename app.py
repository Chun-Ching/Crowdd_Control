from flask import Flask, render_template, url_for, redirect, request, jsonify, abort
import requests
import torch
import cv2
import os
import time
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
app.config['COUNT'] = 0
app.config['MAX_COUNT'] = 10
app.config['PERSON_IN'] = 0
app.config['PERSON_OUT'] = 0
session = requests.Session()

@app.route('/')
def index():
    return render_template('index.html', demo_int = app.config['COUNT'])

@app.route('/send', methods=['GET', 'POST'])
def count():
 
    if request.args.get('data') == '+':
        app.config.update(COUNT=app.config['COUNT'] + 1)
        app.config.update(PERSON_IN=app.config['PERSON_IN'] + 1)
    elif request.args.get('data') == '-':
        if app.config['COUNT'] > 0:
            app.config.update(COUNT=app.config['COUNT'] - 1)
            app.config.update(PERSON_OUT=app.config['PERSON_OUT'] + 1)

    print('目前人數:', app.config['COUNT'])
    if app.config['COUNT'] >= 5 and app.config['COUNT'] < app.config['MAX_COUNT']:

        r = session.post('https://maker.ifttt.com/trigger/Bang_Line/with/key/dy8myiX8pZLNKmbsKGKyBL', 
                        params={"value1":f"目前室內人數為 :{app.config['COUNT']}","value2":f"人數上限為 :{app.config['MAX_COUNT']}"})
    elif app.config['COUNT'] >= app.config['MAX_COUNT']:

        r = session.post('https://maker.ifttt.com/trigger/Bang_Line/with/key/dy8myiX8pZLNKmbsKGKyBL', 
                        params={"value1":f"人數上限為 :{app.config['MAX_COUNT']}","value2":f"目前室內人數已滿，請勿進入"})
    socketio.emit('count', {'data' : app.config['COUNT']}, broadcast=True)

    return ''
        
@app.route('/trig', methods=['GET', 'POST'])
def trig_time():
 
    t1_day = request.args.get('data1')
    t2_day = request.args.get('data2')
    print(t1_day, t2_day)

    session.post('https://maker.ifttt.com/trigger/Person_Sheets/with/key/dy8myiX8pZLNKmbsKGKyBL', 
                        params={"value1":f"{t1_day} ~ {t2_day}", "value2":app.config['PERSON_IN'], "value3":app.config['PERSON_OUT']})
    app.config.update(PERSON_IN=0)
    app.config.update(PERSON_OUT=0)
    return ""


if __name__ == "__main__":
    # COUNT = 0
    # app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

