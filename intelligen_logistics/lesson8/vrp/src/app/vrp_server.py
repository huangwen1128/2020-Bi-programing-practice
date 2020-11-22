from flask import Flask
from flask import render_template
#from flask_script import Manager
import tsp
import os

app=Flask(__name__)
#manager = Manager(app)

@app.route('/tsp_chain', methods=['GET','POST'])
def tsp_chain():
    print(os.getcwd())
    otsp = tsp.tsp('../resource/distance.xlsx', '../resource/cities.xlsx',4)
    routing, solution, manager = otsp.work()
    route_list, route_location_list, distance_list = otsp.get_solution(routing, solution, manager)
    return render_template('map_line1.html', route_list=route_location_list)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, threaded=False, debug=False)
