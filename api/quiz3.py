from flask import Flask, request, jsonify

app = Flask(__name__)

def compare_arrays(arr1, arr2):
    if arr1 == arr2:
        return True
    else:
        return False

@app.route('/check_arrays', methods=['POST'])
def check_arrays():
    try:
        data = request.get_json()

        if 'array1' in data and 'array2' in data:
            array1 = data['array1']
            array2 = data['array2']

            are_arrays_same = compare_arrays(array1, array2)

            return jsonify({"are_arrays_same": are_arrays_same})
        else:
            return jsonify({"error": "Missing 'array1' or 'array2' in request data."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
