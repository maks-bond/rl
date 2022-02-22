import json

class QTableEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return str(item)
                return tuple_str
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {hint_tuples(key): value for key, value in item.items()}
            else:
                return item

        return super(QTableEncoder, self).encode(hint_tuples(obj))

def decode_dict_key(str):
    #Tuple
    # TODO: how to do this less ugly?
    if str.startswith("("):
        return tuple(int(x) for x in str[1:-1].split(','))
    return int(str)

def hinted_tuple_hook(obj):
    if isinstance(obj, dict):
        return {decode_dict_key(key): value for key, value in obj.items()}
    else:
        return obj

def write_q_table(q_table, file_name = 'q.json'):
    with open(file_name, 'w') as handle:
        enc = QTableEncoder()
        jsonstring =  enc.encode(q_table)
        handle.write(jsonstring)

def read_q_table(file_name = 'q.json'):
    with open(file_name, 'r') as handle:
        jsonstring = handle.read()
        return json.loads(jsonstring, object_hook=hinted_tuple_hook)