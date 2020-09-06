import pickle
import time

def dump_data(filename, objects):
    with open(filename, 'wb') as out:
        if isinstance(objects, list):
            for o in objects:
                pickle.dump(o, out)
        else:
            pickle.dump(objects, out)
            
def read_data(filename):
    objects = []
    with open(filename, 'rb') as infile:
        while True:
            try:          
                objects.append(pickle.load(infile))
            except EOFError:
                break
    if len(objects) == 1:
        return objects[0]
    return tuple(objects)

