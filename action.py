

## Action) pick, place, pour


class Action:
    def __init__(self, param_list, obj, new_obj, merge):
        self.param_list = param_list
        self.obj = obj
        self.new_obj = new_obj
        self.merge = merge

    def pick(self, object):
        self.object = object
        
        

   

