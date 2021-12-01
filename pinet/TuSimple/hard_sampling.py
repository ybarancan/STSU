import numpy

class hard_sampling():
    def __init__(self):
        self.total_num = 0
        self.first_node = None
        self.last_node = None
        self.minimum_loss = 10000
        self.maximum_size = 1000

    def insert(self, node):
        if self.total_num < 1:
            self.total_num += 1
            self.first_node = node
            self.last_node = node
            self.minimum_loss = node.get_loss()
        else:
            target_loss = node.get_loss()
            if self.minimum_loss < target_loss or self.total_num < self.maximum_size:
                if self.first_node.get_loss() < target_loss:
                    self.total_num += 1
                    node.set_next(self.first_node)
                    self.first_node.set_previous(node)
                    self.first_node = node
                else:
                    current_node = self.first_node             
                    while True:
                        if current_node.get_loss() >= target_loss and current_node.get_next() == None:
                            self.total_num += 1    
                            node.set_previous(current_node)
                            current_node.set_next(node)
                            self.last_node = node 
                            self.minimum_loss = target_loss
                            break
                        if current_node.get_loss() >= target_loss and target_loss >= current_node.get_next().get_loss():  
                            self.total_num += 1 
                            node.set_previous(current_node)
                            node.set_next(current_node.get_next())
                            current_node.get_next().set_previous(node)
                            current_node.set_next(node)
                            break
                        current_node = current_node.get_next()
                        if current_node == None:
                            break
        if self.total_num > self.maximum_size:
            self.total_num -= 1
            self.minimum_loss = self.last_node.get_previous().get_loss()
            self.last_node = self.last_node.get_previous()

    def get_list(self):
        data_list = []
        current_node = self.first_node
        while True:
            data_list.append(current_node.get_data())
            current_node = current_node.get_next()
            if current_node == None:
                break
        return data_list

    def get_num(self):
        return self.total_num

class sampling_node():
    def __init__(self, loss = 10000, data = None, previous_node = None, next_node = None):
        self.loss = loss
        self.data = data
        self.previous_node = previous_node
        self.next_node = next_node

    def set_previous(self, previous_node):
        self.previous_node = previous_node

    def set_next(self, next_node):
        self.next_node = next_node

    def set_loss(self, loss):
        self.loss = loss

    def set_data(self, data):
        self.data = data

    def get_previous(self):
        return self.previous_node

    def get_next(self):
        return self.next_node

    def get_loss(self):
        return self.loss

    def get_data(self):
        return self.data
