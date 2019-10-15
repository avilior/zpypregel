import collections
import threading

# using Queue or deque -- deque is signifcantly faster
from collections import deque

class PregelVertex():

    def __init__(self, id, value, out_vertices):
        """

        :param id: id of the PregelVertex a string or an int
        :param value: the value associated with this PregelVertex -- this could be anything
        :param out_vertices: neighbour ids
        """
        self.id = id
        self.value = value
        self.out_vertices = out_vertices  # list of neighbor Vertex ids

        # a message is a tuple of (destination_id, value)
        # TODO make it an object (source_id,destionation_id, payload)
        self.incoming_messages = collections.deque()  # the set of incoming messages
        self.outgoing_messages = collections.deque()  # set of outgoing messages

        # pregel states
        self.active = True
        self.superstep = 0

    def getState(self):
        state = { "id" : self.id, "value": self.value, "active" : self.active, "superstep" : self.superstep, "neighbours": [] }
        for v in self.out_vertices:
            state["neighbours"].append(v.id)

        return state

    @staticmethod
    def createPregelMsg(source, destination, msg):
        return {"from": source, "to": destination, "msg": msg}

class PregelWorker(threading.Thread):

    def __init__(self, verticesDB, vertices_id, updateFunction):
        threading.Thread.__init__(self)
        self.vertices_id = vertices_id
        self.updateFunction = updateFunction
        self.verticesDB = verticesDB

    def run(self):
        self.superstep()

    def superstep(self):
        """Completes a single superstep for all the vertices in
        self."""
        for vertex_id in self.vertices_id:
            if self.verticesDB[vertex_id].active:
                self.updateFunction(self.verticesDB[vertex_id])
