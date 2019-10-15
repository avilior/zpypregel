import collections
import logging
from pathlib import Path
from pregel import PregelWorker, PregelVertex
import random
import threading
import functools
from numpy import mat, eye, zeros, ones, linalg

LOG = logging.getLogger(Path(__file__).name)

verticesDB = None

def pregelMaster(verticesDB, pageRankUpdateFunction):
    LOG.info("PregelMaster: Starting")

    # create partition for workers

    # Returns a dict with keys 0,...,self.num_workers-1
    # representing the worker threads.  The corresponding values are
    # lists of vertices assigned to that worker."""
    partition = collections.defaultdict(list)
    for vertex in verticesDB.values():
        p = hash(vertex.id) % numberOfWorkers
        partition[p].append(vertex.id)

    try:
        while True:
            # check active -- if none are active we are done
            if not any([v.active for v in verticesDB.values()]):
                LOG.info("Master: Vertices or not active")
                break

            LOG.info("Master: distributing work for another super step")
            workers = []
            for vertex_list in partition.values():
                worker = PregelWorker(verticesDB, vertex_list, pageRankUpdateFunction)
                workers.append(worker)
                worker.start()

            LOG.info("Master: waiting for workers to complete super step")
            for worker in workers:
                worker.join()

            # todo distribute the messages

            LOG.info("Master: compute complete. Distributing messages")

            # update each pregel vertex by incrementing the superstep and clearing the incoming messages
            for pv in verticesDB.values():
                pv.superstep += 1
                pv.incoming_messages.clear() # clear the incoming messages -- there really should be none

            # process the outgoing message of each pregel vertex
            for pv in verticesDB.values():
                # process the outgoing message for each pv
                for pm in pv.outgoing_messages:
                    #LOG.debug(F"Master from {pm['from']} forwarding to {pm['to']} msg: {pm['msg']}")
                    # send the message to the PregelVertex
                    verticesDB[pm['to']].incoming_messages.append(pm)

                pv.outgoing_messages.clear() # after iterating clear them outgoing messages

            LOG.info("Master: super step is complete")

    except KeyboardInterrupt:
        LOG.info("Master: Terminating due to keyboard interrupt")

    LOG.info("Done processing")

    LOG.info("Printing results")

    for k, v in verticesDB.items():
        LOG.info(F"Node {k} pagerank: {v.value}")

#------------------------------------------------------------------------------#
#  PageRank Specific
#------------------------------------------------------------------------------#
#TODO make this more generic
#


def pageRankUpdateFunction(pv : PregelVertex, num_vertices : int):
    """
    Computes the page rank value.  After 50 supper steps is deactivates.

    :param pv: the PregelVertex as required by PregelUpdate function
    :param num_vertices: global data required by this particular function
    :return: None
    """

    LOG.debug(F"Update function for node {pv.id} with num_vertices {num_vertices}")
    if pv.superstep < 50:

        superstep_sum = sum([pm['msg'] for pm in pv.incoming_messages])

        # process incoming messages and compute value
        pv.value = 0.15 / num_vertices + 0.85 * superstep_sum

        outgoing_pagerank = pv.value / len(pv.out_vertices)

        # generate outgoing message.
        pv.outgoing_messages.extend([pv.createPregelMsg(pv.id, outgoing_vertex_id, outgoing_pagerank) for outgoing_vertex_id in pv.out_vertices])

    else:
        pv.active = False

def constructTopology(num_vertices, number_of_neighbours):

    # build the vertices

    verticesDB = {k:PregelVertex(k, 1.0 / num_vertices, []) for k in range(num_vertices)}

    # connect the vertices - randomly connect them
    vertices = list(verticesDB.values())

    # assign neighbours by randomly picking a sample of vertices
    # note a vertice can connect to itself.
    for vertex in vertices:
        vertex.out_vertices = [v.id for v in random.sample(vertices, number_of_neighbours)]

    return verticesDB

def pagerank_test(vertices, num_vertices):
    """

    :param vertices: are PregelVertecies
    :param num_vertices:
    :return:
    """
    """Computes the pagerank vector associated to vertices, using a
    standard matrix-theoretic approach to computing pagerank.  This is
    used as a basis for comparison."""

    I = mat(eye(num_vertices))
    G = zeros((num_vertices,num_vertices))
    for vertex in vertices:
        num_out_vertices = len(vertex.out_vertices)
        for out_vertex_id in vertex.out_vertices:
            G[out_vertex_id, vertex.id] = 1.0/num_out_vertices
    P = (1.0/num_vertices)*mat(ones((num_vertices,1)))
    return 0.15*((I-0.85*G).I)*P

def main( numberOfVertices, number_of_neighbours, numberOfWorkers):
    # Construct the Topology.  Normally this will be done somewhere else
    global verticesDB

    verticesDB = constructTopology(numberOfVertices, number_of_neighbours)

    """
    # display the topology
    for k,v in verticesDB.items():
        print(F"Vertex {k} value: {v.value} neighbours: {v.out_vertices}")
    """
    # passing in the function to execute initialling
    pregelMaster(verticesDB, functools.partial(pageRankUpdateFunction, num_vertices=numberOfVertices))

    page_rank_result_pregel = mat([pv.value for pv in verticesDB.values()]).transpose()

    # set the value to  1.0 / num_vertices
    for pv in verticesDB.values():
        pv.value = 1.0 / numberOfVertices

    page_rank_result = pagerank_test(verticesDB.values(),numberOfVertices)


    diff = page_rank_result_pregel - page_rank_result
    print("Difference between the two pagerank vectors:\n%s" % diff)
    print("The norm of the difference is: %s" % linalg.norm(diff))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d|%(name)-12s:%(lineno)4s|%(levelname)-8s| %(message)s',datefmt='%m-%d %H:%M:%S')
    numberOfWorkers = 6
    numberOfVertices = 1000
    number_of_neighbours = 8
    main(numberOfVertices, number_of_neighbours, numberOfWorkers)