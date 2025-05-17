from rdflib import Graph as RdfGraph
from rdflib import Literal
from os.path import join
import numpy as np
from urllib.parse import quote
import pickle
import glob


import os
import multiprocessing as mp
from itertools import chain
from tqdm import tqdm

global_nodeId = None
global_relId = None

from setuptools.sandbox import save_argv

def all_strings(seq):
    return all(isinstance(x, str) for x in seq)

prefix_map = {
    "wd:": "http://www.wikidata.org/entity/",
    "wdt:": "http://www.wikidata.org/prop/direct/",
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
}

def process_batch_result(batch_result, batch_id):
    save_path = f"adj_matrix_results_batches/batch_{batch_id}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(batch_result, f)

def process_batch(batch):
    global global_nodeId, global_relId
    return process_triples_batch(batch, global_nodeId, global_relId)

def init_process(nodeId, relId):
    global global_nodeId, global_relId
    global_nodeId = nodeId
    global_relId = relId

def process_triples_batch(batch, nodeId, relId):
    facts = []
    for triple in batch:
        if len(triple) != 3:
            print(f"Skipping invalid triple format: {triple}")
            continue
        sub, pred, obj = triple
        try:
            facts.append([nodeId[str(sub)], nodeId[str(obj)], relId[str(pred)]])
        except KeyError:
            continue  # Skip any missing IDs
    return facts
class GraphTransformer:
    """
    Transform a graph in turtle representation into adjacency matrix
    requred to build Graph.
    """
#    def expand(term):
#        for p, uri in prefix_map.items():
#            if term.startswith(p):
#                return f"<{quote(uri + term[len(p):], safe=':/#')}>"
#        return term
    def save_triples_in_chunks(self, triples, save_dir, chunk_size=1_000_000):
        """
        Save a large list of triples into multiple smaller pickle files.

        Args:
            triples (list): Your list of triples.
            save_dir (str): Directory where chunks will be saved.
            chunk_size (int): Number of triples per chunk file.
        """
        os.makedirs(save_dir, exist_ok=True)

        total = len(triples)
        print(f"Total triples: {total}")

        for i in range(0, total, chunk_size):
            chunk = triples[i:i + chunk_size]
            chunk_filename = os.path.join(save_dir, f"triples_chunk_{i//chunk_size:04d}.pkl")
            with open(chunk_filename, 'wb') as f:
                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            print(f"Saved {chunk_filename} with {len(chunk)} triples.")

        print("All chunks saved successfully.")

    def stream_batches_from_chunks(slef, save_dir, batch_size):
        """
        Stream triples batch-by-batch from multiple chunked pickle files.

        Args:
            save_dir (str): Directory where chunked files are saved.
            batch_size (int): Number of triples per batch to yield.

        Yields:
            list: A batch of triples.
        """
        batch = []
        chunk_files = sorted(f for f in os.listdir(save_dir) if f.endswith('.pkl'))
        print(f"Found {len(chunk_files)} chunks.")

        for filename in chunk_files:
            path = os.path.join(save_dir, filename)
            print(f"Loading {filename}...")
            with open(path, 'rb') as f:
                try:
                    while True:
                        triples = pickle.load(f)
                        for triple in triples:  # Iterate through each triple
                            batch.append(triple)
                            if len(batch) == batch_size:
                                yield batch
                                batch = []
                        # batch.append(triple)
                        # if len(batch) == batch_size:
                        # yield batch
                            # batch = []
                except EOFError:
                    print(f"EOFError in {path}")
                    pass  # Finished current file

        if batch:
            yield batch  # Yield any remaining triples after last file
    def sanitize_predicate(self,line):
        # Fix invalid predicate like wdt:22-rdf-syntax-ns#type → rdf:type
        if "wdt:22-rdf-syntax-ns#type" in line:
            return line.replace("wdt:22-rdf-syntax-ns#type", "rdf:type")
        elif "wdt:rdf-schema#subClassOf" in line:
            return  line.replace("wdt:rdf-schema#subClassOf", "rdfs:subClassOf")
        elif ":P-" in line:
            return line.replace("wdt:P-", "dice:P-")
        return line



    def __init__(self, idPath):
        self.nodeId = dict()
        self.relId = dict()
        self.nodeIdCount = 0
        self.relIdCount = 0
        self.idPath = idPath

    def generateAdjacency(self, graphPath):
        """
        Generate an adjacency for the turtle graph at graphPath.
        First, read the graph line by line and assign a ID to each resource.
        Second, save the IDs in a text file.
        Finally, generate a list of facts in the form [subjectID, objectID, predicateID]

        Return that list as a numpy array.
        """
        nodes_file = join(self.idPath, "data/kg/nodes.txt")
        relations_file = join(self.idPath, "data/kg/relations.txt")
        # adj_file = join(self.idPath, "data/kg/adjacency.npy")
        triples_file_path = join(self.idPath, "triples_chunks/all_triples.pkl")
        triple_ids = join(self.idPath, "triples_chunks")
        save_dir = triple_ids

        adj_path = join(self.idPath, "data/kg/adjacency.npy")


        if os.path.exists(nodes_file) and os.path.exists(relations_file):
            self._loadIDs()
        else:
            graphIterator = self._getGraphIterator(graphPath)
            # t_count = sum(1 for _ in self._getGraphIterator(graphPath))

            count = 0

            for rdfGraph in graphIterator:
                self._generateIndices(rdfGraph)
                count += len(rdfGraph)  # Count triples instead of iterations
                if count % 1000000 == 0:
                    print("Generated IDs for {} facts".format(count))

            print("Generated all IDs")
            self._saveIDs()
            print("Saved IDs")

        ### Generate adjacency matrix
        if os.path.exists(adj_path):
            print("Loading existing adjacency matrix...")
            return np.load(adj_path)

        graphIterator = self._getGraphIterator(graphPath)
        # all_triples = list(chain.from_iterable(graphIterator))  # Flatten the graphs to a single list of triples
        if not os.path.exists(save_dir):
            print("generating triples...")
            all_triples = list(chain.from_iterable(graphIterator))  # Flatten the graphs to a single list of triples
            self.save_triples_in_chunks(all_triples, save_dir, chunk_size=1_000_000)
            # Save to file
            # with open("all_triples.pkl", "wb") as f:
            #     pickle.dump(all_triples, f)
        else:
            print("Loading triples from saved file directly..." + triple_ids)
            # with open(triples_file_path, "rb") as f:
            #     all_triples = pickle.load(f)

        # num_workers = min(mp.cpu_count(), 100)
        batch_size = 50000
        mp.set_start_method('forkserver', force=True)
        batch_gen = self.stream_batches_from_chunks(triple_ids, batch_size=batch_size)
        os.makedirs("adj_matrix_results_batches", exist_ok=True)
        batch_id = 0
        with mp.Pool(processes=25, initializer=init_process, initargs=(self.nodeId, self.relId)) as pool:
            for batch_result in tqdm(
                    pool.imap_unordered(process_batch, batch_gen, chunksize=1),
                    total=None,
                    desc="Generating Adjacency Matrix"
            ):
                process_batch_result(batch_result,batch_id)  # <--- handle it immediately (save to file, build matrix, etc)
                batch_id += 1

        # Flatten list of results
        batch_files = sorted(glob.glob('adj_matrix_results_batches/batch_*.pkl'))
        all_results = []
        for file in batch_files:
            with open(file, 'rb') as f:
                batch_result = pickle.load(f)
                all_results.extend(batch_result)
        # facts = list(chain.from_iterable(all_results))
        adj = np.array(all_results)

        print("Created adjacency matrix")
        # exit(1)
        # Step 3: Save it
        os.path.exists(adj_path) or open(adj_path, 'a').close()

        np.save(adj_path, adj)
        print(f"Saved adjacency matrix to {adj_path}")

        return adj
        # batch_size = 5000000  # Tune this depending on your memory
        # batches = [all_triples[i:i + batch_size] for i in range(0, len(all_triples), batch_size)]

        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     results = list(tqdm(
        #         pool.imap_unordered(
        #             lambda b: self.process_triples_batch(b, self.nodeId, self.relId), batches
        #         ),
        #         total=len(batches),
        #         desc="Generating Adjacency Matrix"
        #     ))
        #
        # # Flatten list of results
        # facts = list(chain.from_iterable(results))
        # adj = np.array(facts)
        #
        # print("Created adjacency matrix")
        # # Step 3: Save it
        # np.save(adj_path, adj)
        # print(f"Saved adjacency matrix to {adj_path}")
        #
        # return adj
    # def generateAdjacency(self, graphPath):
    #     """
    #     Generate an adjacency for the turtle graph at graphPath.
    #     First, read the graph line by line and assign a ID to each resource.
    #     Second, save the IDs in a text file.
    #     Finally, generate a list of facts in the form [subjectID, objectID, predicateID]
    #
    #     Return that list as a numpy array.
    #     """
    #     #
    #     # ### Generate IDs
    #     # count = 0
    #     # graphIterator = self._getGraphIterator(graphPath)
    #     # for rdfGraph in graphIterator():
    #     #     self._generateIndices(rdfGraph)
    #     #     count += 1
    #     #     if count % 100000 == 0:
    #     #         print("Generated IDs for {} facts".format(count))
    #     graphIterator = self._getGraphIterator(graphPath)
    #     count = 0
    #
    #     for rdfGraph in graphIterator:
    #         self._generateIndices(rdfGraph)
    #         count += len(rdfGraph)  # Count triples instead of iterations
    #         if count % 1000000 == 0:
    #             print(f"Generated IDs for {count} facts")
    #
    #     print("Generated all IDs")
    #     self._saveIDs()
    #     print("Saved IDs")
    #
    #     ### Generate adjacency matrix
    #     facts = []
    #     count = 0
    #     graphIterator = self._getGraphIterator(graphPath)
    #     for rdfGraph in graphIterator:
    #         for sub, pred, obj in rdfGraph:
    #             if type(obj) != Literal:
    #                 facts.append([self.nodeId[sub], self.nodeId[obj], self.relId[pred]])
    #                 count += 1
    #                 if count % 10000 == 0:
    #                     print("Generated array for {} facts".format(count))
    #
    #     adj = np.asarray(facts)
    #     print("Created adjacency matrix")
    #     return adj
    
    def getShape(self):
        nodes = len(self.nodeId.keys())
        relationships = len(self.relId.keys())
        return (nodes, nodes, relationships)

    def _generateIndices(self, rdfGraph):
        nodeId = self.nodeId  # Local reference to avoid repeated attribute lookup
        relId = self.relId
        nextNodeId = self._nextNodeId
        nextRelId = self._nextRelationshipId

        for sub, pred, obj in rdfGraph:
            # Assign IDs efficiently using setdefault()
            nodeId.setdefault(sub, nextNodeId())

            if not isinstance(obj, Literal):
                nodeId.setdefault(obj, nextNodeId())

            relId.setdefault(pred, nextRelId())

    def _generateIndices_backup(self, rdfGraph):
        for sub, pred, obj in rdfGraph:
            # Set subject id
            try:
                self.nodeId[sub]
            except KeyError:
                self.nodeId[sub] = self._nextNodeId()

            # Set object id
            try:
                if type(obj) != Literal:
                    self.nodeId[obj]
            except KeyError:
                self.nodeId[obj] = self._nextNodeId()

            # Set predicate id
            try:
                self.relId[pred]
            except KeyError:
                self.relId[pred] = self._nextRelationshipId()

    def _nextNodeId(self):
        nextId = self.nodeIdCount
        self.nodeIdCount += 1
        return nextId

    def _nextRelationshipId(self):
        nextId = self.relIdCount
        self.relIdCount += 1
        return nextId

#    def _getGraphIterator(self, graphPath):
#        graphFile = open(graphPath, 'r')
#        def graphIterator():
#            while line := graphFile.readline():
#                rdfGraph = RdfGraph()
#                rdfGraph.parse(data=line, format='ttl')
#                yield rdfGraph
#            graphFile.close()

#        return graphIterator

    def _getGraphIterator(self, graphPath, batch_size=100000000):
        def expand(term):
            return f"<{quote(term, safe=':/#')}>"

        with open(graphPath, 'r') as graphFile:
            buffer = []
            for line in graphFile:
                line = self.sanitize_predicate(line)
                parts = line.strip().split()
                if '@' not in parts[0] and len(parts) == 4 and parts[3] == ".":
                    subj = expand(parts[0])
                    pred = expand(parts[1])
                    obj = expand(parts[2])
                    triple = f"{subj} {pred} {obj} .\n"
                    buffer.append(triple)

                    if len(buffer) >= batch_size:
                        print(len(buffer))
                        rdfGraph = RdfGraph()
                        try:
                            rdfGraph.parse(
                                data=(
                                        "@prefix wd: <http://www.wikidata.org/entity/> .\n"
                                        "@prefix wdt: <http://www.wikidata.org/prop/direct/> .\n"
                                        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
                                        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                                        "@prefix dice: <http://dice-research.org/> .\n"
                                        + ''.join(buffer)
                                ),
                                format='ttl'
                            )
                            yield rdfGraph
                        except Exception as e:
                            print(f"Error parsing batch\nException: {e}")
                            exit(1)
                        buffer = []

            # Remaining triples
            if buffer:
                rdfGraph = RdfGraph()
                try:
                    rdfGraph.parse(
                        data=(
                                "@prefix wd: <http://www.wikidata.org/entity/> .\n"
                                "@prefix wdt: <http://www.wikidata.org/prop/direct/> .\n"
                                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
                                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                                "@prefix dice: <http://dice-research.org/> .\n"
                                + ''.join(buffer)
                        ),
                        format='ttl'
                    )
                    yield rdfGraph
                except Exception as e:
                    print(f"Error parsing final batch\nException: {e}")
                    exit(1)

#     def _getGraphIterator(self, graphPath):
#         def expand(term):
#             return f"<{quote(term, safe=':/#')}>"
#         graphFile = open(graphPath, 'r')
#         def graphIterator():
#             # Initialize an RDF graph
#             #rdfGraph = RdfGraph()
#             # Bind prefixes globally for the graph
#             #rdfGraph.bind("wd", "http://www.wikidata.org/entity/")
#             #rdfGraph.bind("wdt", "http://www.wikidata.org/prop/direct/")
#             for line in graphFile:
#                 line = self.sanitize_predicate(line)
#                 parts = line.strip().split()
#                 if '@' not in parts[0] and len(parts) == 4 and parts[3] == ".":
#                     subj = expand(parts[0])
#                     pred = expand(parts[1])
#                     obj = expand(parts[2])
#                     fixed_line = f"{subj} {pred} {obj} .\n"
#                     # Add prefix declarations to each line
#                      # line = self.sanitize_predicate(line)
#                     triple = (
#                         "@prefix wd: <http://www.wikidata.org/entity/> .\n"
#                         "@prefix wdt: <http://www.wikidata.org/prop/direct/> .\n"
#                         "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
#                         "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
#                         "@prefix dice: <http://dice-research.org/> .\n"
#                         + fixed_line
#                     )
# #                    print(fixed_line)
#                     rdfGraph = RdfGraph()
#                     #rdfGraph.bind("wd", "http://www.wikidata.org/entity/")
#                     #rdfGraph.bind("wdt", "http://www.wikidata.org/prop/direct/")
#                     try:
#                       rdfGraph.parse(data=triple, format='turtle')
#                     except Exception as e:
#                       print(f"Error parsing line: {line}\nException: {e}")
#                       exit(1)
#                     yield rdfGraph
#             graphFile.close()
#
#         return graphIterator()


    def _loadIDs(self):
        """
        Load node and relation IDs from saved files.
        """
        self.nodeId = {}
        self.relId = {}

        # Load shape (optional)
        shape_file = join(self.idPath, "data/kg/shape.txt")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                self.shape = tuple(map(int, f.readline().strip().strip("()").split(',')))
        else:
            self.shape = None  # or raise an error if shape is required

        # Load node IDs
        nodes_file = join(self.idPath, "data/kg/nodes.txt")
        max_node = 0
        with open(nodes_file, 'r') as f:
            for line in f:
                node_id, resource = line.strip().split(" ", 1)
                self.nodeId[resource] = int(node_id)
                if max_node < int(node_id):
                    max_node = int(node_id)
        self.nodeIdCount = max_node + 1


        # Load relation IDs
        relations_file = join(self.idPath, "data/kg/relations.txt")
        max_rel = 0
        with open(relations_file, 'r') as f:
            for line in f:
                rel_id, relation = line.strip().split(" ", 1)
                self.relId[relation] = int(rel_id)
                if max_rel < int(rel_id):
                    max_rel = int(rel_id)
        self.relIdCount = max_rel + 1

        print("Loaded {} node IDs and {} relation IDs.".format(len(self.nodeId), len(self.relId)))

    # def _saveIDs(self):
    #     with open(join(self.idPath, "data/kg/shape.txt"), 'w') as shapeFile:
    #         shapeFile.write(str(self.getShape()) + '\n')
    #
    #     with open(join(self.idPath, "data/kg/nodes.txt"), 'w') as nodesFile:
    #         for resource in self.nodeId.keys():
    #             nodesFile.write("{} {}\n".format(self.nodeId[resource], resource))
    #
    #     with open(join(self.idPath, "data/kg/relations.txt"), 'w') as relationsFile:
    #         for relation in self.relId.keys():
    #             relationsFile.write("{} {}\n".format(self.relId[relation], relation))
    def _saveIDs(self):
        with open(join(self.idPath, "data/kg/shape.txt"), 'w') as shapeFile:
            shapeFile.write(str(self.getShape()) + '\n')

        # Reassign sequential IDs to nodes
        sorted_resources = sorted(self.nodeId.keys(), key=lambda r: self.nodeId[r])
        new_nodeId = {resource: idx for idx, resource in enumerate(sorted_resources)}

        with open(join(self.idPath, "data/kg/nodes.txt"), 'w') as nodesFile:
            for resource, idx in new_nodeId.items():
                nodesFile.write(f"{idx} {resource}\n")

        # Reassign sequential IDs to relations
        sorted_relations = sorted(self.relId.keys(), key=lambda r: self.relId[r])
        new_relId = {relation: idx for idx, relation in enumerate(sorted_relations)}

        with open(join(self.idPath, "data/kg/relations.txt"), 'w') as relationsFile:
            for relation, idx in new_relId.items():
                relationsFile.write(f"{idx} {relation}\n")

        # Optional: Update self.nodeId and self.relId in memory too
        self.nodeId = new_nodeId
        self.relId = new_relId