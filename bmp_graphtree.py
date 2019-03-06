from bmp_decisiontree import isnumeric
from graphviz import Digraph


def treetograph(node, outname, attribnames = None, dot = None, nodeid = 0, view = False):
        if attribnames == None:
                print('You need to provide a list of attribute names')
                return

        if dot == None:
                dot = Digraph()

        nodename = None
        if node['type'] == 'leaf':
                nodename = str(node['prediction']) + ' ' + str(node['classes'])
        else:
                question = node['question']
                attrib = attribnames[question[0]]
                if isnumeric(question[1]):
                        nodename = attrib + ' >= ' + str(question[1])
                else:
                        nodename = attrib + ' == ' + question[1]


        dot.node(str(nodeid), nodename)

        if nodeid != 0:
                parentid = int((nodeid - 1) / 2)
                edgelabel = 'True'
                if nodeid == (parentid*2) + 1:
                        edgelabel = 'False'

                dot.edge(str(parentid), str(nodeid), edgelabel)

        if node['type'] == 'leaf':
                return

        treetograph(node['falsechild'], outname, attribnames, dot, (nodeid * 2) + 1)
        treetograph(node['truechild'], outname, attribnames, dot, (nodeid * 2) + 2)

        if nodeid == 0:
                dot.render(outname, view=view)
