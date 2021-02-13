class Grafo:
    def __init__(self, vertices):
        self.vertices = vertices
        self.grafo = [[0]*self.vertices for i in range(self.vertices)]

    def adiciona_aresta(self, u, v, peso): #considerando peso positivo
        self.grafo[u-1][v-1] = peso #grafo direcionado
        # self.grafo[v-1][u-1] = 1 #grafo não direcionado


    def mostra_matriz(self):
        print('Matriz Distancias:')
        for i in range(self.vertices):
            print(self.grafo[i])

#g = Grafo(4)
#g.adiciona_aresta(1,2,5)
#g.adiciona_aresta(3,4,9)
#g.adiciona_aresta(2,3,3)
#g.mostra_matriz()

