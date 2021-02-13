class Grafo:
    def __init__(self, vertices):
        self.vertices = vertices
        self.grafo = [[0]*self.vertices for i in range(self.vertices)]

    def adiciona_aresta(self, u, v):
        self.grafo[u-1][v-1] = 1 #grafo direcionado
        # self.grafo[v-1][u-1] = 1 #grafo não direcionado


    def mostra_matriz(self):
        print('Matriz Adjacencias:')
        for i in range(self.vertices):
            print(self.grafo[i])


#Função para contar as linhas do arquivo que serão a quantidade de vértices
def fileLineCount():
    arq = open('coordenadas.txt', 'r')
    return sum(1 for line in arq)

#Iniciando o grafo com a quantidade de vértices contados no arquivo
g = Grafo(fileLineCount())

#Abrindo o arquivo
file = open('coordenadas.txt', 'r')

#Adicionando as arestas
for line in file.readlines():
    u, v = line.split()
    g.adiciona_aresta(int(u), int(v))

g.mostra_matriz()

file.close()



