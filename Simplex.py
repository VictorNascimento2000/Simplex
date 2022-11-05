import numpy as np

dim = input()

n, m = [int(s) for s in dim.split() if s.isdigit()]

c_original = np.fromstring(input(), dtype=float, sep=' ')
mat = np.zeros((1,c_original.shape[0]+1))

save = []
init = 0
while n > 0:
   Array = np.fromstring(input(), dtype=float, sep=' ')
   Array.reshape((-1,1))
   mat = np.concatenate((mat, Array.T.reshape((1,-1))), axis=0)
   n -= 1
mat = np.delete(mat,0,0)
A_original= mat[:,:m]
b_ = mat[:,m]
b = b_.reshape((b_.shape[0],1))

def FPI(A, b, c):
    identidade = np.identity(A.shape[0])
    A_fpi = np.concatenate((A, identidade), axis=1)
    zeros = np.zeros(A_original.shape[0])
    c_fpi = np.concatenate((c, zeros), axis=0)
    c_fpi = np.atleast_2d(c_fpi)
    c_A_fpi = np.concatenate((c_fpi, A_fpi), axis=0)
    zero = np.zeros(1)
    zero = np.atleast_2d(zero)
    b_new = np.concatenate((zero,b), axis=0)
    c_A_b_fpi = np.concatenate((c_A_fpi,b_new), axis=1)
    return c_A_b_fpi

#checando se há elementos negativos no vetor b
def IsNegative(A):
    flag = 0
    for i in range(A.shape[0]):
        if A[i,-1] < 0:
            flag = 1
            A[i,:] = (-1)*A[i,:]
    return A

def Pivoteamento(matriz, linha, coluna):
    matriz[linha,:] = (1/matriz[linha,coluna])*matriz[linha,:]
    for x in range(0,linha):
        novo_zero = matriz[x,coluna]
        matriz[x,:] = matriz[x,:] - novo_zero*matriz[linha,:]
    for x in range(linha+1,matriz.shape[0]):
        novo_zero = matriz[x,coluna]
        matriz[x,:] = matriz[x,:] - novo_zero*matriz[linha,:]
    return matriz

def FormaCanonica(A):
    base_inicial = np.zeros(A.shape[0])
    b = A[:,-1]
    b = np.atleast_2d(b)
    A_del = np.delete(A, -1, 1)
    identidade = np.identity(A.shape[0]-1)
    zeros = np.zeros(A.shape[0]-1)
    zeros = np.atleast_2d(zeros)
    A_aux = np.concatenate((zeros, identidade), axis=0)
    A_new = np.concatenate((A_del, A_aux), axis=1)
    A_new = np.concatenate((A_new, b.T), axis=1)
    for i in range(1, A.shape[0]):
        A_new = Pivoteamento(A_new,i,A_new.shape[1]-A_new.shape[0]+i-1)
        base_inicial[i]=A_new.shape[1]-A_new.shape[0]+i-1
    return A_new, base_inicial

def PL_Auxiliar(A, b):
    A[0,:] = 0
    for i in range(1, A.shape[0]):
        A[0,A.shape[1]-A.shape[0]+i-1] = 1
    return A

def adiciona_VERO(matriz):
    identidade = np.identity(matriz.shape[0]-1)
    zeros = np.zeros(A_original.shape[0])
    zeros = np.atleast_2d(zeros)
    VERO = np.concatenate((zeros,identidade), axis=0)
    matriz = np.concatenate((VERO, matriz), axis=1)
    return matriz

def escolhe_coluna(vetor):
        index = 0
        for i in np.nditer(vetor):
            if i < 0:
                return index
            else:
                index += 1

def escolhe_linha(coluna, b):
        aux = np.inf
        for i in range(coluna.shape[0]):
            if coluna[i] > 0:
                razao = b[i]/coluna[i]
                if razao < aux:
                    aux = razao
                    index = i
        return index

def Simplex(matriz, base):

    linha = 1
    base = base.astype(int)
    for coluna in base:
        if coluna != 0:
            matriz = Pivoteamento(matriz, linha, coluna)
            linha += 1
    
    while np.all(matriz[0,matriz.shape[0]-1:matriz.shape[1]-1] >= 0) == False:
        coluna = escolhe_coluna(matriz[0,matriz.shape[0]-1:matriz.shape[1]-1])
        b = matriz[1:,matriz.shape[1]-1]
        linha = escolhe_linha(matriz[1:,coluna+matriz.shape[0]-1], b)
        base[linha+1] = coluna+matriz.shape[0]-1
        matriz = Pivoteamento(matriz, linha+1, coluna+matriz.shape[0]-1)
        matriz[np.isclose(matriz, 0)] = 0
    return matriz, base

def vetor_sol_otimo(matriz, base, num_linhas):
    solucao = np.zeros(num_linhas)
    for j in base[1:]:
        if j < matriz.shape[1]-matriz.shape[0]:
            count = 0
            for i in range(1,matriz.shape[0]):
                if matriz[i][j] == 1:
                    solucao[j-matriz.shape[0]+1] = matriz[i,-1]
                else:
                    count += 1
    return solucao

#verificando se as variáveis da base são variáveis artificiais da PL Auxiliar. Se não forem então excluímos 
# as variáveis artificiais do tableau. Se houver uma variáveis artificiais na base então vemos a linha correspondente ao pivô da coluna
# dessa base. Então nessa linha procuramos o primeiro elemento diferente de zero após da matriz VERO e pivoteamos esse elemento.   
def variaveis_auxiliares_base(matriz, base):
    if matriz[0,-1] < 0:
        return matriz, base
    else:
        if np.all(base <= matriz.shape[1]-matriz.shape[0]) == True:
            count = matriz.shape[0]-1
            ultima_coluna = matriz.shape[1]-1
            i = 1
            while count > 0:
                matriz = np.delete(matriz, ultima_coluna-i, 1)
                count -= 1
                i += 1
            return matriz, base
        else:
            while np.all(base <= matriz.shape[1]-matriz.shape[0]) == False:
                index = 0
                for i in base:
                    if i > matriz.shape[1]-matriz.shape[0]-1:
                        for j in range(matriz.shape[0]-1, matriz.shape[1]):
                            if matriz[index, j] != 0:
                                matriz = Pivoteamento(matriz, index, j)
                                base[index] = j
                                break
                    else:
                        index += 1
            count = matriz.shape[0]-1
            ultima_coluna = matriz.shape[1]-1
            i = 1
            while count > 0:
                matriz = np.delete(matriz, ultima_coluna-i, 1)
                count -= 1
                i += 1
            return matriz, base

def classificacao(A, base, c_original):
    if A[0,-1] < 0:
        print('inviavel')
        print(A[0, 0:(A.shape[0]-1)])
    else:
        if A[0, -1] == 0:
            c_original = np.atleast_2d(c_original)
            c_original = (-1)*c_original
            A[0,A.shape[0]-1:c_original.shape[1]+A.shape[0]-1] = c_original
            resultado_simplex, base = Simplex(A, base)
            print('otimo')
            print(resultado_simplex[0, -1])
            print(vetor_sol_otimo(resultado_simplex, base, c_original.shape[1]))
            print(resultado_simplex[0, 0:resultado_simplex.shape[0]-1])
            
c_A_fpi = FPI(A_original, b, c_original)
c_A_fpi_vero = adiciona_VERO(c_A_fpi)
c_A_fpi_vero = IsNegative(c_A_fpi_vero)
c_A_fpi_vero_ = np.copy(c_A_fpi_vero)
c_A_can, base_inicial = FormaCanonica(c_A_fpi_vero)
tableau_Auxiliar = PL_Auxiliar(c_A_can, b)
resultado_auxiliar, base = Simplex(tableau_Auxiliar, base_inicial)
resultado_auxiliar, base = variaveis_auxiliares_base(resultado_auxiliar, base)
#Simples fase 2
classificacao(resultado_auxiliar, base, c_original)
