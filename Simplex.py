import numpy as np
import scipy.linalg

# Lendo a entrada de dados:
# A PL no formato:
# max    c.T @ x + z
# suj a: Ax <= b
#       x>=0
# é uma tupla: PL = (c, A, b, z, cert, oper)
# cert e oper são certificado e matriz de operações do lado esquerdo do tableau.
dimensao = np.fromstring(input(), dtype=int, sep=' ')
n = dimensao[0]
m = dimensao[1]
c = np.fromstring(input(), dtype=float, sep=' ')
z = np.array([0])
restricao = np.zeros((n, m+1))
for i in range(0, n):
    restricao[i] = np.fromstring(input(), dtype=float, sep=' ')
cert = np.zeros(n)
oper = np.identity(n)
A = restricao[:n, :m]
b = restricao[:n, m:m+1]
PL = [c, A, b, z, cert, oper]

# Coloca a PL em FPI:


def fpi(PL):
    #PL = [c, A, b, z, cert, oper]
    c = PL[0]
    A = PL[1]
    b = PL[2]
    I = np.identity(n)
    A = np.concatenate((A, I), axis=1)
    c = np.concatenate((c, np.zeros((n))))
    for i in range(0, n):
        if b[i] < 0:
            b[i] = b[i] * -1
            A[i] = A[i] * -1
            oper[i] = oper[i] * -1
    PL[0] = c
    PL[1] = A
    PL[5] = oper
    return PL

# Gera a PL auxiliar


def auxiliar(PL):
    #PL = [c, A, b, z, cert, oper]
    c = PL[0]
    A = PL[1]
    b = PL[2]
    z = PL[3]
    I = np.identity(n)
    c = np.concatenate((np.zeros(np.size(c)), np.full(n, -1)))
    A = np.concatenate((A, I), axis=1)
    aux_PL = [c, A, b, z, PL[4], PL[5]]
    return aux_PL

# Coloca a PL em forma canônica, de acordo com uma base viável
# Preferi utilizar as formulas de algebra linear ao invés de pivoteamento
# Gosto da teoria por trás, e é mais bonito


def forma_canonica(PL, base):
    #PL = [c, A, b, z, cert, oper]
    n = np.size(PL[2])
    c = PL[0]
    A = PL[1]
    b = PL[2]
    z = PL[3]
    cert = PL[4]
    oper = PL[5]
    A_base = np.zeros((n, n))
    c_base = np.zeros(n)
# Gero meu vetor c_base e matriz A_base
    for i in range(0, n):
        k = base[i]
        c_base[i] = c[k]
        A_base[:, i] = A[:, k]
# Faço as operações matriciais que transformam o tableau na forma canônica:
# (C1) Fazendo as colunas da base se tornarem a Identidade:
# multiplicando embaixo pela inversa de A_base:
    PL[5] = (np.linalg.inv(A_base) @ oper)  # Operações
    PL[1] = (np.linalg.inv(A_base) @ A)  # Restrições
    PL[2] = (np.linalg.inv(A_base) @ b)  # Vetor b
# (C2) Fazendo os coeficientes da base na função objetivo zerarem:
# multiplicando embaixo por c_base.T e somando na primeira linha:
    PL[4] = PL[4] + c_base @ PL[5]  # Certificado
    PL[0] = PL[0] - c_base @ PL[1]  # Vetor c
    PL[3] = PL[3] + c_base @ PL[2]  # Valor objetivo


def encontra_coluna(c):
    #PL = [c, A, b, z, cert, oper]
    # Seleciona o elemento positivo de c.
    # O indice dele é a coluna de A que entra na base.
    # Se não existir tal elemento, atingimos um ótimo.
    m = np.shape(c)[0]
    for i in range(0, m):
        if c[i] > 0:
            return i
    return -1


def solucao(b, base, m):
    #PL = [c, A, b, z, cert, oper]
    # Construo o vetor x que me dá o valor ótimo para a PL
    # Os x_i correspondentes à base ótima recebem os valores de b
    n = np.size(PL[2])
    x = np.zeros(m)
    for i in range(0, n):
        k = base[i]
        x[k] = b[i]
    return x


def encontra_pivo(b, Ak):
    #PL = [c, A, b, z, cert, oper]
    # Faço o teste da razão entre b_i/A_ik
    # o indice corresponde ao indice do elemento da base que irá deixá-la
    # Se A_k for todo negativo, então a PL é ilimitada.
    n = np.size(PL[2])
    t = -1
    test = np.all(Ak <= 0)
    x = np.array([])
    index = np.array([])
    if test:
        return t
    for i in range(0, n):
        if Ak[i] > 0:
            x = np.append(x, b[i] / Ak[i])
            index = np.append(index, i)
    r = int(index[np.argmin(x)])
    return r


def simplex(PL, base, aux):
    #PL = [c, A, b, z, cert, oper]
    while (True):
        # 1 - Coloco ela em base canônica
        forma_canonica(PL, base)
        # 2 - Condição de parada.
        # Checo se existe (c_k > 0).
        # Se existir, k entra na base
        # se não z é o V.O e P é ótima
        k = encontra_coluna(PL[0])
        # checando redundância:
        if k == -1:
            delete = np.array([], dtype=int)
            if PL[3] == 0:
                if (aux == True):
                    dim_original = np.size(PL[0]) - np.size(PL[2])
                    tamanho_base = np.size(base)
                    for i in range(0, tamanho_base):
                        if base[i] >= dim_original:
                            linha = PL[1][i, :dim_original]
                            if np.all(np.allclose(linha, np.zeros(dim_original))):
                                delete = np.append(delete, i)
                                break
                            for j in range(0, dim_original):
                                if PL[1][i, j] != 0:
                                    base = np.delete(base, i)
                                    base = np.append(base, j)
                                    base = np.sort(base)
                                    forma_canonica(PL, base)
                                    break
            # Envia solucao e resultados
            vetor_x = solucao(PL[2], base, np.size(PL[0]))
            resultado = ("otima", vetor_x[:m], PL[3], PL[4], base, delete)
            return resultado
        # 3 - Condição de parada.
        # Checo se A[:,k]<=0. Se for, a PL é ilimitada.
        # Se não, faço o teste t = min{b_i/A_ik}.
        # O i onde o min ocorre é o indice do elemento da base que irá sair.
        r = encontra_pivo(PL[2], PL[1][:, k])
        if r == -1:
            tam = np.size(PL[0]) - np.size(PL[2])
            print("ilimitada")
            vetor_x = solucao(PL[2], base, np.size(PL[0]))
            d = np.zeros(np.size(PL[0]))
            d[k] = 1
            Ak = PL[1][:, k]
            for i in range(0, np.size(PL[2])):
                j = base[i]
                d[j] = -1 * Ak[i]
            vetor_x = vetor_x[:tam]
            d = d[:tam]
            for i in range(0, np.size(vetor_x)):
                print(vetor_x[i], end="  ")
            print()
            for i in range(0, np.size(d)):
                print(d[i], end='  ')
            print()
            return -1
        # Atualizo a base viável
        base = np.delete(base, int(r))
        base = np.append(base, k)
        base = np.sort(base)


def Solving(PL):
    #PL = [c, A, b, z, cert, oper]
    # transforma FPI
    PL = fpi(PL)
    n = np.size(PL[0])
    m = np.size(PL[2])
    # FASE 1:
    # Resolvendo a PL Auxiliar
    aux_PL = auxiliar(PL)
    aux_base = np.arange(n, n+m)
    # Verifica se a PL original é inviavel, de acordo com o otimo da auxliar
    # Resultado é uma tupla r=(estado, valor objetivo, certificado, solucao x)
    aux_resultado = simplex(aux_PL, aux_base, True)
    aux_otimo = aux_resultado[2]
    if aux_otimo < 0:
        certificado = aux_resultado[3]
        resultado_original = ("inviavel", certificado)
        print(resultado_original[0])
        for i in range(0, np.size(resultado_original[1])):
            print(resultado_original[1][i], end='  ')
        print()
        return
    # Deduzindo a base viável para a Pl original:
    base = aux_resultado[4]
    # Verificando se precisa deleter colunas redundantes:
    deletar = aux_resultado[5]
    if np.size(deletar) != 0:
        for i in range(0, np.size(deletar)):
            base = np.delete(base, deletar[i])
            PL[1] = np.delete(PL[1], deletar[i], axis=0)
            PL[2] = np.delete(PL[2], deletar[i], axis=0)
            PL[5] = np.delete(PL[5], deletar[i], axis=0)
    # Resolve a PL original:
    solucao = simplex(PL, base, False)
    if solucao == -1:
        return
    # Imprime resultados:
    print(solucao[0])
    print(solucao[2][0])
    for i in range(0, np.size(solucao[1])):
        print(solucao[1][i], end=' ')
    print()
    for i in range(0, np.size(solucao[3])):
        print(solucao[3][i], end=' ')
    print()


Solving(PL)
