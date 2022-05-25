import numpy as np

def assignment(c: np.matrix):
    # assert isinstance(C, np.matrix), "use np.matrix"
    n = len(c)
    # Zij = Cij - ui - vj
    z = np.matrix(c)
    u = np.zeros(n)
    v = np.zeros(n)
    # dual cost
    dcost = 0
    # assignemt
    x = -np.ones(n, dtype=int)
    # first phase
    for i in range(n):
        minv = z[i, :].min()
        if minv > 0:
            u[i] += minv
            z[i, :] -= minv
            dcost += minv
    for j in range(n):
        minv = z[:, j].min()
        if minv > 0:
            v[j] += minv
            z[:, j] -= minv
            dcost += minv
    # print("dual cost", dcost)

    flagR = np.zeros(n, dtype=bool)
    flagC = np.zeros(n, dtype=bool)
    while True:
        # assigment attempt
        x.fill(-1)
        flagR.fill(False)
        flagC.fill(False)
        zAsArray = np.array(z)
        zeroC = n - np.count_nonzero(zAsArray, axis=0)
        zeroR = n - np.count_nonzero(zAsArray, axis=1)
        xcount = 0
        zeros = np.column_stack(np.where(z == 0))
        while len(zeros) > 0:

            i, j = min(zeros, key=lambda a: min(zeroR[a[0]], zeroC[a[1]]))

            if zeroR[i] >= zeroC[j]:
                flagR[i] = True
            else:
                flagC[j] = True
            # flagR[i] = True
            # flagC[j] = True
            for k, l in zeros:
                if k == i or l == j:
                    zeroR[k] -= 1
                    zeroC[l] -= 1
            x[i] = j

            xcount += 1
            zeros = [z for z in zeros if not flagR[z[0]] and not flagC[z[1]]]
            # zeros = [z for z in zeros if z[0] != i and z[1] != j]

        # print("assigned ", xcount)
        if xcount < n:
            minv = np.inf
            for i in range(n):
                if not flagR[i]:
                    for j in range(n):
                        if not flagC[j] and minv > z[i, j]:
                            minv = z[i, j]
                            if minv == 0:
                                print("ops")
                # print(C)

            assert minv > 0, 'minv deve ser > zero'

            for k in range(n):
                if not flagR[k]:
                    v[k] += minv
                    z[k, :] -= minv
                    dcost += minv
                if flagC[k]:
                    u[k] -= minv
                    z[:, k] += minv
                    dcost -= minv
            # print('dcost', dcost)

        else:
            break

    return dcost, x, v, u

# import numpy as np

# C = np.matrix([[4, 2, 5, 7], [8, 3, 10, 8], [12, 5, 4, 5], [6, 3, 7, 14]], dtype=float)
# C = np.matrix([[978.,351.,259.,781.,106.,966.,257.,681.,713.,156.,756.,302.,939.,149., 623.],
#  [333.,751.,573.,394.,129.,188.,869.,916.,614.,195.,967.,351.,858., 11.,  13.],
#  [924.,715.,784.,830.,900.,112.,111.,187.,249.,122., 85.,385.,503.,285.,  144.],
#  [ 74.,160., 56.,733.,542.,415.,175.,232.,563.,718.,816.,494.,269.,538.,  125.],
#  [981.,967.,790.,512.,345.,628.,126.,798.,389.,402., 65.,840.,596.,698.,  546.],
#  [368.,875., 57.,111.,493.,  2.,666.,481.,731.,198.,750.,459., 68.,350.,  794.],
#  [481.,307.,545.,698.,744.,310.,723., 66.,635.,241.,118.,418.,964.,992.,  131.],
#  [216.,989.,533.,739.,506.,494.,926.,980.,640.,894.,289.,291.,391.,179.,  218.],
#  [272.,539.,595.,663.,534.,739.,730.,570.,912., 38.,707.,183.,633.,751.,  364.],
#  [486.,191.,832.,410.,953.,657.,474.,173.,201.,709.,524.,263.,233.,583.,  153.],
#  [383.,604.,594.,359.,  5.,846.,364.,590.,267.,618.,682.,143.,448.,860.,  956.],
#  [740.,619.,530.,528.,608.,858.,911., 27.,192.,630.,235.,422.,784.,544.,  148.],
#  [133.,105.,187.,617.,454.,131.,322.,197.,612.,789.,175.,975.,538.,544.,  562.],
#  [332.,397.,  5.,974.,563.,109.,992.,182., 38., 23.,373.,620.,663.,144.,  127.],
#  [319.,758.,973.,355.,392.,165.,356.,153.,574.,454.,161.,972.,506.,831.,  364.]])
# np.random.seed(7)
# n = 100
# C = np.matrix(np.round(np.random.rand(n, n) * 1000))
# print(assigment(C))

# while chg and xcount < n:
#     chg = False
#     for i in range(n):
#         if not flagR[i]:
#             count = 0
#             arg = -1
#             for j in range(n):
#                 if z[i, j] == 0 and not flagC[j]:
#                     count += 1
#                     if count == 1:
#                         arg = j
#                     else:
#                         break
#             if count == 1:
#                 xcount += 1
#                 x[i] = arg
#                 flagR[i] = flagC[arg] = True
#                 chg = True
# while xcount < n:
#     argj = argi = -1
#     min = n
#     for i in range(n):
#         if not flagR[i]:
#             zcount = 0
#             zidx = -1
#             for j in range(n):
#                 if not flagC[j] and np.isclose(z[i, j], 0):
#                     zcount += 1
#                     zidx = j
#             if zcount < min:
#                 min = zcount
#                 argi = i
#                 argj = zidx
#         if min == 1:
#             break
#     if min >= 1:
#         x[argi] = argj
#         flagC[argj] = flagR[argi] = True
#         xcount += 1
#     else:
#         break
