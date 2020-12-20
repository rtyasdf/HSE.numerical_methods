import numpy as np


def jacobi_eig(A: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Метод Якоби поиска всех собственных значений симметричной матрицы.

    Идея метода -- привести матрицу к почти-диагональному виду последовательными
    умножениями на ортогональные матрицы простого вида (матрицы поворота):

                         A := J(i,j).T @ A @ J(i,j)

    Матрица J(i,j) выбирается таким образом, чтобы A[i,j] == 0.
    Общий вид матрицы J(i,j) есть в (1).


    В качестве критерия остановки метода используется неравенство:

     off(A) < tol, где off(A) -- сумма модулей внедиагональных элементов A


    В (1) также приведено квадратное уравнение которое необходимо решить
    для построения матрицы поворота J(i,j):

    t^2 + 2 * t * tau - 1 = 0, где tau = (A[j,j] - A[i,i])/(2 * A[i,j])

    Необходимо выбирать наименьший корень этого уравнения, с тем чтобы
    минимизировать off(A).


    Здесь используется вторая стратегия выбора pivot'а из трёх указанных в (2):
    - выбор максимального внедиагонального элемента матрицы (слишком долго)
    - проход по всей матрице и поворот на каждом элементе
    - учёт максимальных элементов для каждой строки


    Returns:
    eig -- вектор собственных значений матрицы А

    Ссылки:
    1. https://web.stanford.edu/class/cme335/lecture7.pdf
    2. https://wiki.compscicenter.ru/images/1/10/NMM20_lec6.pdf   (слайды 16 - 19)
    """
    n = A.shape[0]
    eig = np.zeros(n)

    for _ in range(max_iter):

        off = 0
        for i in range(n):
            off += np.sum(np.abs(A[i, (i + 1):]))

        if (off < tol):
            break
        else:
            limit = off / (n * (n - 1))
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if (np.abs(A[i, j]) > limit):

                        t = (A[j, j] - A[i, i]) / (2.0 * A[i, j])
                        sq = np.sign(t) * np.sqrt(1 + t**2)
                        tan = t - sq
                        cos = 1 / np.sqrt(1 + tan**2)
                        sin = tan * cos

                        A[i, :], A[j, :] = cos * A[i, :] + sin * A[j, :],  -sin * A[i, :] + cos * A[j, :]
                        A[:, i], A[:, j] = cos * A[:, i] + sin * A[:, j],  -sin * A[:, i] + cos * A[:, j]
                        A[i, j] = A[j, i] = 0.0

    for i in range(n):
        eig[i] = A[i, i]

    return eig


def test(N, it, err):

    for _ in range(it):
        A = np.random.random((N, N))
        A = A @ A.T
        w, v = np.linalg.eig(A)
        u = jacobi_eig(A)
        assert np.linalg.norm(np.sort(u) - np.sort(w)) < err


if __name__ == "__main__":
    test(20, 100, 1e-8)
