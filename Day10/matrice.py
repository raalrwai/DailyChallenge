matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

def rotateMatrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix [j][i], matrix[i][j]

    for row in matrix:
        row.reverse()
    return matrix

print(rotateMatrix(matrix))

def transposeMatrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix [j][i], matrix[i][j]
    return matrix

print(transposeMatrix(matrix))


def rotateCounterClockwise(matrix):
    n = len(matrix)
    
    # Step 1: transpose
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Step 2: reverse columns
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
    
    return matrix

print(rotateCounterClockwise(matrix))
