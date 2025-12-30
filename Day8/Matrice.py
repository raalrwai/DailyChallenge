# =========================
# Example Matrices
# =========================

A = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

B = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]

I = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# =========================
# Helper Functions
# =========================

def shape(matrix):
    return len(matrix), len(matrix[0])


def transpose(matrix):
    return [
        [matrix[r][c] for r in range(len(matrix))]
        for c in range(len(matrix[0]))
    ]


def rotate_90_clockwise(matrix):
    return [
        [matrix[r][c] for r in range(len(matrix) - 1, -1, -1)]
        for c in range(len(matrix[0]))
    ]


def rotate_90_counterclockwise(matrix):
    return [
        [matrix[r][c] for r in range(len(matrix))]
        for c in range(len(matrix[0]) - 1, -1, -1)
    ]


def rotate_180(matrix):
    return [row[::-1] for row in matrix[::-1]]


def flip_horizontal(matrix):
    return [row[::-1] for row in matrix]


def flip_vertical(matrix):
    return matrix[::-1]


def scalar_multiply(matrix, scalar):
    return [
        [scalar * value for value in row]
        for row in matrix
    ]


def add_matrices(A, B):
    rows, cols = shape(A)
    return [
        [A[r][c] + B[r][c] for c in range(cols)]
        for r in range(rows)
    ]


def multiply_matrices(A, B):
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)

    if cols_A != rows_B:
        raise ValueError("Incompatible dimensions")

    return [
        [
            sum(A[i][k] * B[k][j] for k in range(cols_A))
            for j in range(cols_B)
        ]
        for i in range(rows_A)
    ]


# =========================
# Example Usage
# =========================

print("Transpose of A:")
print(transpose(A))

print("\nRotate A 90° clockwise:")
print(rotate_90_clockwise(A))

print("\nFlip B horizontally:")
print(flip_horizontal(B))

print("\nA × I:")
print(multiply_matrices(A, I))
