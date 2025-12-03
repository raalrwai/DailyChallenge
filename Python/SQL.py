import pandas as pd
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())

customers = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles'],
    'signup_year': [2020, 2021, 2020, 2019, 2021]
})

orders = pd.DataFrame({
    'id': [101, 102, 103, 104, 105, 106, 107],
    'customer_id': [1, 2, 1, 3, 5, 2, 4],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Monitor', 'Keyboard', 'Mouse'],
    'quantity': [1, 2, 1, 1, 2, 1, 3],
    'price': [1200, 25, 50, 1200, 300, 50, 25],
    'order_year': [2020, 2021, 2020, 2020, 2021, 2022, 2020]
})

query1 = """
SELECT name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.customer_id IS NULL;
"""

query2 = """
SELECT c.name, SUM(o.quantity * o.price) AS total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.name
ORDER BY total_spent DESC;
"""

query3 = """
SELECT product, SUM(quantity) AS total_quantity
FROM orders
GROUP BY product
ORDER BY total_quantity DESC
LIMIT 1;
"""

query4 = """
SELECT c.name, MIN(o.order_year) AS first_order_year
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.name
ORDER BY first_order_year;
"""

query5 = """
SELECT DISTINCT c.name, o.product, o.quantity
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE c.city = 'New York' AND o.quantity > 1;
"""

results1 = pysqldf(query1)
results2 = pysqldf(query2)
results3 = pysqldf(query3)
results4 = pysqldf(query4)
results5 = pysqldf(query5)

print("Query 1:\n", results1, "\n")
print("Query 2:\n", results2, "\n")
print("Query 3:\n", results3, "\n")
print("Query 4:\n", results4, "\n")
print("Query 5:\n", results5, "\n")
