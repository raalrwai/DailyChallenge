def sum_of_even_squares(n):
    if len(n) == 0:
        return 0
    sum = 0
    for i in n:
        if i %2 ==0:
            sum += i*i
    return sum

print(sum_of_even_squares([1, 2, 3, 4, 5]))  # 20
print(sum_of_even_squares([]))               # 0
print(sum_of_even_squares([1, 3, 5]))        # 0

orders = [
{"order_id": 1, "customer": "Alice", "amount": 120.5, "status": "shipped"},
{"order_id": 2, "customer": "Bob", "amount": 35.0, "status": "pending"},
{"order_id": 3, "customer": "Alice", "amount": 75.25, "status": "shipped"},
{"order_id": 4, "customer": "Dina", "amount": 220.0, "status": "cancelled"},
{"order_id": 5, "customer": "Bob", "amount": 15.0, "status": "shipped"},
]

def get_total_revenue(orders):
    count = 0
    for i in orders:
        for j in i.items():
            if j[1] == "shipped":
                count +=1
                print(count)
    return count

def totalPerCustomer(orders):
    spent = {}
    for i in orders:
        print(i)
        spent[i["customer"]] = spent.get(i.get("customer"), 0) + i["amount"]   
    return spent
totalPerCustomer(orders)          

def top_n_customers(dictCust, n):
    i = 0
    result = []
    while i < n:
        for i, n in dictCust.items():

            if n > max:
                result.append(i)
    return result

def filterOrders(orders, minAmount, status):
    
    for i in orders:
        if status in i:
            del(i)
        