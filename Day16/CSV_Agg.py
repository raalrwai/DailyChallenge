import sys
import pandas as pd

def main():
    print("Hello World!")
    spending = {}
    filtered = []
    with open("orders.csv", "r") as f:
        lines = f.readlines()
    header = lines[0]
    for line in lines[1:]:
        parts = line.strip().split(",")
        order_id, user_id, user_name, product, amount, date = parts
        spending[user_name] = spending.get(user_name, 0) + int(amount)
        if int(amount) > 200:
            filtered.append(parts)
    print(filtered)  
    top_three = sorted(spending.items(), key = lambda x : x[1], reverse = True)[:3]
    print(top_three)

if __name__ == "__main__":
    main()
