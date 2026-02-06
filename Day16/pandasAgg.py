import sys
from time import time
import pandas as pd

def main():  
    def process_orders():
        chunk_size = 10000

        df = pd.read_csv("orders.csv")
        required_columns = {"user_name", "amount"}
        df_totals = df.groupby("user_name")["amount"].sum()
        top_users = df_totals.sort_values(ascending=False).head(3)
        print("Top 3 users by total order amount:")
        for user, amount in top_users.items():
            print(f"{user}: {amount}")
    process_orders()
if __name__ == "__main__":
    main()

