import csv

class FileParse:
    
    print("FileParse class is being defined.")

    def __init__(self, filename, userTotals=None):
        self.filename = filename
        self.userTotals = userTotals if userTotals is not None else {}
        print(f"FileParse instance created for file: {filename}")

    def parse(self):
        try:
            print(f"Parsing file: {self.filename}")
            with open(self.filename) as file:
                reader = csv.DictReader(file)
                for row in reader:
                        username = row["user_name"]
                        amount = float(row["amount"])
                        self.userTotals[username] = self.userTotals.get(username, 0) + int(amount)
        except FileNotFoundError:   
            print(f"Error: File {self.filename} not found.")
        else:
            print(f"Successfully parsed file: {self.filename}")

    def topThreeSpenders(self):
            sorted_users = sorted(self.userTotals.items(), key=lambda x: x[1], reverse=True)
            return sorted_users[:3]
    
def main():
   
    print("This is the main function of main.py.")
    parser = FileParse("orders.csv")
    parser.parse()
    top_spenders = parser.topThreeSpenders()
    print("Top 3 spenders:")
    
    for user, total in top_spenders:
        print(f"User: {user}, Total: {total}")

if __name__ == "__main__":
    main()