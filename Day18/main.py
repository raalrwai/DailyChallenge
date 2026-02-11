from html import parser


class FileParse:
    print("FileParse class is being defined.")
    def __init__(self, filename, userTotals=None):
        self.filename = filename
        self.userTotals = userTotals if userTotals is not None else {}
        print(f"FileParse instance created for file: {filename}")
    def parse(self):
        seen = {}
        try:
            print(f"Parsing file: {self.filename}")
            with open(self.filename, 'r') as file:
                while True:
                    file.readline()
                    if not file.readline():
                        break
                    username, amount = file.readline().strip().split(",")[2:4]  
                    self.userTotals[username] = self.userTotals.get(username, 0) + int(amount)
        except FileNotFoundError:   
            print(f"Error: File {self.filename} not found.")
        else:

    def topThreeSpenders(self):
            sorted_users = sorted(self.userTotals.items(), key=lambda x: x[1], reverse=True)
        return sorted_users[:3]
def main():
    print("This is the main function of main.py.")
    parser = FileParse("orders.csv")
    parser.parse()

if __name__ == "__main__":
    main()