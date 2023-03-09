class Bank_Account:
	def __init__(self):
		self.balance=0
		print("Hello!!!")

	def deposit(self):
		amount=float(input("Enter amount to be Deposited: "))
		self.balance += amount
		print("\n Amount Deposited:",amount)

	def withdraw(self):
		amount = float(input("Enter amount to be Withdrawn: "))
		if self.balance>=amount:
			self.balance-=amount
			print("\n You Withdrew:", amount)
		else:
			print("\n Insufficient balance ")

	def checkbalance(self):
		print("\n The remaining Balance=",self.balance)
	def customer(self):
	    print("Account number:1250098")
	    print("Account opened:2008")
	    print("Account name:Mary Kane")



s = Bank_Account()


s.deposit()
s.withdraw()
s.checkbalance()
