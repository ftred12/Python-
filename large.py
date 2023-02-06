#program to display the largest of three numbers
num1=int(input("Enter the first number:"))
num2=int(input("Enter the second number:"))
num3=int(input("Enter the third number:"))
if((num1>num2) and (num1>num3)):
    print("Number 1 is the largest")
elif((num2>num1) and (num2>num3)):
    print("Number 2 is the largest")
else:
    print("Number 3 is the largest")
