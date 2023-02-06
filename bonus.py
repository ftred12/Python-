#program to display bonus
sal=int(input("Enter the salary earned:"))
years=int(input("Enter the years of service:"))
if(years>10):
    print("Net bonus=",sal+(sal*0.1))
elif((years>=6) and (years<=10)):
    print("Net bonus=",sal+(sal*0.08))
else:
    print("Net bonus=",sal+(sal*0.05))
