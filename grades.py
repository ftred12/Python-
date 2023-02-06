#program to display grades
username=input(Enter your username)
sub1=int(input("Enter the marks scored for Maths:"))
sub2=int(input("Enter the marks scored for SDLC:"))
sub3=int(input("Enter the marks scored for OOSAD:"))
avg=(sub1+sub2+sub3)/3
if((avg>=70) and (avg<=100)):
    print("Grade A scored")
elif((avg>=60) and (avg<=69)):
    print("Grade B scored")
elif((avg>=50) and (avg<=59)):
    print("Grade C scored")
elif((avg>=40) and (avg<=49)):
    print("Grade D scored")
else:
    print("Fail")
