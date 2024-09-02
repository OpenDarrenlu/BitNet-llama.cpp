with open('check_float.txt', 'r') as filef: 
    content = filef.read()
    float_vec = str(content).split(",")[0:-1]

with open('check_i8.txt', 'r') as filei: 
    content = filei.read()
    i8_vec = str(content).split(",")[0:-1]

for i in range(len(float_vec)):
    if float_vec[i] != i8_vec[i]:
        print(float_vec[i])
        print(i8_vec[i])