


ch = input()

dict = {}
for i in range(len(ch)):
    if ch[i] not in dict:
        dict[ch[i]] = 1
    else:
        dict[ch[i]] += 1




def solution(dict):
    palindrome = ''
    flag = 1
    solution = True
    value =0

    minValue = min(dict.values())
    maxValue = max(dict.values())

    
    if (minValue%2 ==0) and (maxValue%2 == 0):
        for key in dict:
            if dict[key]%2 == 1:
                value = dict[key]
                break
            else:
                value = minValue
    elif minValue % 2 == 0:
        value = maxValue
    else:
        value = minValue

    


    while dict:

        for key in dict:
            if dict[key] == value:
                if value % 2 == 0:
                    palindrome = (key * int(value/2) + palindrome + key * int(value/2))
                   
                    flag = 0
                    break
                elif flag == 1:
                    palindrome +=  key * value
                    flag = 0
                    break
                else:
                    solution = False
                    return solution, palindrome
        dict.pop(key)
        if dict:
            value = min(dict.values())
    return solution, palindrome


solution, palindrome = solution(dict)
if solution:
    print(palindrome)
else:
    print("NO SOLUTION")
    
