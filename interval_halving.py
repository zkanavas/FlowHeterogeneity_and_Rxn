upper_pt = 1
lower_pt = upper_pt/2
stop = False
while not stop:
    print(upper_pt,lower_pt)
    if not stop:
        if not stop:
            upper_pt = lower_pt
            lower_pt = upper_pt/2
        else:
            lower_pt += lower_pt/2
    if upper_pt - lower_pt < 1e-3:
        stop=True