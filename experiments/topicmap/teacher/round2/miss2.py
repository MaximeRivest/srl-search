dm = J.decider_scores(DA, mt, print)
print("decider on the 159 missed: >=0.5:", int((dm >= .5).sum()), " >=0.9:", int((dm >= .9).sum()), " <0.1:", int((dm < .1).sum()))
lowS = ms < .1
print("  of the 101 the students put below 0.1, decider >= 0.5:", int((dm[lowS] >= .5).sum()))
inDec = sum(w in decided for w in miss); print("missed refs the decider already labelled in its 10k:", inDec)
