import numpy as np
import pandas as pd
data = pd.read_csv("ws.csv", header=None)
# header = None indicates csv file has no headers

concepts = np.array(data.iloc[:, 0:-1])  # [rows, columns]
# iloc() functions helps us to select specific row or column from  data set.

print("\nInstances are:\n", concepts)
target = np.array(data.iloc[:, -1])
print("\nTarget Values are: ", target)

for i, h in enumerate(concepts):
    print(i+1, h)


def learn(concepts, target):
    specific_h = concepts[0].copy()
    # copy first row of training data to specific_h
    print("\nInitialization of specific_h and genearal_h")
    print("\nSpecific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]
    print("\nGeneric Boundary: ", general_h)

    for i, h in enumerate(concepts):  # iterate through all the attributes of the concept
        print("\nInstance", i+1, "is ", h)
        if target[i] == "Yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'   # {Q,Q,Q,Q,Q,Q} -specific_hypothesis
                    # {?,?, ?, ?, ?} - generic_hypothesis
                    # attribute not in specific_hypothesis but in generic_hypothesis should also be removed
                    general_h[x][x] = '?'
        if target[i] == "No":
            # instance classified as no by specific_hypothesis is (which is right) classified as yes by generic_hypothesis because of ? (don't care condition) so we have to make it less generic by replacing the consist attribute of specific_hypothesis.
            print("Instance is Negative ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Specific Boundary after ", i+1, "Instance is ", specific_h)
        print("Generic Boundary after ", i+1, "Instance is ", general_h)
        print("\n")

    # list comprehension
    # remove list items containing only question marks
    indices = [index for index, val in enumerate(general_h) if val == [
        '?', '?', '?', '?', '?', '?']]
    for i in range(len(indices)):
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")
