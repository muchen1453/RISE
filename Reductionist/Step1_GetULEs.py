# This file read in the obtained SOAP discriptor and train the LAEs
from dscribe.descriptors import SOAP  #Import SOAP descriptor
import math
import numpy as np
import shelve
import os
import datetime
from random import shuffle

############################# Print current time ##############################
currentDT = datetime.datetime.now()
print(str(currentDT))



global LAEs, corresponding_structure
LAEs, corresponding_structure = [], []
global G_v
G_v = globals()

def Get_LAEs(real_threshold, coolingrate, temperature, stru_idx):
    file_idx = stru_idx - (stru_idx - 1) // num_eachtemp * num_eachtemp
    soapStru = np.load('../../soap/' + coolingrate + '_' + str(temperature) + '_soap_MG' + str(file_idx) + '.npy')
    normStru = np.linalg.norm(soapStru, axis=1)
    for i in range(len(soapStru)):
        soap_v = soapStru[i] / normStru[i]
        same = 1
        for uni_feat in LAEs:
            if np.dot(uni_feat, soap_v) > real_threshold:
                same = 0
                break
        if same:
            LAEs.append(soap_v)
            corresponding_structure.append((coolingrate, stru_idx, i+1))

################################### Main #####################################
num_structures = 1000
num_eachtemp = 200
random_partion = 0.4   #percentage of training data, rest of them would be test data
threshold = 0.83    #In paper it is 0.9975
real_threshold = 1 - (1 - threshold)**2

for inilabel in ['10_1', '11_1', '12_1', '13_1']:
    if inilabel == '12_1' or inilabel == '10_1':
        sublabelSet = ['a', 'b', 'c']
    if inilabel == '11_1' or inilabel == '13_1':
        sublabelSet = ['a', 'b']

        
    for sublabel in sublabelSet:
        coolingrate = inilabel + sublabel
        structures = list(range(1,num_structures + 1))
        shuffle(structures)
            
        for idx in range(int(random_partion*num_structures)):
            temperature = (structures[idx] - 1) // num_eachtemp * 50 + 500
            Get_LAEs(real_threshold, coolingrate, temperature, structures[idx])

            with open('Processing.txt','a') as output:
                output.write("%d " % temperature)
                output.write("%d " % idx)
                output.write("%d    " % structures[idx])
                output.write("%d\n" % len(LAEs))


############################### Write the corresponding_structure to get the actual atomic configuraion #############
filename = './Atomic_configuration.txt'
with open(filename,'x') as output:
    for i in range(len(corresponding_structure)):
        output.write("%s %d %d\n" % corresponding_structure[i])

######################## Save the workspace for LAES ##################################
filename = './LAES.out'
my_shelf = shelve.open(filename,'n')
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

############################# Print current time ##############################
print('length of LAEs is',len(LAEs),'length of corresponding_structure is', len(corresponding_structure))
currentDT = datetime.datetime.now()
print(str(currentDT))
