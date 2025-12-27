# This file read in the obtained SOAP discriptor and Get the LER for each IS structure with trained LAEs
from ase import Atoms   #Import ase.Atoms type
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


global G_v, energy_train, energy_test
G_v = globals()


#short_feature = [628]

#Given a system of soap discriptor of a structure and LAEs,
#get the LER (local environment representation) vector
def feat_matrix(coolingrate, real_threshold, stru_idx):
    matrix_train = np.array([]).reshape(0,len(feature_LAEs[0]))
    matrix_test = np.array([]).reshape(0,len(feature_LAEs[0]))
    
    for idx in range(1,num_structures+1):
        temperature = (idx - 1) // num_eachtemp * 50 + 500
        vec_ler = np.zeros(len(feature_LAEs[0]))
        vec_sum = 0
        
        currentDT = datetime.datetime.now()
        with open('processing.txt','a') as output:
            output.write("%s " % str(currentDT))
            output.write("%s " % coolingrate)
            output.write("%d\n" % idx)
        
        print('cur state is',coolingrate,' ', idx)
        
        file_idx = idx - (idx - 1) // num_eachtemp * num_eachtemp
        soapStru = np.load('../../soap/' + coolingrate + '_' + str(temperature) + '_soap_MG' + str(file_idx) + '.npy')
        normStru = np.linalg.norm(soapStru, axis=1)
        for i in range(len(soapStru)):
            soapStru[i] /= normStru[i]

        similarity = np.dot(soapStru, feature_LAEs)
        for vecSim in similarity:
            if max(vecSim) > real_threshold:
                vec_ler[np.argmax(vecSim)] += 1
                vec_sum += 1
        if vec_sum > 1:
            vec_ler = vec_ler/vec_sum

        if idx in stru_idx:
            matrix_train = np.vstack((matrix_train, vec_ler))
            energy_train.append(ISenergy[idx-1])
        else:
            matrix_test = np.vstack((matrix_test, vec_ler))
            energy_test.append(ISenergy[idx-1])
            
        del soapStru
    return matrix_train, matrix_test

def energy_label(filename):
    with open(filename,'r') as input:
        label = input.readlines()
    for i in range(len(label)):
        label[i] = float(label[i].strip())
    return label

############################################ Main ##############################################
num_structures = 1000
num_eachtemp = 200
random_partion = 0.8        #percentage of training data, rest of them would be test data
threshold = 0.83        #In paper it is 0.9975
real_threshold = 1 - (1 - threshold)**2

############# Read in obtained LAEs ###############################
shelfname = '../LAEs/LAES.out'
my_shelf = shelve.open(shelfname)
for key in my_shelf:
    if key == 'LAEs':
        try:
            G_v[key]=my_shelf[key]
        except:
            print('ERROR shelving: {0}'.format(key))
        break
    else:
        print(key)
my_shelf.close()

#feature_LAEs = [LAEs[i] for i in short_feature]
feature_LAEs = np.transpose(LAEs)      # Include all the featueres
print('Size of feature_LAEs is',len(feature_LAEs), '  ',len(feature_LAEs[0]))


for inilabel in ['10_1', '11_1', '12_1', '13_1']:
    if inilabel == '12_1' or inilabel == '10_1':
        sublabelSet = ['a', 'b', 'c']
    if inilabel == '11_1' or inilabel == '13_1':
        sublabelSet = ['a', 'b']

    for sublabel in sublabelSet:
        coolingrate = inilabel + sublabel

        energy_train, energy_test = [], []
        training_label = list(range(1,num_structures + 1))

        ISenergy = []
        for temperature in [500, 550, 600, 650, 700]:
            energy_file = '../../../../../2000_Data/Cooling' + coolingrate + '/' + str(temperature) + '/Sort/ML' + coolingrate + '_Energy.txt'
            ISenergy += energy_label(energy_file)

        matrix_train, matrix_test = feat_matrix(coolingrate, real_threshold, training_label)
        del training_label, ISenergy

        
        ########################## Write the SOAP descriptors into a file #############
        output_train_file = './SOAP_train_' + coolingrate + '.txt'
        with open(output_train_file,'a') as output:
            for i in range(len(matrix_train)):
                output.write("%lf" % energy_train[i])
                for id in range(len(matrix_train[i])):
                    output.write(" %d:%e" %(id,matrix_train[i][id]))
                output.write("\n")
        print('coolingrate is',coolingrate,'training data length is',len(matrix_train))

        output_test_file = './SOAP_test_' + coolingrate + '.txt'
        with open(output_test_file,'a') as output:
            for i in range(len(matrix_test)):
                output.write("%lf" % energy_test[i])
                for id in range(len(matrix_test[i])):
                    output.write(" %d:%e" %(id,matrix_test[i][id]))
                output.write("\n")
        print('coolingrate is',coolingrate,'test data length is',len(matrix_test))
        del energy_train, energy_test, matrix_train, matrix_test
    
############################# Print current time ##############################
currentDT = datetime.datetime.now()
print(str(currentDT))




numStruct = 1000
train_number = 600
validation_number = 200
testing_number = 200
parameter_number = 200

for inilabel in ['10_1', '11_1', '12_1', '13_1']:
    if inilabel == '12_1' or inilabel == '10_1':
        sublabelSet = ['a', 'b', 'c']
    if inilabel == '11_1' or inilabel == '13_1':
        sublabelSet = ['a', 'b']

    for sublabel in sublabelSet:
        coolingrate = inilabel + sublabel

        structures = list(range(numStruct))
        dataname = './SOAP_train_' + coolingrate + '.txt'
        with open(dataname,'r') as data:
            lines = data.readlines()        

        shuffle(structures)
        with open('SOAP_training.txt','a') as training:
            with open('ML_training_ISenergy.txt','a') as trainingenergy:
                for i in range(train_number):
                    training.write('%s' % lines[structures[i]])
                    line = lines[structures[i]].split()
                    trainingenergy.write('%s\n' % line[0])

        with open('SOAP_validation.txt','a') as validation:
            with open('ML_validation_ISenergy.txt','a') as validationenergy:
                for i in range(train_number, train_number + validation_number):
                    validation.write('%s' % lines[structures[i]])
                    line = lines[structures[i]].split()
                    validationenergy.write('%s\n' % line[0])

        with open('SOAP_testing.txt','a') as testing:
            with open('ML_testing_ISenergy.txt','a') as testingenergy:
                for i in range(train_number + validation_number, train_number + validation_number + testing_number):
                    testing.write('%s' % lines[structures[i]])
                    line = lines[structures[i]].split()
                    testingenergy.write('%s\n' % line[0])

        structures = structures[:train_number]
        shuffle(structures)
        with open('SOAP_parameter.txt','a') as parameter:
            with open('ML_parameter_ISenergy.txt','a') as parameterenergy:
                for i in range(parameter_number):
                    parameter.write('%s' % lines[structures[i]])
                    line = lines[structures[i]].split()
                    parameterenergy.write('%s\n' % line[0])
