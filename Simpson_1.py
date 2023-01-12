
## THIS SCRIPT TEST SIMPSON'S PARADOX FOR A POPULATION OF INDIVIDUALS (COOEPERATORS AND NON-COOPERATORS) DIVIDED INTO SUBPOPULATIONS 
## WITH GROUP SELECTION
## THE COOEPERATORS GIVE A BENEFIT TO EVERY INDIVIDUAL IN THE SUBPOPULATION
## THE BENEFIT VALUE CAN BE VARIED BY YOU ON LINE 321-322
## A CARRYING CAPACITY IS ALSO INCLUDED FOR THE SUBPOPULATIONS
## YOU CAN VARY THE CARRYING CAPACITY ON LINE 353


import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import timeit


### CLASSES ###

# class for the parameters
class sim_parameters:
    def __init__(self, popsize=100, subpop=10, numgen=20, cycles=500, freqcoop=0.5, 
                    b=0.0, r1=0.1, r0=0.2, K1 = None, K0=0, m=0.01, replicates=5, step=5):
        self.popsize = popsize
        self.subpop = subpop
        self.numgen = numgen
        self.cycles = cycles
        self.freqcoop = freqcoop
        self.b = b
        self.r1 = r1
        self.r0 = r0
        self.K1 = K1
        self.K0 = K0
        self.m = m
        self.replicates = replicates
        self.step = step

    def __str__(self) -> str:
        return f"popsize,{self.popsize},,subpop,{self.subpop},,numgen,{self.numgen},,cycles,{self.cycles}\nfreqcoop,{self.freqcoop},,b,{self.b},,r1,{self.r1},,r0,{self.r0}\nK1,{self.K1},,K0,{self.K0},,mut_chance,{self.m},,replicates,{self.replicates},,datastep,{self.step}\n\n"

# class for the individuals in the population, they have a type and subpopulation number they belong to.
class Ind_class:
    def __init__(self, type=None, subpop=None):
        self.type = type
        self.subpop = subpop

    def __str__(self):
        return f"{self.type}, {self.subpop}"


### FUNCTIONS ###

### FUNCTIONS TO ASSIGN TYPE AND SUBPOP

# assign type of individual; cooperator(1) or non-cooperator(0)
def assign_type(ind, freqcoop):
    if random.uniform(0,1) <= freqcoop:
        ind.type = 1 
    else:
        ind.type = 0 

# assign which subpop individual belongs to for all ind in the population
def assign_subpop(pop, subpop):
    
    new_pop = []
    num_ind_in_subpop = int(len(pop)/subpop)

    for i in range(subpop):

        for j in range(num_ind_in_subpop):
            rnd_ind = random.choice(pop)
            rnd_ind.subpop = i
            pop.remove(rnd_ind)
            new_pop.append(rnd_ind)

    return new_pop


### REPRODUCTION FUNCTIONS ###

# count frequency of cooperators  within subpopulations, returns list of freqs, index is also subpop number

def count_freq_in_subpops(pop, subpop):
    subpop_freqlist = []
    subpop_size = []

    for i in range(subpop):
        subpopsize = 0
        count = 0

        for ind in pop:
            if ind.subpop == i:
                subpopsize += 1
                count += ind.type
        
        subpop_freq = count/subpopsize
        subpop_freqlist.append(subpop_freq)
        subpop_size.append(subpopsize)

    
    return subpop_freqlist


# reproduction of ind in pop includiing mutation chance (for 1 generation)
def reprod(pop, mut_rate, freq_list, benefit, r1, r0, K1, K0):

    dupli_pop = pop.copy()
    
    # reproduction with carrying capacity K0
    if K0 != 0:

        # create dictionary with all subpopulations
        subpop_dict = {}

        for i in range(len(freq_list)):
            subpop_dict[f"Subpop{i}"] = []
        
        for ind in pop:
            subpop_dict[f"Subpop{ind.subpop}"].append(ind)
        
        # select individuals (without putting them back) to reproduce for next generation from a randomly shuffled list
        # do this for each subpop
        for sp in subpop_dict: # sp = subpop
            random.shuffle(subpop_dict[sp])

            count = 0
            # allow reproduction until no individuals are left in subpop list or until carrying capacity has been reached 
            while len(subpop_dict[sp]) != 0 and count < K0:
                ind = subpop_dict[sp].pop()

                # reproduction chance for either cooperator (1) or non-cooperator (0)
                if ind.type == 1:
                    reprod_chance = r1 + freq_list[ind.subpop] * benefit * (1 - r0)
                elif ind.type == 0:
                    reprod_chance = r0 + freq_list[ind.subpop] * benefit * (1 - r0)

                # reproduce
                if random.uniform(0,1) <= reprod_chance:
                    new_ind = Ind_class()
                    new_ind.subpop = ind.subpop
                    
                    # mutation (if chance happens then individual switches type from coop to non-coop or vice versa)
                    if random.uniform(0,1) <= mut_rate:
                        new_ind.type = abs(ind.type - 1)  
                    else:
                        new_ind.type = ind.type

                    pop.append(new_ind)
                    count += 1
        
     # reproduction without carrying capacity   
    else:

        for ind in dupli_pop:
            if ind.type == 1:
                reprod_chance = r1 + freq_list[ind.subpop] * benefit * (1 - r0)
            elif ind.type == 0:
                reprod_chance = r0 + freq_list[ind.subpop] * benefit * (1 - r0)

            if random.uniform(0,1) <= reprod_chance:
                new_ind = Ind_class()
                new_ind.subpop = ind.subpop
                
                if random.uniform(0,1) <= mut_rate:
                    new_ind.type = abs(ind.type - 1)  
                else:
                    new_ind.type = ind.type
            
                pop.append(new_ind)
    
    


 ## reproduction for 20 generations of the subpops = reproduction for 1 cycle

def subpop_reprod(numgen, pop, subpop, m, b, r1, r0, K1, K0, step):

    # dataframe to store data of freqs
    df = pd.DataFrame( columns= ["Avg_freq", *[f"Subpop_{j}" for j in range(subpop)], "Avg_subpops"])
    
    avg_freqlist = []
    
    for i in range(numgen):

        # create freq list with freq of cooperators in each subpop
        freq_list = count_freq_in_subpops(pop, subpop)
        # add the average of the freq list to the same list
        freq_list.append(np.average(freq_list))
        
        # record data for every (step) generation instead of all generations, saves compute time
        if i%step == 0:

            count = 0
            for ind in pop:
                count += ind.type
            
            avg_frequency =  count/len(pop)

            data_list = [avg_frequency, *freq_list]

            df.loc[len(df)] = data_list

        #reproduction
        reprod(pop, m, freq_list, b, r1, r0, K1, K0)
    
    
    return df


## selecting new population and subpopulations for the next cycle

def new_metapop(pop, popsize, subpop):
    new_pop = []
    dupli_pop = pop.copy()

    for i in range(popsize):

        ind = random.choice(dupli_pop)
        new_pop.append(ind)
        dupli_pop.remove(ind)

    pop = new_pop
    
    # assign subpop to new individuals in population
    pop = assign_subpop(pop, subpop)
    
    return pop


## simulation function (includes data collection)

def simulator(parameters, v_par, v_value, path):

    p = parameters

    # create folder location to store data
    if os.path.exists(path) == False:
        os.mkdir(path)

    # dataframe (df) to store data for all replicates
    df_all_replicates = pd.DataFrame(columns= ["Avg_freq", *[f"Subpop_{k}" for k in range(p.subpop)], "Avg_subpops"], index=[0])


    ## run replicates
    for j in range(p.replicates):
       
        ## make populations

        pop = []

        for i in range(p.popsize):
            ind = Ind_class()
            assign_type(ind, p.freqcoop)
            pop.append(ind)

        pop = assign_subpop(pop, p.subpop)

        ## run cycles
        
        # keep track of number of generations
        numgen_count = 0
        numgen_list = []
        
        # dataframe (df) to store data for single replicate
        df_old = pd.DataFrame(columns= ["Avg_freq", *[f"Subpop_{k}" for k in range(p.subpop)], "Avg_subpops"], index=[0])
        

        for i in range(p.cycles):
            # keep track of number of generations
            numgen_list = [j for j in range(numgen_count, numgen_count + p.numgen, p.step)]
            numgen_count = p.step + numgen_list[-1]
            
            # get dataframe with data after one cycle
            df_new = subpop_reprod(p.numgen, pop, p.subpop, p.m, p.b, p.r1, p.r0, p.K1, p.K0, p.step)
        
            df_new.index = numgen_list

            # add new dataframe to old dataframe: df_old = df_old + df_new
            df_old = pd.concat([df_old, df_new], ignore_index=False)

            # create new population and subpop of individuals for next cycle
            pop = new_metapop(pop, p.popsize, p.subpop)

        # add result of new replicate to that of old replicates
        df_all_replicates = df_all_replicates.add(df_old, fill_value=0)
        
        # convert dataframe to csv file and also add all parameters necessary to csv file
        f = open(path + f"\{v_par}_{v_value}_R{j}.csv", 'w')
        f.write(current_parameters.__str__())
        f.close() 
        f = open(path + f"\{v_par}_{v_value}_R{j}.csv", 'a',  newline='')
        df_old.to_csv(f)
        f.close()
    
    # get average results of all replicates
    df_all_replicates =  df_all_replicates/p.replicates
    

    # convert dataframe all replicates to csv file and also add all parameters necessary to csv file
    f = open(path + f"\{v_par}_{v_value}_all_replis.csv", 'w')
    f.write(current_parameters.__str__())
    f.close() 
    f = open(path + f"\{v_par}_{v_value}_all_replis.csv", 'a',  newline='')
    df_all_replicates.to_csv(f)
    f.close()
    
    

    

###############################################################################################
###                                          MAIN                                           ###
###############################################################################################

# parameters for benefit (b) experiments
current_parameters = sim_parameters(K0=25, numgen=20, cycles=20, r0=0.125, r1=0.1, freqcoop=0.3, step=4)
print(current_parameters)

# values of benefit we want to test 
b_values = [round(i*0.1, 1) for i in range(0,11)]

# ADD YOUR OWN B-VALUES HERE
# b_values = []

# run simulator for each benefit value
for v_value in b_values:

    v_par = "Benefit" #variable parameter

    current_parameters.b = v_value
    print(current_parameters.b)

    # change working directory so files are saved in same directory as the script
    cwd = os.getcwd()
    path = cwd + f"\{v_par}_{v_value}"

    # timer
    start = timeit.default_timer()

    # simulator
    current_df = simulator(current_parameters, v_par, v_value, path)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  


# parameters for carrying capacity (K0) experiments
current_parameters = sim_parameters(b=0.8, K0=25, numgen=20, cycles=20, r0=0.125, r1=0.1, freqcoop=0.3, step=4)

# carrying cap values we want to test
K0_values = [0, 25, 50, 500, 1000, 5000]

for v_value in K0_values:

    v_par = "K0"

    current_parameters.K0 = v_value
    print(current_parameters.K0)

    # set working directory to right one and create path to store data files
    cwd = os.getcwd()
    path = cwd + f"\{v_par}_{v_value}"

    # timer
    start = timeit.default_timer()

    # run simulator
    current_df = simulator(current_parameters, v_par, v_value, path)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  



#END

    

    