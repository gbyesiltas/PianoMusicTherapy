from dataclasses import dataclass
from dataclasses import dataclass, field
import numpy as np

#This file was created starting from the reference above:
#Xie et al, 2020

#It implements the MAIBT algorithm for beat tracking


@dataclass
class Cluster:
    score: float = 0
    IOIs: list = field(default_factory=list)
    mean_IOI: float = 0
    num: int = 0
    related_clusters: list = field(default_factory=list)

@dataclass
class RelatedInfo:
    d: int=0
    interval: float=0

@dataclass
class Agent:
    beat_interval: float=0
    predict: float=0
    tempo_score: float=0
    score: float = 0
    hist: list = field(default_factory=list)


def average(lst):
    return sum(lst) / len(lst)

def cluster_score(d,orig=False):
    if d <= 4 and d >= 1:
        score = 6 - d
    elif d<=8 and d >= 5:
        if orig:
            score = 1
        else:
            score = 0.2
        
    else:
        score = 0

    return score

def clean_notes(note_events,thres,mode):
    # depends on how the filter window is sliced and which note is removed, three modes are implemented in this function, the third mode is used in our paper
    # mode 1: the filter window is sliced based on note events, only the first note event is retained
    # mode 2: the whole sequence is sliced into equal pieces(each one is a filter window), filter by velocity
    # mode 3: the filter window is sliced based on note events, filter by velocity

    if mode == 1:
        # only retain the first note event within each small window
        i = 0
        while i (note_events):
            basetime = note_events[i][5]
            i = i + 1
            while (i<=len(note_events) and note_events[i][5] < basetime + thres):
                note_events[i][5] = -1  # delete all other note events
                i = i + 1
            
    elif mode == 2:
        # filter by slicing the whole sequence into equal pieces, each piece is
        # essentially a filter window, whose width is determined by threshold
        thres = 0.3
        pointer = 0        # pointer to the last note to be processed
        for t in range(0,note_events[-1][5],thres) :  
            vmax_index = pointer
            
            # pick out the event with the highest velocity within the threshold
            # window

            while (pointer <= len(note_events) and note_events[pointer][5] < t):
                if note_events[pointer][4] < note_events[vmax_index][4]:
                    note_events[pointer][5] = -1  # delete this note event
                else:
                    if pointer != vmax_index:
                        note_events[vmax_index][5] = -1  # delete previous largest velocity note
                    
                    vmax_index = pointer     # update index of maximum velocity
                
                pointer = pointer + 1
            
    elif mode == 3:
        # The mode used in the paper
        # the window starts from the timestamp of a certain note event, and
        # pick out the event with the highest velocity within that window. 
        thres = 0.25    
        i = 0
        while i < len(note_events):
            basetime = note_events[i][5]   # the basetime when the filter window starts
            vmax_index = i
            i = i + 1
            
            # pick out the event with the highest velocity within the threshold
            # window
            while (i<len(note_events) and note_events[i][5] < basetime + thres):
                if note_events[i][4] < note_events[vmax_index][4]:
                    note_events[i][5] = -1  # delete this note event
                else:
                    note_events[vmax_index][5] = -1  # delete previous largest velocity note event
                    vmax_index = i     # update index of maximum velocity
                
                i = i + 1
    
    start_time = note_events[:,5]
    filtered_events = note_events[start_time!=-1][:]  # remove deleted notes

    return filtered_events
    
def get_clusters(events, orig=False):
    # cluster the notes, corresponding to the second step as described in the paper
    # Inputs:
    #         events: note matrix
    #         orig: option, use score calculation in the original method or not
    # Output:
    #         clusters: an array of struct, each cluster has these attributes
    #             score: the cluster score
    #             IOIs: all inter-onset intervals
    #             mean_IOI: average IOI
    #             num: number of note events fall into this cluster
    #             related_clusters: information of the related clusters
    # Reference:
    #         S.  Dixon,  Automatic  extraction  of  tempo  and  beat  from  expressiveperformances,Journal Of New Music Research, vol. 30, no. 1, pp. 3958, 2001.

    cluster_width = 0.080  # set threshold to separate clusters ==>> resolution of the tempo
    min_IOI = 0.1    # minimum IOI that two events will be counted as two separate notes, to avoid noise from duplicate notes
    clusters = []

    if orig:
        upper_limit = len(events)  # the original version: search for all possible pairs 
    else:
        upper_limit = 3   # maximum number of events to search after current note event

    for i in range(len(events)):
        upper_bound = i + upper_limit + 1
        if upper_bound > len(events):
            upper_bound = len(events)     # prevent index out of boundary
        for j in range(i,upper_bound):

            IOIij = abs(events[i][5] - events[j][5])

            if IOIij > min_IOI:
                # two events can be counted as two separate notes
                for k in range(len(clusters)): 
                    if abs(IOIij - clusters[k].mean_IOI) < cluster_width:
                        if IOIij not in clusters[k].IOIs:
                            clusters[k].IOIs.append(IOIij)   # add to existing cluster
                            clusters[k].mean_IOI = average(clusters[k].IOIs)   
                            clusters[k].num = clusters[k].num + 1  # increment number
                        
                        IOIij = -1 # clear IOIij
                        break
            
                if IOIij !=-1:
                    # create a new cluster
                    cluster = Cluster()
                    cluster.num = 1
                    cluster.IOIs = [IOIij]   
                    cluster.mean_IOI = IOIij
                    clusters.append(cluster)
    # merge close clusters
    if len(clusters)>1:
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if (j!= i) and abs(clusters[i].mean_IOI - clusters[j].mean_IOI) < cluster_width:
                    clusters[i].num = clusters[i].num + clusters[j].num     # merge two clusters
                    clusters[i].IOIs = clusters[i].IOIs + clusters[j].IOIs
                    clusters[i].mean_IOI = np.mean(clusters[i].IOIs)
                    clusters[j].IOIs = []  # delete cluster j
                    clusters[j].mean_IOI = -1
                    clusters[j].num = 0
                
        #clusters = clusters([clusters.num] != 0)
        clusters = list(filter(lambda x: x.num != 0, clusters))
        
    # calculate score
    if orig:
        # following the method defined in the origianl paper
        for i in range(len(clusters)):
            clusters[i].score = 0
            clusters[i].related_clusters = []
            for j in range(len(clusters)):
                for d in range(1,9):
                    if abs(clusters[i].mean_IOI - d*clusters[j].mean_IOI) < cluster_width:
                        # score added to the cluster with larger interval
                        clusters[i].score = clusters[i].score + clusters[j].num*cluster_score(d)
                        related = RelatedInfo()
                        related.d = d
                        related.interval = clusters[j].mean_IOI    # save the interval information
                        clusters[i].related_clusters.append(related)
                    
    else:
        # use our own score calculation arithmic
        for i in range(len(clusters)):
            clusters[i].score = 0
            clusters[i].related_clusters = []
            for j in range(len(clusters)):
                if (i!=j):
                    for d in range(1,9):
                        if abs(clusters[j].mean_IOI - d*clusters[i].mean_IOI) < cluster_width:
                            # i is the base tempo, j is the multiple of i, score of
                            # i will be higher, indicate j as the related cluster of i
                            clusters[i].score = clusters[i].score + clusters[j].num*cluster_score(d)   # add the score into the cluster with smaller interval
                            
                            # save information of the related cluster
                            related = RelatedInfo()
                            related.d = d
                            related.interval = clusters[j].mean_IOI    
                            clusters[i].related_clusters.append(related) 

    return clusters

def get_salience(event, mode):
    # calculate the salience of the event, based on given mode
    # Input:
    #         event: an event vector
    #               (1) - note start in beats
    #               (2) - note duration in beats
    #               (3) - channel
    #               (4) - midi pitch (60 --> C4 = middle C)
    #               (5) - velocity
    #               (6) - note start in seconds
    #               (7) - note duration in seconds
    #         mode: the function to calculate salient

    # some hyperparameters
    c1 = 300
    c2 = -4
    c3 = 1
    c4 = 84
    pmin = 48
    pmax = 72

    # read out note information
    d = event[6]    # duration in seconds
    p = event[3]    # pitch
    v = event[4]    # note velocity

    if p <= pmin:
        p = pmin
    elif p>= pmax:
        p = pmax

    if mode == 1:
        # linear mode
        salience = c1*d + c3*v # pitch does not reflect information
    elif mode == 2:
        # non-linear mode, used in our paper
        salience = d*v 
    elif mode ==3:
        # the original linear salience
        salience = c1*d + c2*p + c3*d
    elif mode ==4:
        # the original nonlinear salience
        salience = d*(c4 - p)*log(v)
    else:
        salience = v    # velocity only
    
    return salience

def agent_init(clusters, events, start_period, orig=False):
    # initialize agents based on given clusters and note events
    # Inputs:
    #         clusters: a struct array of all possible tempos
    #         events: note events matrix, defined in Christine's MIDI toolbox
    #         meaning of each column: 
    #           (1) - note start in beats
    #           (2) - note duration in beats
    #           (3) - channel
    #           (4) - midi pitch (60 --> C4 = middle C)
    #           (5) - velocity
    #           (6) - note start in seconds
    #           (7) - note duration in seconds
    #         start_period: start time of the first event
    # Outputs:
    #         agents: an array of generated beat agents
    # Reference:
    #         E. S. Christine, Midi Tools - File Exchange - MATLAB Central, 2019.

    agents = []
    salience_mode = 2  # should be 2
    for i in range(len(clusters)):
        for j in range(len(events)):
            if events[j][5] < start_period:
                agent = Agent()
                agent.beat_interval = clusters[i].mean_IOI
                agent.predict = events[j][5] + agent.beat_interval
                agent.hist = [events[j][5]]     # only save the start time
                agent.tempo_score = clusters[i].score
                if orig:
                    agent.score = get_salience(events[j][:], 3)    # use original salience calculation
                else:
                    agent.score = get_salience(events[j][:], salience_mode) + agent.tempo_score # use improved version
                agents.append(agent)
    
    #print(agents[-1])
    return agents

def beat_tracking_main(agents, events, orig=False):
    # main logic for agent selection
    # Inputs:
    #         agents: all possible agents returned by agent_init()
    #         events: note events matrix
    # Output:
    #         result: the agent with the highest score, it should contain most of the beats in its history

    if orig:
        # hyperparameters of the original version
        time_out = 20   # if the agent prediction varies too much to the correct next beat time, delete this agent
        outer_up = 0.056   # upper outer window boundary
        outer_low = 0.048 # lower outer window boundary

        inner_win = 0.040   # inner window boundary, original = 40ms
        correction_factor = 10  # not mentioned in the paper, set to the same value as ours
        tempo_tole = 0.01     # tempo threshold to merge two agents ==>> the resolution of the tempo
        phase_tole = 0.020    # phase difference threshold to merge two agents ==>> the resolution of the phase
        salience_ratio = 1     # original version, no this hyperparameter
        
    else:
        # hyperparameters of the improved version
        time_out = 20   # if the agent prediction varies too much to the correct next beat time, delete this agent
        outer_up = 0.1   # upper outer window boundary
        outer_low = 0.080 # lower outer window boundary

        # for strict outer window, add tempo score when calculating agent score
        # gives better result.

        inner_win = 0.050   # inner window boundary
        correction_factor = 10  # can be optimized
        tempo_tole = 0.02     # tempo threshold to merge two agents ==>> the resolution of the tempo
        phase_tole = 0.020    # phase difference threshold to merge two agents ==>> the resolution of the phase
        salience_ratio = 5

    removed_number = 0
    for i in range(len(events)):
        # remove duplicate agents, i.e. two agents having approximate phase and tempo
        # tempo tolerance: 10ms, phase tolerance: 20ms
        for m in range(len(agents)):
            for n in range(m,len(agents)):
                if m!=n:
                    if (abs(agents[m].beat_interval - agents[n].beat_interval) < tempo_tole) and (abs(agents[m].hist[-1] - agents[n].hist[-1]) < phase_tole):
                        # retain the agent with higher score
                        removed_number +=1
                        if agents[m].score < agents[n].score:
                            agents[m].beat_interval = -1
                            agents[m].score = 0
                        else:
                            agents[n].beat_interval = -1
                            agents[n].score = 0
        agents = list(filter(lambda x: x.beat_interval != -1, agents))

        j = 0
        new_agent = Agent()
        new_agent.beat_interval = -1
        
        while j < len(agents):
            if events[i][5] - agents[j].hist[-1] > time_out:
                agents[j].beat_interval = -1   # delete this agent
            else:
                while agents[j].predict + outer_up < events[i][5]:
                    agents[j].predict = agents[j].predict + agents[j].beat_interval
                
                if (agents[j].predict < events[i][5] + outer_up) and (agents[j].predict > events[i][5] - outer_low):
                    # lie within the outer window
                    if abs(agents[j].predict - events[i][5]) > inner_win:
                        # create new agent that does not accept the event as a
                        # beat time,as insurance against a wrong decision
                        new_agent.beat_interval = agents[j].beat_interval
                        new_agent.hist = agents[j].hist
                        new_agent.predict = agents[j].predict
                        new_agent.score = agents[j].score
                        new_agent.tempo_score = agents[j].tempo_score
                    
                    # the prediction matches the event with some tolerance
                    error = events[i][5] - agents[j].predict       # calculate absolute error
                    relative_error = abs(error)/(outer_low + outer_up) # calculate relative error
                    agents[j].beat_interval = agents[j].beat_interval + (error/correction_factor)    # adjust the tempo 
                    agents[j].hist.append(events[i][5])   # add this event to history
                    agents[j].predict = events[i][5] + agents[j].beat_interval
                    if orig:
                        agents[j].score = agents[j].score + (1 - relative_error)*get_salience(events[i][:], 4)
                    else:
                        agents[j].score = agents[j].score + agents[j].tempo_score + salience_ratio*(1 - relative_error)*get_salience(events[i][:], 2)
            
            j = j + 1
        
        # add newly created agents
        if new_agent.beat_interval != -1:
            agents.append(new_agent)
    agent_scores = []

    #print('number of agents: ',len(agents))
    for agent in agents:
        agent_scores.append(agent.score)

    if(len(agents)==0):
        return None
    
    best_score = max(agent_scores)
    
    for agent in agents:
        if(agent.score == best_score):
            result = agent # return agent with best score
            break

    return result

def parse_section(section,start,tempo,alpha,save_tempo=False):
    # The beat insertion and deletion step as described in our paper
    # Inputs:
    #         section: original section, only timestamp information used
    #         start: the start timestamp
    #         tempo: the estimated tempo
    #         alpha: allowed tolearance for checking quarter note beats
    #         save_tempo: add the estimated tempo as an extra column to the new
    # Outputs:
    #         new_section: a new sequence after insertion and deletion
    #         inserts: number of inserted notes
    #         deletes: number of deleted notes
    #         inserts and deletes are intermediate information that might be used to develop a fully automatic beat tracker
    
    
    #print('\n****************   ****************    ****************    ****************\n')
    
    inserts = 0
    deletes = 0

    if save_tempo:
        new_section = [start, tempo]
    else:
        new_section = start
    

    for k in range(len(section)):
        
        #print('\n****************\n')
        
        n = 1  # n start from 1
        while (section[k] - new_section[-1] > tempo * (n + alpha)):
            n = n + 1
        
        interval = 0  # the interval to insert artificial beats
        remove = False # remove this beat or not
        if section[k] - new_section[-1] < tempo*(n - alpha):
            # delete this note event
            deletes = deletes + 1      # record total number of deletes
            interval = tempo
            remove = True
        else:
            # retain this note event ==>> self-correction
            interval = (section[k] - new_section[-1])/n
        
        # insert n-1 beats
        predict = []
        for i in range(1,n):
            predict.append(new_section[-1] + i*interval)
        
        """ print('\npredict:')
        print('length: ',len(predict)) """
        
        new_section = new_section+predict
        
        
        """ print('\nnew_section:')
        print('length: ',len(new_section))
        print('first: ',new_section[0])
        print('last: ',new_section[-1]) """
        
        inserts = inserts + n - 1
        
        if not remove:
            new_section.append(section[k])

        """ print('\nnew_section_2:')
        print('length: ',len(new_section))
        print('first: ',new_section[0])
        print('last: ',new_section[-1]) """

    return new_section, inserts, deletes

def beat_from_events(therapist_notes):
    therapist_matrix = therapist_notes
    #print('Cleaning the therapist notes')
    
    cleaned_therapist = clean_notes(therapist_matrix, 0.25, 3) # step 1: preprocessing
    
    #print('Cleaning has finished, there are this many onsets left: ',len(cleaned_therapist))

    """ print('\ncleaned therapist:')
    print('length: ',len(cleaned_therapist))
    print('first: ',cleaned_therapist[0])
    print('last: ',cleaned_therapist[-1]) """

    # hyperparameters
    section_length = 20    # the length of a section 
    qnote_thres = 0.7    # threshold to determine whether a beat interval is 8th note or smaller
    alpha = 0.20       # tolerance, used in beat insertion and deletion

    tempos = []            # a list of estimated tempo
    best_agents = []       # a list of best agents
    total_insertion = 0
    total_deletion = 0
    new_matrix = [cleaned_therapist[0,5]]  # initialize new matrix(automatic result)

    number_of_loops = int(len(cleaned_therapist)/section_length)
    loop_number = 0

    for i in range(0,len(cleaned_therapist),section_length):
        #print('Starting the loop:'+str(loop_number)+'/'+str(number_of_loops))
        upper_bound = i + section_length
        if upper_bound > len(cleaned_therapist):
            upper_bound = len(cleaned_therapist)    # avoid index out of boundary

        section = cleaned_therapist[i:upper_bound][:]  # pick out a small section for analysis
        tempo = 0
        if len(section) > 2:
            clusters = get_clusters(section)   # step 2: cluster notes
            """ print('\nget_clusters:')
            print('length: ',len(clusters))
            print('first: ',clusters[0])
            print('last: ',clusters[-1]) """
            agents = agent_init(clusters, section, section[-1][5])    # step 3: agent initialization
            """ print('\nagent_init:')
            print('length: ',len(agents))
            print('first: ',agents[0])
            print('last: ',agents[-1]) """
            best_agent = beat_tracking_main(agents, section)   # step 4: agent selection
            """ print('\nbest_agent:')
            print(best_agent) """
            if(best_agent == None): continue 

            tempo = best_agent.beat_interval
            while tempo < qnote_thres:
                #print(tempo)
                tempo = tempo*2

            tempos.append(tempo)
            best_agents.append(best_agent)
        elif(len(tempos)!=0):
            # section too small, use previous tempo, skip step 2-4
            tempo = tempos[-1]
            tempos.append(tempo)
            best_agent = best_agents[-1]
            best_agent.predict = -1    # indicate this is an exception
            best_agents.append(best_agent)
        
        """ print('\nbest agents 2:')
        print('length: ',len(best_agents))
        print('first: ',best_agents[0])
        print('last: ',best_agents[-1]) """

        # beat insertion and deletion.
        # either the agent history or the original sequence can be used, the difference is very small. but the agent
        # selecetion step is still needed since it finds out the best tempo and fine tune it
        """ print('\nnew_matrix: ')
        print('length: ',len(new_matrix))
        print('first: ',new_matrix[0])
        print('last: ',new_matrix[-1]) """

        """ print('\nsection: ')
        print('length: ',len(section))
        print('first: ',section[0])
        print('last: ',section[-1]) """
        
        #print(tempo)
        new_section, inserts, deletes = parse_section(section[:,5], [new_matrix[-1]], tempo, alpha)
        
        """ print('\nnew_section:')
        print('length: ',len(new_section))
        print('first: ',new_section[0])
        print('last: ',new_section[-1]) """

        total_deletion = total_deletion + deletes
        total_insertion = total_insertion + inserts
        
        # append new section to the new matrix
        # Since the first element in the new_section is always the last element
        # in new_matrix (see parse_section()), append from the second element
        if len(new_section) > 1:          # avoid index out of boundary
            
            """ print("\n******************\n")
            print('new_matrix initially: ')
            print(new_matrix)
            print('\nnew_section[1:]: ')
            print(new_section[1:]) """

            new_matrix = new_matrix + new_section[1:]

        loop_number+=1

    #print('returning from MAIBT')
    return new_matrix


