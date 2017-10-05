import numpy as np
import pandas as pd
import talib
from sys import argv

# read data
P = pd.read_csv(argv[1])
P.dropna(inplace=True)

# general constants
train_ratio = 0.7
nIter = 100
epsilon = 1e-10
cci_period = 14
c     = 0.0
alpha = 0.4
gamma = 0.9

# state defines
delta = [-1,0,1]
state_cut = [i for i in range(-1000, 1000+1, 100)]

# calculate index (cci)
P_cci = talib.CCI( P['HIGH'],P['LOW'],P['CLOSE'], timeperiod=cci_period )
P_cci = [x for x in P_cci if x is not np.isnan(x)]
n_P_cci = len( P_cci )

# initialize states
P = P[ -n_P_cci : ]
state = np.zeros( (1, n_P_cci) )

for i_P_cci in range(n_P_cci):
    if P_cci[i_P_cci] < state_cut[0]:
        state[i_P_cci] = 0
    elif P_cci[i_P_cci] < state_cut[-1]:
        for i_state in range(1 , len(state_cut)):
            if P_cci[i_P_cci] < state_cut[i_state] and P_cci[i_P_cci] >= state_cut[i_state-1]: 
                state[i_P_cci] = i_state
    else:
       state[i_P_cci] = len(state_cut)

# slice training & testing datas
P_length = len( P )
train_amount = int(np.floor( P_length * train_ratio ))

P_train = P[ : train_amount ]
P_test  = P[ train_amount : ]
state_train = state[ : train_amount ]
state_test  = state[ train_amount : ]

# training constants
nDelta = len( delta )
T      = len( P_train )
nState = len( state_cut )
Q_t    = np.zeros( (nState , nDelta) )
action = np.zeros( (1, T) )

update_Q_t = np.zeros( (1, nDelta) )
 
# value function
def valueFunction(pre_P, curr_P, pre_delta, curr_delta, C):
    return 0.0
    
# update Q-value
def updateQValue(Q_t_value, next_Q_t, next_V_t, alpha, gamma):
    pass
    


for t in range(T,1,-1):
    print('t: %d/%d'%(T-t,T))
    nowState = state_train[t]
    
    # Value Calculation
    V_t = np.zeros(nDelta, nDelta)
    for nowDelta in range(nDelta):
        for lastDelta in range(nDelta):
            delta_t = delta[ nowDelta ]
            delta_t_1 = delta[ lastDelta ]
            V_t[ lastDelta, nowDelta ] = valueFunction( P_train[t-1], P_train[t], delta_t_1, delta_t, c)
    
    for nowDelta in range(nDelta):
        # update Q value
        if t == T
            Q_t[ :, nowDelta ] = max( V_t[ :, nowDelta ] ) * np.ones( (1, nState) )
        else
            Q_t_diff = 1
            #V_t_next = all_next_V_t[ nowDelta ]
            V_t_next = V_t[ nowDelta ]
            while Q_t_diff > epsilon:
                Q_t[ nowState, nowDelta ] = update_Q_t[nowDelta]
                update_Q_t[nowDelta] = updateQValue( Q_t( nowState, nowDelta ), all_next_Q_t, V_t_next, alpha, gamma )
                Q_t_diff = abs( Q_t[ nowState, nowDelta ] - update_Q_t[nowDelta] )
            
    all_next_Q_t = Q_t[ nowState, : ]
    action_candidate = find( all_next_Q_t == max(all_next_Q_t) );
    action(t) = action_candidate(randi([1 length(action_candidate)],1,1));
    all_next_V_t = V_t( : , action(t) );
end

stateAction = zeros(1, nState);
for i_state = 1 :nState
    action_candidate = find( Q_t(i_state,:) == max(Q_t(i_state,:)) );
    stateAction(i_state) = action_candidate(randi([1 length(action_candidate)],1,1));
end

train_reward = zeros(1, length( P_train ) );
for t = 1 : length( P_train )
    action(t) = stateAction(state_train(t));
    if t > 1
        train_reward(t) = train_reward(t-1) + rewardFunction( P_train(t-1), P_train(t), action(t-1), action(t), c);
    end
end



test_reward = zeros(1, length( P_test ) );
test_action = zeros(1, length( P_test ));
for t = 1 : length( P_test )
    test_action(t) = stateAction(state_test(t));
    if t > 1
        test_reward(t) = test_reward(t-1) + rewardFunction( P_train(t-1), P_train(t), test_action(t-1), test_action(t), c);
    end
end








