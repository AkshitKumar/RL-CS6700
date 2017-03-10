% Assume there are three arms - A, B, C
% They are all Gaussian Distributions with mean 1, 3 and 5 and variance 1
% Being greedy with action selection
q_estimate = zeros(3,1);
count = 0;
actions = zeros(3,1);
% Play each arm once
actions = [normrnd(1,1) ; normrnd(3,1) ; normrnd(5,1)];
q_estimate = actions;
best_arm_estimate = max(actions);
regret1 = 0;
regret2 = 0;
for i = 1:100000
    [best,index] = max(q_estimate);
    actions = [normrnd(1,1) ; normrnd(3,1) ; normrnd(5,1)];
    best_arm_atm = max(actions);
    regret1 = regret1 + (best_arm_atm - actions(index));
    q_estimate = ((q_estimate * i) + actions)/(i+1); 
end

for i = 1:100000
    [minimum,index] = min((best_arm_estimate * ones(3,1) - q_estimate));
    actions = [normrnd(1,1) ; normrnd(3,1) ; normrnd(5,1)];
    best_arm_atm = max(actions);
    regret2 = regret2 + (best_arm_atm - actions(index));
    q_estimate = ((q_estimate * i) + actions)/(i+1); 
    %best_arm_estimate = best_arm_estimate + (0.3 * (best_arm_atm - best_arm_estimate));
    best_arm_estimate = ((best_arm_estimate * i) + best_arm_atm)/(i+1);
end

fprintf('Regret being greedy : %d\n', regret1);
fprintf('Regret following algo : %d\n', regret2);
    