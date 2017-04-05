%{




this code is not giving correct answer i have tried to tune its parameter ,
use differnt function like tan, sigmoid .. etc , converted {1,0} to {-1,1}
rescaling and all but nothing seemed to work it either predicts all -1 or 1




%}
% READ DATA FROM XLS FILE

train_A = xlsread('train.xls',1,'B2:H4000');        %train features for A
test_A = xlsread('train.xls',2,'B1:H500');          %test features for A                  
train_B = xlsread('train.xls',1,'I2:O4000');        %tarin features for B 
test_B = xlsread('train.xls',2,'I1:O500');          %test features for B 

train_C = xlsread('train.xls',1,'A2:A4000');        %train feature for Choice 
test_C = xlsread('train.xls',2,'A1:A500');

maximumA = max(train_A);
minimumA = min(train_A);
maximumB = max(train_B);
minimumB = min(train_B);

for i = 1:7                                    %rescaling features 
train_A(:,i) = (train_A(:,i)-minimumA(i))/(maximumA(i)-minimumA(i));
train_B(:,i) = (train_B(:,i)-minimumB(i))/(maximumB(i)-minimumB(i));
test_A(:,i)  = (test_A(:,i)-minimumA(i))/(maximumA(i)-minimumA(i));
test_B(:,i)  = (test_B(:,i)-minimumB(i))/(maximumB(i)-minimumB(i));
end

%CREATING NEW ARRAYS FOR DATA
[m,n] = size(train_A);
output = zeros(m,1);
%delta = zeros(m,1);
train_C = 2.*train_C-1;   %shifting to -1 and 1
test_C = 2.*test_C-1;     %shifting to -1 and 1

% Initialize the bias
bias =[1,1];
% Learning coefficient
coeff = .50;
% Calculate weights randomly using seed.


%WEIGHTS INITIALIZATION 
rand('state',sum(100*clock));
weights =(25/8).* rand(3,1);             %weight for bias 
wght_A_1 =(25/8).*rand(7,1);                   %weight for hidden layers's FIRST node input A
wght_B_1 =(25/8).*rand(7,1);                   %weight for hidden layers's FIRST node input B
wght_A_2 =(25/8).*rand(7,1);                   %weight for hidden layers's SECOND node input A
wght_B_2 =(25/8).*rand(7,1);                   %weight for hidden layers's SECOND node input B
wght_1 =(25/8).*rand();                        %weight for output layer's input from node 1
wght_2 =(25/8).*rand();                        %weight for output layer's input from node 2



%TRAINING THE NEURAL NET

   for j = 1:m

      % Hidden layer
      H1 = bias(1,1)*weights(1,1)+ train_A(j,:)*wght_A_1+ train_B(j,:)*wght_B_1;    %hidden layer 1 and its output 
      H2 = bias(1,1)*weights(2,1)+ train_A(j,:)*wght_A_2 +train_A(j,:)*wght_B_2;       %hidden layer2 and its output
      x2 = [(atan(H1)*2)/pi,(atan(H2)*2)/pi];                                           %output from hidden layer in sigmoid function
      %x2 = [(H1),(H2)];
      % Output layer
      x3 = bias(1,2)*weights(3,1)+ x2(1,1)*wght_1+ x2(1,2)*wght_2
      x3 = (atan(x3)*2)/pi
      
      output(j) = 1/((atan(x3)*2)/pi);
    %{
      if(output(j) >= 0.5)
          output(j) = 1;
      else 
          output(j) = -1;
      end
%}
      % Adjust delta values of weights
      delta3_1 = (1-abs(output(j)))*output(j)*(train_C(j)-output(j));
      
      % Propagate the delta backwards into hidden layers
      delta2_1 = x2(1,1)*(1-x2(1,1))*wght_1*delta3_1;
      delta2_2 = x2(1,2)*(1-x2(1,2))*wght_2*delta3_1;
      
      % Update weights simultaneously 
            weights(1,1) = weights(1,1) + coeff*bias(1,1)*delta2_1;
            weights(2,1) = weights(2,1) + coeff*bias(1,2)*delta2_2;
            weights(3,1) = weights(3,1) + coeff*bias(1,2)*delta3_1;

            wght_A_1 =  wght_A_1 + transpose(coeff*train_A(j,:)*delta2_1);
            wght_B_1 =  wght_B_1 + transpose(coeff*train_B(j,:)*delta2_1);
            wght_A_2 =  wght_A_2 + transpose(coeff*train_A(j,:)*delta2_2);
            wght_B_2 =  wght_B_2 + transpose(coeff*train_B(j,:)*delta2_2);
            
            wght_1 =  wght_1 + coeff*x2(1,1)*delta3_1;
            wght_2 = wght_2 + coeff*x2(1,2)*delta3_1;
   end 
   %testing the data 
   [a,b] = size(test_C);
   out = zeros(a,b);
   count = 0;
   
   
  for j = 1:a
      % Hidden layer
      H1 = bias(1,1)*weights(1,1)+ test_A(j,:)*wght_A_1+ test_B(j,:)*wght_B_1;    %hidden layer 1 and its output 
      H2 = bias(1,1)*weights(2,1)+ test_A(j,:)*wght_A_2 +test_A(j,:)*wght_B_2;       %hidden layer2 and its output 
      
      X2 = [1/(1+exp(-H1)),1/(1+exp(-H2))];                                           %output from hidden layer in sigmoid function

      % Output layer
      X3 = bias(1,2)*weights(3,1)+ X2(1,1)*wght_1+ X2(1,2)*wght_2;
      out(j) = 1/(1+exp(-X3));
      if(out(j) >= 0.5)
          out(j) = 1;
      else 
          out(j) = -1;
      end
      if(out(j) == test_C(j))
          count = count+1;
      end 
  end
  count 
  a
  s = (count/a)*100
