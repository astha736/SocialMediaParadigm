% READ DATA FROM XLS FILE 

A = xlsread('train.xls',1,'B2:H4000');        %train features for A
test_A = xlsread('train.xls',2,'B1:H500');    %test features for A                  
B = xlsread('train.xls',1,'I2:O4000');        %tarin features for B 
test_B = xlsread('train.xls',2,'I1:O500');    %test features for B 

C = xlsread('train.xls',1,'A2:A4000');        %train feature for Choice 
test_C = xlsread('train.xls',2,'A1:A500');    %test feature for Choice 

test_main_A  = xlsread('test.xls',1,'A2:G1002');
test_main_B  = xlsread('test.xls',1,'H2:N1002');

%CREATING NEW ARRAYS FOR PRE-PROCESSED DATA

[m,n] = size(A);
output = zeros(m,1);
delta = zeros(m,1);

%PRE-PROCESSING DATA : SUBTRACTING A FROM B: NEW FEATURES 

maximumA = max(A);
minimumA = min(A);
maximumB = max(B);
minimumB = min(B);

for i = 1:7                                    %rescaling features 
A(:,i) = (A(:,i)-minimumA(i))/(maximumA(i)-minimumA(i));
B(:,i) = (B(:,i)-minimumB(i))/(maximumB(i)-minimumB(i));
end                               

bias = 1;
coeff = .5;
biasWeight = 1;

weightA = ones(7,1);
weightB = ones(7,1);


for i = 1:m                                                      %dealing with each point(rows) one by one
          y = bias*biasWeight+(A(i,:))*weightA+B(i,:)*weightB;   %multiplying vectors 
          output(i) = 1/(1+exp(-y));                             %sigmoid function
          if (output(i) >= 0.5)
              output(i) = 1;
          else 
              output(i) = 0;
          end 
          delta(i) = C(i) - output(i);
          
          biasWeight = biasWeight + coeff*bias*delta(i);
          weightA = weightA + transpose(coeff*A(i,:)*delta(i));
          weightB = weightB + transpose(coeff*B(i,:)*delta(i));
end
%initialize variables
count1 = 0;
count2 = 0;
[n,o] = size(test_A);
delta = zeros(n,1);
output = zeros(n,1);
for i = 1:n                                                      %dealing with each point(rows) one by one
          y = bias*biasWeight+(test_A(i,:))*weightA+test_B(i,:)*weightB;   %multiplying vectors 
          output(i) = 1/(1+exp(-y));                             %sigmoid function
          if (output(i) >= 0.5)
              output(i) = 1;
          else 
              output(i) = 0;
          end 
          delta(i) = test_C(i) - output(i);
          if (delta(i) ~= 0)
              count1 = count1+1;
          else 
              count2 = count2+1;
          end
          biasWeight = biasWeight + coeff*bias*delta(i);
          weightA = weightA + transpose(coeff*test_A(i,:)*delta(i));
          weightB = weightB + transpose(coeff*test_B(i,:)*delta(i));
end

error_test = (count1/n)*100
accuracy_test = (count2/n)*100

%initialize variables
count1 = 0;
count2 = 0;
[n,o] = size(test_main_A);
delta = zeros(n,1);
output = zeros(n,1);
%pedicting output for test_main_A and test_main_B 
for i = 1:n                                                      %dealing with each point(rows) one by one
          y = bias*biasWeight+(test_main_A(i,:))*weightA+test_main_B(i,:)*weightB;   %multiplying vectors 
          output(i) = 1/(1+exp(-y));                             %sigmoid function
          if (output(i) >= 0.5)
              output(i) = 1;
          else 
              output(i) = 0;
          end 

end
%write output to XLS file 
xlswrite('predictSNN',output,1);