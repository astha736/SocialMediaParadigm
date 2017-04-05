% READ DATA FROM XLS FILE
train = xlsread('train.xls',1,'B2:O4000');        %read feature vector for trainning data  
test = xlsread('train.xls',2,'B1:O500');          %read features vector for test data   

train_C = xlsread('train.xls',1,'A2:A4000');        %read Choice for tainning data  
test_C = xlsread('train.xls',2,'A1:A500');          %read Choice for test Data 


test_main  = xlsread('test.xls',1,'A2:N1002');      %features for  test data  to be PREDICTED


SVMStruct = svmtrain(train,train_C);                  %initialize SVMStruct with help of train data 
Group = svmclassify(SVMStruct,test);                  %testing classificatiob using testing data 

count = 0;

%TESTING THE ACCURACY OF CLASSIFICATION
for i = 1:500
    if(Group(i)~= test_C)
    else 
         count = count +1;
    end 
end 

accuracy = (count/500)*100

group = svmclassify(SVMStruct,test_main);         %predicting ouput for the test_main data read from test.xls file 
%write output to XLS file 
xlswrite('predictSVM',group,1);