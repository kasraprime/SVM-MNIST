%Load the dataset using your MNIST_digit_data.m code so that you can load dataset however you want.
load MNIST_digit_data

train_scale = size(images_train);
test_scale = size(images_test);

%Normalizing images
images_test=images_test/255;
images_train=images_train/255;

newimage_train=zeros(1000,784);
newimage_test=zeros(1000,784);
newlabel_train=zeros(1000,1);
newlabel_test=zeros(1000,1);

index=1;
onecounter=0;
sixcounter=0;
%selecting 1000 train images of 1 and 6
for i=1:train_scale
    if (labels_train(i)==1 && onecounter<500)
        onecounter=onecounter+1;
        newimage_train(index,:)=images_train(i,:);
        newlabel_train(index,:)=labels_train(i,:);
        index=index+1;
    elseif (labels_train(i)==6 && sixcounter<500)        
        sixcounter=sixcounter+1;
        newimage_train(index,:)=images_train(i,:);
        newlabel_train(index,:)=labels_train(i,:);
        index=index+1;    
    end
    
    if index==1001
        break
    end
end

%selecting 1000 test images of 1 and 6 
index=1;
onecounter=0;
sixcounter=0;
for i=1:test_scale
    if (labels_test(i)==1 && onecounter<500)
        onecounter=onecounter+1;
        newimage_test(index,:)=images_test(i,:);
        newlabel_test(index,:)=labels_test(i,:);
        index=index+1;
    elseif (labels_test(i)==6 && sixcounter<500)
        sixcounter=sixcounter+1;
        newimage_test(index,:)=images_test(i,:);
        newlabel_test(index,:)=labels_test(i,:);
        index=index+1;        
    end
    
    if index==1001
        break
    end
end



%SVM algorithm

%training phase

weights=randn(1,784);
y=zeros(1000,1);
%defining classes
for j=1:1000
    if newlabel_train(j)==6
        y(j)=-1;
    else
        y(j)=1;
    end       
end
eachiterationweight=zeros(100,784);

%lambda is our hyper-parameter in regularization term
lambda=0.00001;
%learningrate is our learning rate


for m=1:100
    learningrate=1/m;
    for i=1:1000                
        if ((y(i)*(newimage_train(i,:)*weights'))<=1)
            weights=weights+learningrate*(y(i)*newimage_train(i,:)-(lambda*weights));
        %else
            %weights=weights-lambda*weights;            
        end                                              
    end
    eachiterationweight(m,:)=weights;
end


%SVM test phase

%defining classes
ytest=zeros(1000,1);
for j=1:1000
    if newlabel_test(j)==6
        ytest(j)=-1;
    else
        ytest(j)=1;
    end       
end

%plot the accuracy on the test set w.r.t the number of iterations

accuracytemp=zeros(100,1);
iter=zeros(100,1);

for m=1:100
    weights=eachiterationweight(m,:);
    count=0;  
    for i=1:1000         
        if newimage_test(i,:)*weights'>0        
            yhat=1;
        else        
            yhat=-1;
        end
        if ytest(i)==yhat
            count=count+1;                            
        end       
    end
    accuracytemp(m)=(count/1000)*100;
    iter(m)=m;
end

figure(1);
xlabel('number of iterations') 
ylabel('accuracy') 
hold on
plot(iter,accuracytemp)
accuracy=count/1000;

