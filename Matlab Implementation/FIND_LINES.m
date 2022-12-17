function [L] = FIND_LINES( X,Y,threshold,P,th,r,im)
% ?????????? ????????????????, ece7888

TPx=[];
TPy=[];
m=1;

for i=1:size(P,1)                     % ?????? ??? P
   for j=1:size(P,2)
       if P(i,j)>threshold            % ???? ??? ??? ????????? ??????? ?????? ?? ???????? ??? ?? ???????????
           Px=th(j);                  % ?
           Py=i-round(size(P,1)/2);   % r
           hold on
%            plot(Px,Py,'rx')           % ?????????? ?? 'x' ??? ?????? r,? ??? ??????????? ??????
           
           TPx=[TPx Px];              % ?????????? ??? ? ??? r ????? TPx,TPy
           TPy=[TPy Py];
           invtan(m)=cos(Px)/sin(Px); % ?????????? ??????? ????? ??????? ??? ????? ????????
           Tan(m)=sin(Px)/cos(Px);
           Sin(m)=sin(Px);
           Cos(m)=cos(Px);
           m=m+1;
       end
   end
end

% ?????????? ??????? 
figurem
subplot(1,2,1)
imshow(im)
xlabel('X');
ylabel('Y');
subplot(1,2,2)
imshow(im)
xlabel('X');
ylabel('Y');
title('Detected Lines')
hold on

x=[1:X]; 
 for  m=1:size(Sin,2)             % size(Sin,2)= ??????? ??????? ??? ????????????
    y=TPy(m)/Sin(m)-x'*invtan(m); % ??????????? ??????? 
    if Sin(m)==0                  % ??????? ?? ? ?????? ????? ??????
       plot([1 X],[TPy(m) TPy(m)])
    else
       plot(y,x); 
    end
    hold on
 end

% ?????? ??????? ??? ????? ? ?????? ?????? ?? ???? ??? ???????
edge1=[];
edge2=[];
for  m=1:size(Sin,2) % size(Sin,2)= ??????? ??????? ??? ????????????
    x=1;             % ??? x=1, ????????? ?? y
    y=TPy(m)/Sin(m)-x'*invtan(m);
    if y<1           % ???? ?? y ????? ????? ??? ????? ??? ??????? 
        y=1;         % ???? ???? y=?????? ???? ??? ????????? ?? x
        x=(TPy(m)/Sin(m))/invtan(m)-y/invtan(m);
    elseif y>Y
        y=Y;
        x=(TPy(m)/Sin(m))/invtan(m)-y/invtan(m);     
    end
    if Sin(m)==0     % ??????? ?? ? ?????? ????? ??????
        y=1;
        x=TPy(m);
    end
    edge1=[edge1;[y x]];     % ????????? ?? ????? ???? ?????? edge1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x=X;             % ??? x=?, ????????? ?? y ??? ????? ?? ????????
    y=TPy(m)/Sin(m)-x'*invtan(m);
    if y<1 
        y=1;
        x=(TPy(m)/Sin(m))/invtan(m)-y/invtan(m);
    
    elseif y>Y
        y=Y;
        x=(TPy(m)/Sin(m))/invtan(m)-y/invtan(m);
    end
    if Sin(m)==0 
        y=X;
        x=TPy(m);
    end
    edge2=[edge2;[y x]];    % ????????? ?? ????? ???? ?????? edge2
end

edge=[edge1 edge2];

L=round(edge);              % ????????????? ??? ?????? edge ??? ?? ??? ?????????
end

