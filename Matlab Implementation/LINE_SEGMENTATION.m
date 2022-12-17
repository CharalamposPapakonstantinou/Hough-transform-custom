function [ BL ] = LINE_SEGMENTATION( ROW,im )
% ?????????? ????????????????, ece7888

lamda=(ROW(4)-ROW(2))/(ROW(3)-ROW(1)); % ??????????? ?????? ?
prevLINE=0;  % flag ??? ?? ?????????? ?? ???????? ???? ???? ?????? ? ???
BL=[];     
i=0;

if abs(lamda)<=1
    for x_index=1:size(im,1)  % ??? ???? x 
     y=round(lamda*x_index-lamda*ROW(1)+ROW(2));
     x=x_index;
     if y>=1 && y<=size(im,2) % ?? ?? y ????? ???? ??? ???? ??? ???????
         if im(y,x)==1        % ??? ??????? ?????? ???? ??????
             if prevLINE==1   % ??? ??????? ???? ???? ?????? ?????? ???? ??? ??????? ?????????????
                 BL(i,3)=x;
                 BL(i,4)=y;
             else             % ???? ???? ??? ????????????? ?????? ???? ???????????? ???? ?????,
                 i=i+1;       % ???? ?????? ?????? ???? ?????? BL ??? ????????? ?? x,y ?? ???????
                 BL(i,1)=x;   % ??? ??????? ????????????? 
                 BL(i,2)=y;    
                 BL(i,3)=x;
                 BL(i,4)=y;
             end
              prevLINE=1;     % prevLINE=1; ---> ????? ???? ??? ??????
         else 
             prevLINE=0;
         end
     end
    end

else
    for y_index=1:size(im,2)  % ??? ???? y 
     x=round((1.0/lamda)*(y_index-ROW(2))+ROW(1));
     y=y_index; 
     if x>=1 && x<=size(im,1) % ?? ?? x ????? ???? ??? ???? ??? ???????
         if im(x,y)==1        % ??? ??????? ?????? ???? ??????
             if prevLINE==1   % ??? ??????? ???? ???? ?????? ?????? ???? ??? ??????? ?????????????
                 BL(i,3)=x;
                 BL(i,4)=y;
             else             % ???? ???? ??? ????????????? ?????? ???? ???????????? ???? ?????,
                 i=i+1;       % ???? ?????? ?????? ???? ?????? BL ??? ????????? ?? x,y ?? ???????
                 BL(i,1)=x;   % ??? ??????? ????????????? 
                 BL(i,2)=y;    
                 BL(i,3)=x;
                 BL(i,4)=y;
             end
              prevLINE=1;     % prevLINE=1; ---> ????? ???? ??? ??????
         else 
             prevLINE=0;
         end
     end
    end
end


end

